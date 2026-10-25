// The ARM guest's ops for the x86-64 native backend: data processing with and without flags
// (the host's flags are read back into the guest's N, Z, C, V bytes), the conditional branch
// and conditional exit, the interworking branch, register-amount shifts and LDM / STM.
// Anything else runs through its threaded handler.
#include "native_cond_x64.h"

namespace jit {

#ifdef JIT_HAVE_NATIVE

using namespace x64;

int guest_native_cells(const Op& o) {
    switch (o.kind) {
        case OpKind::A32Jcc: return 2;
        default: return 0;
    }
}

bool guest_native_emit(NativeGen& g, const Op& o) {
    Asm& a = g.a;
    const GuestSlots& s = g.s;
    const int insns = g.insns;
    const uint64_t rip_here = g.start + o.pc;
    switch (o.kind) {
    case OpKind::A32Dp: case OpKind::A32DpI: {
        const unsigned op = o.cc;
        const bool fl = (o.flags & kFlagEager) != 0;
        const bool imm = o.kind == OpKind::A32DpI;
        // operands: eax = Rn, ecx = the second operand (swapped for the reverse subtractions)
        const bool reverse = op == 3 || op == 7;
        if (imm) {
            g.ld_sz(RAX, o.a, 4);
            a.movi(RCX, static_cast<int64_t>(static_cast<uint32_t>(o.imm)));
        } else {
            g.ld_sz(RAX, o.a, 4);
            g.ld_sz(RCX, o.b, 4);
        }
        if (reverse) { a.mov_rr32(RDX, RAX); a.mov_rr32(RAX, RCX); a.mov_rr32(RCX, RDX); }
        bool arith = false, sub = false;
        switch (op) {
            case 0: case 8: a.alu_rr(AND, false, RAX, RCX); break;                   // AND, TST
            case 1: case 9: a.alu_rr(XOR, false, RAX, RCX); break;                   // EOR, TEQ
            case 12: a.alu_rr(OR, false, RAX, RCX); break;                           // ORR
            case 13: a.mov_rr32(RAX, RCX); a.test_rr(false, RAX, RAX); break;        // MOV
            case 14: a.not_r(false, RCX); a.alu_rr(AND, false, RAX, RCX); break;     // BIC
            case 15: a.mov_rr32(RAX, RCX); a.not_r(false, RAX); a.test_rr(false, RAX, RAX); break;   // MVN
            case 16: a.not_r(false, RCX); a.alu_rr(OR, false, RAX, RCX); break;      // ORN
            case 2: case 10: case 3: a.alu_rr(SUB, false, RAX, RCX); arith = true; sub = true; break;   // SUB, CMP, RSB
            case 4: case 11: a.alu_rr(ADD, false, RAX, RCX); arith = true; break;    // ADD, CMN
            case 5:                                                                  // ADC: CF = the guest carry
                a.load8u(RDX, RBX, s.fl_c); a.alu_ri(ADD, false, RDX, -1);
                a.alu_rr(ADC, false, RAX, RCX); arith = true;
                break;
            case 6: case 7:                                                          // SBC, RSC: CF = the guest borrow
                a.load8u(RDX, RBX, s.fl_c); a.alu_ri(XOR, false, RDX, 1); a.alu_ri(ADD, false, RDX, -1);
                a.alu_rr(SBB, false, RAX, RCX); arith = true; sub = true;
                break;
            default: return false;
        }
        if (fl) {
            a.setcc(S, RDX);  a.store8(RBX, s.fl_n, RDX);
            a.setcc(E, RDX);  a.store8(RBX, s.fl_z, RDX);
            if (arith) {
                a.setcc(sub ? AE : B, RDX); a.store8(RBX, s.fl_c, RDX);
                a.setcc(O, RDX); a.store8(RBX, s.fl_v, RDX);
            } else if (o.imm2 >= 0) {
                a.movi(RDX, o.imm2 ? 1 : 0); a.store8(RBX, s.fl_c, RDX);
            }
        }
        if (!(op >= 8 && op <= 11)) g.st(o.d, RAX);
        return true;
    }
    case OpKind::A32Jcc: {
        const int hcc = arm_cond(g, o.cc);
        if (hcc < 0) { g.exit_static(static_cast<uint64_t>(o.imm), insns); g.next_cell(); return true; }
        Label taken;
        a.jcc(hcc, taken);
        g.exit_static(static_cast<uint64_t>(o.imm2), insns);
        a.bind(taken);
        g.exit_static(static_cast<uint64_t>(o.imm), insns);
        return true;
    }
    case OpKind::A32ExitIf: {
        const int hcc = arm_cond(g, o.cc);
        if (hcc < 0) return true;
        Label ok;
        a.jcc(hcc, ok);
        g.st_imm(s.rip, o.imm);
        g.leave_with(o.ni);
        a.bind(ok);
        return true;
    }
    case OpKind::A32Bx: {
        g.ld(RAX, o.a);
        Label arm;
        a.test_ri(false, RAX, 1);
        a.jcc(E, arm);
        a.alu_ri(AND, false, RAX, -2);                 // Thumb: bit 0 is not part of the address
        g.st(s.rip, RAX);
        a.movi(RDX, 1); a.store8(RBX, s.thumb, RDX);
        g.leave_with(insns);
        a.bind(arm);
        a.alu_ri(AND, false, RAX, -4);
        g.st(s.rip, RAX);
        a.movi(RDX, 0); a.store8(RBX, s.thumb, RDX);
        g.leave_with(insns);
        return true;
    }
    case OpKind::A32Shift: {
        // The amount is the low byte of b; 32 or more is not the same as the amount modulo 32.
        g.ld_sz(RAX, o.a, 4);
        g.ld_sz(RCX, o.b, 1);
        const int type = o.cc;
        if (type == 2) {                               // ASR: an amount of 32 or more shifts in the sign
            a.movi(RDX, 31);
            a.alu_ri(CMP, false, RCX, 32);
            a.cmov(AE, false, RCX, RDX);
            a.shift_cl(SAR, false, RAX);
        } else if (type == 3) {
            a.shift_cl(ROR, false, RAX);
        } else {
            a.shift_cl(type == 0 ? SHL : SHR, false, RAX);
            a.movi(RDX, 0);
            a.alu_ri(CMP, false, RCX, 32);
            a.cmov(AE, false, RAX, RDX);
        }
        g.st(o.d, RAX);
        return true;
    }
    case OpKind::A32Ldm: case OpKind::A32Stm: {
        const uint32_t list = static_cast<uint32_t>(o.imm);
        int64_t off = o.c;
        const bool load = o.kind == OpKind::A32Ldm;
        for (unsigned i = 0; i < 16; ++i) {
            if (!(list & (1u << i))) continue;
            g.ld_sz(RAX, o.a, 4);
            a.alu_ri(ADD, false, RAX, static_cast<int32_t>(off));   // 32-bit wraparound
            if (load) {
                g.mem_read(4, rip_here);
                g.st(i == 15 ? o.d : static_cast<int>(i) * 8, RAX);
            } else {
                g.ld(R9, static_cast<int>(i) * 8);
                g.mem_write(4, rip_here);
            }
            off += 4;
        }
        if (!load && (o.flags & kFlagChk)) g.chk_flush(static_cast<uint64_t>(o.imm2), o.ni);
        return true;
    }
    default:
        return false;
    }
}

#endif  // JIT_HAVE_NATIVE

}  // namespace jit
