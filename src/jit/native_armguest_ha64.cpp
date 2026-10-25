// The ARM guest's ops for the AArch64 native backend: data processing with and without flags
// (the host's flags are the guest's, read back into the guest's N, Z, C, V bytes), the
// conditional branch and conditional exit, the interworking branch, register-amount shifts
// and LDM / STM.  Anything else runs through its threaded handler.
#include "native_cond_a64.h"

namespace jit {

#ifdef JIT_NATIVE_A64

using namespace a64;

namespace {
// 32-bit forms of the carry and flag-setting variants that native_a64.h does not name
constexpr uint32_t ADC_ = 0x9A000000u, ADCS_ = 0xBA000000u, SBC_ = 0xDA000000u, SBCS_ = 0xFA000000u;
}  // namespace

int guest_native_cells(const Op& o) {
    switch (o.kind) {
        case OpKind::A32Jcc: return 2;
        default: return 0;
    }
}

bool guest_native_emit(NativeGenA& g, const Op& o) {
    Asm& a = g.a;
    const GuestSlots& s = g.s;
    const int insns = g.insns;
    const uint64_t rip_here = g.start + o.pc;
    switch (o.kind) {
    case OpKind::A32Dp: case OpKind::A32DpI: {
        const unsigned op = o.cc;
        const bool fl = (o.flags & kFlagEager) != 0;
        const bool imm = o.kind == OpKind::A32DpI;
        const bool reverse = op == 3 || op == 7;
        // x0 = Rn, x1 = the second operand; swapped for the reverse subtractions
        g.ld_sz(0, o.a, 4);
        if (imm) a.mov_imm(1, static_cast<uint32_t>(o.imm)); else g.ld_sz(1, o.b, 4);
        if (reverse) { a.mov_rr32(2, 0); a.mov_rr32(0, 1); a.mov_rr32(1, 2); }
        bool arith = false;
        switch (op) {
            case 0: case 8: a.alu(fl ? ANDS : AND, false, 0, 0, 1); break;               // AND, TST
            case 1: case 9: a.alu(EOR, false, 0, 0, 1); break;                           // EOR, TEQ
            case 12: a.alu(ORR, false, 0, 0, 1); break;                                  // ORR
            case 13: a.mov_rr32(0, 1); break;                                            // MOV
            case 14: a.alu(fl ? 0xEA200000u : BIC, false, 0, 0, 1); break;               // BIC
            case 15: a.alu(ORN, false, 0, XZR, 1); break;                                // MVN
            case 16: a.alu(ORN, false, 0, 0, 1); break;                                  // ORN
            case 2: case 10: case 3: a.alu(fl ? SUBS : SUB, false, 0, 0, 1); arith = true; break;   // SUB, CMP, RSB
            case 4: case 11: a.alu(fl ? ADDS : ADD, false, 0, 0, 1); arith = true; break;           // ADD, CMN
            case 5: case 6: case 7: {                                                    // ADC, SBC, RSC: C = the guest carry
                a.ldrb(3, ra::ENV, s.fl_c);
                a.cmp_imm(false, 3, 1);
                const uint32_t base = op == 5 ? (fl ? ADCS_ : ADC_) : (fl ? SBCS_ : SBC_);
                a.alu(base, false, 0, 0, 1);
                arith = true;
                break;
            }
            default: return false;
        }
        if (fl) {
            if (!arith && op != 0 && op != 8 && op != 14) a.alu(ANDS, false, XZR, 0, 0);   // N and Z of the result
            a.cset(false, 2, MI); a.strb(2, ra::ENV, s.fl_n);
            a.cset(false, 2, EQ); a.strb(2, ra::ENV, s.fl_z);
            if (arith) {
                a.cset(false, 2, HS); a.strb(2, ra::ENV, s.fl_c);
                a.cset(false, 2, VS); a.strb(2, ra::ENV, s.fl_v);
            } else if (o.imm2 >= 0) {
                a.mov_imm(2, o.imm2 ? 1 : 0); a.strb(2, ra::ENV, s.fl_c);
            }
        }
        if (!(op >= 8 && op <= 11)) g.st(o.d, 0);
        return true;
    }
    case OpKind::A32Jcc: {
        const int hcc = arm_cond(g, o.cc);
        if (hcc < 0) { g.exit_static(static_cast<uint64_t>(o.imm), insns); g.next_cell(); return true; }
        Label taken;
        a.b_cond(hcc, taken);
        g.exit_static(static_cast<uint64_t>(o.imm2), insns);
        a.bind(taken);
        g.exit_static(static_cast<uint64_t>(o.imm), insns);
        return true;
    }
    case OpKind::A32ExitIf: {
        const int hcc = arm_cond(g, o.cc);
        if (hcc < 0) return true;
        Label ok;
        a.b_cond(hcc, ok);
        g.st_imm(s.rip, o.imm);
        g.leave_with(o.ni);
        a.bind(ok);
        return true;
    }
    case OpKind::A32Bx: {
        g.ld(0, o.a);
        Label arm_state;
        a.mov_imm(1, 1);
        a.tst(false, 0, 1);
        a.b_cond(EQ, arm_state);
        a.mov_imm(1, 0xFFFFFFFEull);                   // Thumb: bit 0 is not part of the address
        a.alu(AND, false, 0, 0, 1);
        g.st(s.rip, 0);
        a.mov_imm(1, 1); a.strb(1, ra::ENV, s.thumb);
        g.leave_with(insns);
        a.bind(arm_state);
        a.mov_imm(1, 0xFFFFFFFCull);
        a.alu(AND, false, 0, 0, 1);
        g.st(s.rip, 0);
        a.strb(XZR, ra::ENV, s.thumb);
        g.leave_with(insns);
        return true;
    }
    case OpKind::A32Shift: {
        // The amount is the low byte of b; 32 or more is not the same as the amount modulo 32.
        g.ld_sz(0, o.a, 4);
        g.ld_sz(1, o.b, 1);
        const int type = o.cc;
        if (type == 2) {                               // ASR: an amount of 32 or more shifts in the sign
            a.mov_imm(2, 31);
            a.cmp_imm(false, 1, 32);
            a.csel(false, 1, 2, 1, HS);
            a.shiftv(ASR, false, 0, 0, 1);
        } else if (type == 3) {
            a.shiftv(ROR, false, 0, 0, 1);
        } else {
            a.shiftv(type == 0 ? LSL : LSR, false, 0, 0, 1);
            a.cmp_imm(false, 1, 32);
            a.csel(false, 0, XZR, 0, HS);
        }
        g.st(o.d, 0);
        return true;
    }
    case OpKind::A32Ldm: case OpKind::A32Stm: {
        const uint32_t list = static_cast<uint32_t>(o.imm);
        int64_t off = o.c;
        const bool load = o.kind == OpKind::A32Ldm;
        for (unsigned i = 0; i < 16; ++i) {
            if (!(list & (1u << i))) continue;
            g.ld_sz(0, o.a, 4);
            if (off >= 0 && off < 4096) a.add_imm(false, 0, 0, static_cast<uint32_t>(off));
            else if (off < 0 && -off < 4096) a.sub_imm(false, 0, 0, static_cast<uint32_t>(-off));
            else { a.mov_imm(1, static_cast<uint64_t>(off)); a.alu(ADD, false, 0, 0, 1); }
            if (load) {
                g.mem_read(4, rip_here);
                g.st(i == 15 ? o.d : static_cast<int>(i) * 8, 0);
            } else {
                g.ld(4, static_cast<int>(i) * 8);
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

#endif  // JIT_NATIVE_A64

}  // namespace jit
