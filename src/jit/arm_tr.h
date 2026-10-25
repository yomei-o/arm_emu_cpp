// Shared by the ARM and Thumb frontends: what the translator learns about an
// instruction, the IR builder, and the operand helpers.
#pragma once

#include <cstring>
#include <vector>

#include "engine.h"

namespace jit {

namespace {

using arm::Cpu;

// What the translator learns about one instruction.
struct Info {
    bool native = false;
    bool term = false;         // ends the block
    bool reads = true;         // may read the flags
    bool writes_all = false;   // certainly overwrites N, Z, C and V
    bool effects = false;      // has effects beyond CPU state and memory (a system call...)
};

struct Tr {
    const GuestSlots& s;
    IrBlock& ir;
    int tmp_next = 1;
    uint32_t cur_off = 0;
    bool last = false;
    bool stored = false;
    int pc_tmp = -1;           // a data-processing result that is going to the pc
    uint32_t pcoff = 8;        // what R15 reads as, relative to the instruction: 8 in ARM state, 4 in Thumb

    Tr(const GuestSlots& sl, IrBlock& b) : s(sl), ir(b) {}

    int newtmp() { return s.tmp0 + 8 * (tmp_next++); }
    int reg(unsigned r) const { return static_cast<int>(r) * 8; }

    Op& emit(OpKind k) {
        ir.ops.emplace_back();
        Op& o = ir.ops.back();
        o.kind = k;
        o.sz = 4;
        o.pc = cur_off;
        o.ni = static_cast<uint16_t>(ir.insns);
        return o;
    }
    void mov(int d, int a) { Op& o = emit(OpKind::Mov); o.d = d; o.a = a; }
    void movi(int d, int64_t v) { Op& o = emit(OpKind::MovI); o.d = d; o.imm = v; }
    void bin(BinOp op, int sz, int d, int a, int b) {
        Op& o = emit(OpKind::Bin);
        o.cc = static_cast<uint8_t>(op); o.sz = static_cast<uint8_t>(sz); o.d = d; o.a = a; o.b = b;
    }
    void bini(BinOp op, int sz, int d, int a, int64_t imm) {
        Op& o = emit(OpKind::BinI);
        o.cc = static_cast<uint8_t>(op); o.sz = static_cast<uint8_t>(sz); o.d = d; o.a = a; o.imm = imm;
    }
    void un(UnOp op, int d, int a) {
        Op& o = emit(OpKind::Un);
        o.cc = static_cast<uint8_t>(op); o.sz = 4; o.d = d; o.a = a;
    }
    void ext(int d, int a, int src_sz, bool sgn, int dst_sz) {
        Op& o = emit(OpKind::Ext);
        o.d = d; o.a = a; o.sz = static_cast<uint8_t>(src_sz); o.cc = static_cast<uint8_t>(dst_sz);
        o.flags = sgn ? kFlagSigned : 0;
    }
    // Loads and stores address with 32-bit wraparound.
    void ld(int d, int a, int c, int cc, int64_t imm, int sz) {
        Op& o = emit(OpKind::Ld);
        o.d = d; o.a = a; o.c = c; o.cc = static_cast<uint8_t>(cc); o.imm = imm; o.sz = static_cast<uint8_t>(sz);
        o.flags = kFlagA32;
    }
    void lds(int d, int a, int c, int cc, int64_t imm, int sz) {
        Op& o = emit(OpKind::LdS);
        o.d = d; o.a = a; o.c = c; o.cc = static_cast<uint8_t>(cc); o.imm = imm; o.sz = static_cast<uint8_t>(sz);
        o.imm2 = 4;
        o.flags = kFlagA32;
    }
    void st(int a, int c, int cc, int64_t imm, int val, int sz) {
        Op& o = emit(OpKind::St);
        o.a = a; o.c = c; o.cc = static_cast<uint8_t>(cc); o.imm = imm; o.b = val; o.sz = static_cast<uint8_t>(sz);
        o.flags = kFlagA32;
        stored = true;
    }
    // After an instruction that stored: leave if that invalidated code.  Only the
    // instruction's last op may carry the check.
    void flush_check(uint64_t next_pc) {
        if (!stored || last || ir.ops.empty()) return;
        Op& s = ir.ops.back();
        if (s.kind == OpKind::St) {
            s.flags |= kFlagChk;
            s.imm2 = static_cast<int64_t>(next_pc);
            return;
        }
        Op& o = emit(OpKind::ChkFlush);
        o.imm = static_cast<int64_t>(next_pc);
    }
    void exit_to(uint64_t target) { Op& o = emit(OpKind::Exit); o.imm = static_cast<int64_t>(target); }
};

// The destination of a data-processing instruction.  A result for R15 is a branch:
// it goes to a temporary, and emit_insn ends the instruction with a jump to it.
int dreg(Tr& t, unsigned rd) {
    if (rd != 15) return t.reg(rd);
    t.pc_tmp = t.newtmp();
    return t.pc_tmp;
}

// A source register as a slot.  R15 reads as pc + 8, built in a temporary.
int src(Tr& t, unsigned r, uint32_t pc) {
    if (r != 15) return t.reg(r);
    const int d = t.newtmp();
    t.movi(d, static_cast<int64_t>(pc + t.pcoff));
    return d;
}

// Rm shifted by an immediate.  Returns the slot holding it, or -1 for a form the
// translator does not do (RRX).  `plain` says the shift was LSL #0, i.e. the shifter
// carry is the flag it was.
int shifted(Tr& t, unsigned rm, unsigned type, unsigned imm5, uint32_t pc, bool* plain) {
    const int a = src(t, rm, pc);
    *plain = false;
    if (type == 0) {   // LSL
        if (imm5 == 0) { *plain = true; return a; }
        const int d = t.newtmp();
        t.bini(BinOp::Shl, 4, d, a, imm5);
        return d;
    }
    const int d = t.newtmp();
    if (type == 1) {   // LSR; #0 encodes 32
        if (imm5 == 0) t.movi(d, 0); else t.bini(BinOp::Shr, 4, d, a, imm5);
        return d;
    }
    if (type == 2) {   // ASR; #0 encodes 32
        t.bini(BinOp::Sar, 4, d, a, imm5 == 0 ? 31 : imm5);
        return d;
    }
    if (imm5 == 0) return -1;   // RRX
    t.bini(BinOp::Ror, 4, d, a, imm5);
    return d;
}

inline bool is_logical(unsigned op) {
    return op == Cpu::kAND || op == Cpu::kEOR || op == Cpu::kTST || op == Cpu::kTEQ ||
           op == Cpu::kORR || op == Cpu::kMOV || op == Cpu::kBIC || op == Cpu::kMVN || op == Cpu::kORN;
}
inline bool is_compare(unsigned op) { return op >= Cpu::kTST && op <= Cpu::kCMN; }
// Whether a data-processing instruction reads the flags: the carry-using ones do,
// and so does a logical one that sets them, which leaves V (and often C) as it was.
inline bool reads_flags(unsigned op, bool setflags) {
    return op == Cpu::kADC || op == Cpu::kSBC || op == Cpu::kRSC || (setflags && is_logical(op));
}


// LDM / STM and PUSH / POP: `list` is the register list, u/p the increment and before bits.
// The base register is not R15.  Ends with a jump when R15 is loaded.
inline void block_transfer(Tr& t, unsigned rn, uint32_t list, bool load, bool u, bool p, bool w,
                           uint32_t pc, uint32_t next_pc) {
        unsigned count = 0;
        for (unsigned i = 0; i < 16; ++i) if (list & (1u << i)) ++count;
        const int64_t start = u ? (p ? 4 : 0) : (p ? -4 * static_cast<int64_t>(count) : -4 * static_cast<int64_t>(count) + 4);
        const bool rn_in_list = (list >> rn) & 1;
        int base = t.reg(rn);
        if (load && rn_in_list) {   // the loads must go on addressing from the old base
            base = t.newtmp();
            t.mov(base, t.reg(rn));
        }
        int64_t off = start;
        int pc_value = -1;
        if (count >= 3 && (load || !(list & 0x8000))) {
            // a long list is one op that loops
            Op& o = t.emit(load ? OpKind::A32Ldm : OpKind::A32Stm);
            o.a = base; o.imm = list & 0x7FFF; o.c = static_cast<int32_t>(start);
            if (load && (list & 0x8000)) { pc_value = t.newtmp(); o.imm = list; o.d = pc_value; }
            if (!load) {
                t.stored = true;
                // (with a write-back to come, the check goes after that instead)
                if (!t.last && !w) { o.flags = kFlagChk; o.imm2 = static_cast<int64_t>(next_pc); }
            }
        } else {
            for (unsigned i = 0; i < 16; ++i) {
                if (!(list & (1u << i))) continue;
                if (load) {
                    if (i == 15) { pc_value = t.newtmp(); t.ld(pc_value, base, t.s.zero, 0, off, 4); }
                    else t.ld(t.reg(i), base, t.s.zero, 0, off, 4);
                } else {
                    t.st(base, t.s.zero, 0, off, src(t, i, pc), 4);
                }
                off += 4;
            }
        }
        if (w && !(load && rn_in_list))
            t.bini(u ? BinOp::Add : BinOp::Sub, 4, t.reg(rn), t.reg(rn), 4 * static_cast<int64_t>(count));
        if (load && pc_value >= 0) {
            Op& o = t.emit(OpKind::A32Bx);
            o.a = pc_value;
        }
        if (!load && !(count >= 3 && !(list & 0x8000) && !w)) t.flush_check(next_pc);
}


// A data-processing operation on two slots.  `fl`: set the flags too (the logical
// operations leave C alone, which is what a shift-free operand does).  Compares have
// no result; the caller passes a scratch slot as `d`.
inline void dp_reg(Tr& t, unsigned op, int d, int a, int b, bool fl) {
    if (!fl && op != Cpu::kADC && op != Cpu::kSBC && op != Cpu::kRSC && op != Cpu::kORN) {
        switch (op) {
        case Cpu::kAND: t.bin(BinOp::And, 4, d, a, b); return;
        case Cpu::kEOR: t.bin(BinOp::Xor, 4, d, a, b); return;
        case Cpu::kORR: t.bin(BinOp::Or, 4, d, a, b); return;
        case Cpu::kBIC: t.bin(BinOp::Andn, 4, d, a, b); return;
        case Cpu::kSUB: t.bin(BinOp::Sub, 4, d, a, b); return;
        case Cpu::kRSB: t.bin(BinOp::Sub, 4, d, b, a); return;
        case Cpu::kADD: t.bin(BinOp::Add, 4, d, a, b); return;
        case Cpu::kMOV: t.mov(d, b); return;
        case Cpu::kMVN: t.bini(BinOp::Xor, 4, d, b, 0xFFFFFFFFll); return;
        default: break;
        }
    }
    Op& o = t.emit(OpKind::A32Dp);
    o.cc = static_cast<uint8_t>(op); o.d = d; o.a = a; o.b = b;
    o.imm2 = -1;
    if (fl) o.flags |= kFlagEager;
}

// The same with an immediate second operand; `carry` is what the operand's own
// carry-out is (-1: C stays as it was).
inline void dp_imm(Tr& t, unsigned op, int d, int a, int64_t imm, bool fl, int64_t carry) {
    if (!fl && op != Cpu::kADC && op != Cpu::kSBC && op != Cpu::kRSC && op != Cpu::kRSB && op != Cpu::kORN) {
        switch (op) {
        case Cpu::kAND: t.bini(BinOp::And, 4, d, a, imm); return;
        case Cpu::kEOR: t.bini(BinOp::Xor, 4, d, a, imm); return;
        case Cpu::kORR: t.bini(BinOp::Or, 4, d, a, imm); return;
        case Cpu::kBIC: t.bini(BinOp::Andn, 4, d, a, imm); return;
        case Cpu::kSUB: t.bini(BinOp::Sub, 4, d, a, imm); return;
        case Cpu::kADD: t.bini(BinOp::Add, 4, d, a, imm); return;
        case Cpu::kMOV: t.movi(d, imm); return;
        case Cpu::kMVN: t.movi(d, static_cast<int64_t>(~static_cast<uint32_t>(imm))); return;
        default: break;
        }
    }
    Op& o = t.emit(OpKind::A32DpI);
    o.cc = static_cast<uint8_t>(op); o.d = d; o.a = (op == Cpu::kMOV || op == Cpu::kMVN) ? t.s.zero : a;
    o.imm = imm; o.imm2 = carry;
    if (fl) o.flags |= kFlagEager;
}

}  // namespace

}  // namespace jit
