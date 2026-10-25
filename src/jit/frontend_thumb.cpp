// Thumb frontend: guest halfwords -> IR.
//
// Thumb instructions are two bytes or four.  Outside an IT block every one of them
// is unconditional (except the branches), so there is no per-instruction test as in
// ARM state; and an IT block - the instruction and the up to four after it - is not
// translated at all: all of it is handed to the interpreter, one instruction at a
// time, which keeps ITSTATE exact between them.
//
// Almost all of the 16-bit set is translated, and the 32-bit instructions a compiler
// emits all the time: branches and calls, loads and stores, data processing with
// immediates and shifted registers, multiplies and divides, MOVW/MOVT, bitfields.
// What is not becomes an `Interp` op, as in ARM state.
//
// Addresses in the IR are the real, even ones.  A Thumb block is identified by
// address | 1 (see Glue::pc), which is why an `Interp` op's expected next pc is odd.
#include "arm_tr.h"

namespace jit {

namespace {

using arm::Cpu;

inline uint32_t sext(uint32_t v, unsigned bits) {
    const uint32_t m = 1u << (bits - 1);
    return (v ^ m) - m;
}

// ---- 16-bit instructions ----------------------------------------------------------

// `in_it`: the instruction is inside an IT block, where the flag-setting forms of the
// 16-bit data-processing instructions do not set the flags (the compares still do).
void emit_t16(Tr& t, uint16_t insn, uint32_t pc, bool flags_live, Info& info, bool in_it) {
    const uint32_t next_pc = pc + 2;
    const bool live_all = flags_live;      // for the instructions that always set the flags
    flags_live = flags_live && !in_it;     // for the ones that set them outside IT blocks
    const unsigned op = (insn >> 10) & 0x3F;
    auto native = [&](bool reads, bool writes_all) {
        info.native = true;
        info.reads = reads;
        info.writes_all = writes_all;
    };

    // shift (immediate), add, subtract
    if ((op & 0x38) == 0) {
        const unsigned opc = (insn >> 9) & 0x1F;
        const unsigned rd = insn & 7, rn = (insn >> 3) & 7;
        if ((opc & 0x1C) != 0x0C) {   // the three shifts, which set N, Z and C
            const unsigned type = (opc >> 2) & 3, imm5 = (insn >> 6) & 0x1F;
            if (type == 0 && imm5 == 0) {   // MOVS Rd, Rn
                native(false, false);
                dp_reg(t, Cpu::kMOV, t.reg(rd), t.s.zero, t.reg(rn), flags_live);
                return;
            }
            if (flags_live) return;   // the shifter carry is wanted: the interpreter does it
            native(false, false);
            const int d = t.reg(rd), a = t.reg(rn);
            if (type == 0) t.bini(BinOp::Shl, 4, d, a, imm5);
            else if (type == 1) { if (imm5 == 0) t.movi(d, 0); else t.bini(BinOp::Shr, 4, d, a, imm5); }
            else t.bini(BinOp::Sar, 4, d, a, imm5 == 0 ? 31 : imm5);
            return;
        }
        native(false, true);
        const bool fl = flags_live;
        const unsigned kind = opc & 3;
        const unsigned dop = (kind & 1) ? Cpu::kSUB : Cpu::kADD;
        if (kind < 2) dp_reg(t, dop, t.reg(rd), t.reg(rn), t.reg((insn >> 6) & 7), fl);
        else dp_imm(t, dop, t.reg(rd), t.reg(rn), (insn >> 6) & 7, fl, -1);
        return;
    }
    // MOV / CMP / ADD / SUB (immediate)
    if ((op & 0x38) == 0x08) {
        const unsigned rd = (insn >> 8) & 7;
        const int64_t imm = insn & 0xFF;
        switch ((insn >> 11) & 3) {
        case 0: native(false, false); dp_imm(t, Cpu::kMOV, t.reg(rd), t.s.zero, imm, flags_live, -1); return;
        case 1:
            native(false, true);
            if (live_all) dp_imm(t, Cpu::kCMP, t.newtmp(), t.reg(rd), imm, true, -1);
            return;
        case 2: native(false, true); dp_imm(t, Cpu::kADD, t.reg(rd), t.reg(rd), imm, flags_live, -1); return;
        default: native(false, true); dp_imm(t, Cpu::kSUB, t.reg(rd), t.reg(rd), imm, flags_live, -1); return;
        }
    }
    // data processing, register
    if (op == 0x10) {
        const unsigned opc = (insn >> 6) & 0xF;
        const unsigned rd = insn & 7, rm = (insn >> 3) & 7;
        switch (opc) {
        case 0x0: case 0x1: case 0xC: case 0xE: {   // AND EOR ORR BIC: N, Z set, C and V kept
            static const unsigned kOp[16] = {Cpu::kAND, Cpu::kEOR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Cpu::kORR, 0, Cpu::kBIC, 0};
            native(false, false);
            dp_reg(t, kOp[opc], t.reg(rd), t.reg(rd), t.reg(rm), flags_live);
            return;
        }
        case 0x2: case 0x3: case 0x4: case 0x7: {   // shifts by register
            if (flags_live) return;
            native(false, false);
            static const uint8_t kType[8] = {0, 0, 0, 1, 2, 0, 0, 3};
            Op& o = t.emit(OpKind::A32Shift);
            o.cc = kType[opc]; o.d = t.reg(rd); o.a = t.reg(rd); o.b = t.reg(rm);
            return;
        }
        case 0x5: case 0x6:   // ADC SBC
            native(true, true);
            dp_reg(t, opc == 5 ? Cpu::kADC : Cpu::kSBC, t.reg(rd), t.reg(rd), t.reg(rm), flags_live);
            return;
        case 0x8:   // TST
            native(false, false);
            if (live_all) dp_reg(t, Cpu::kTST, t.newtmp(), t.reg(rd), t.reg(rm), true);
            return;
        case 0x9:   // RSBS Rd, Rm, #0
            native(false, true);
            dp_imm(t, Cpu::kRSB, t.reg(rd), t.reg(rm), 0, flags_live, -1);
            return;
        case 0xA: case 0xB:   // CMP CMN
            native(false, true);
            if (live_all) dp_reg(t, opc == 0xA ? Cpu::kCMP : Cpu::kCMN, t.newtmp(), t.reg(rd), t.reg(rm), true);
            return;
        case 0xD:   // MUL
            if (flags_live) return;
            native(false, false);
            t.bin(BinOp::Mul, 4, t.reg(rd), t.reg(rd), t.reg(rm));
            return;
        default:   // MVN
            native(false, false);
            dp_reg(t, Cpu::kMVN, t.reg(rd), t.s.zero, t.reg(rm), flags_live);
            return;
        }
    }
    // special data instructions and BX
    if (op == 0x11) {
        const unsigned opc = (insn >> 8) & 3;
        const unsigned rm = (insn >> 3) & 0xF;
        const unsigned rd = (insn & 7) | (((insn >> 7) & 1) << 3);
        switch (opc) {
        case 0:   // ADD (no flags)
            if (rd == 15) return;
            native(false, false);
            t.bin(BinOp::Add, 4, t.reg(rd), src(t, rd, pc), src(t, rm, pc));
            return;
        case 1:   // CMP
            native(false, true);
            if (live_all) {
                const int a = src(t, rd, pc), b = src(t, rm, pc);
                dp_reg(t, Cpu::kCMP, t.newtmp(), a, b, true);
            }
            return;
        case 2:   // MOV
            if (rd == 15) return;
            native(false, false);
            t.mov(t.reg(rd), src(t, rm, pc));
            return;
        default:
            if (rm == 15) return;
            if ((insn >> 7) & 1) {   // BLX
                native(false, false);
                info.term = true;
                const int target = t.newtmp();
                t.mov(target, t.reg(rm));
                t.movi(t.reg(14), static_cast<int64_t>(next_pc | 1));
                Op& o = t.emit(OpKind::A32Bx);
                o.a = target;
            } else {   // BX
                native(false, false);
                info.term = true;
                Op& o = t.emit(OpKind::A32Bx);
                o.a = t.reg(rm);
            }
            return;
        }
    }
    // LDR (literal)
    if ((op & 0x3E) == 0x12) {
        native(false, false);
        t.ld(t.reg((insn >> 8) & 7), t.s.zero, t.s.zero, 0, ((pc + 4) & ~3u) + (insn & 0xFF) * 4, 4);
        return;
    }
    // load / store, register offset
    if ((op & 0x3C) == 0x14) {
        const unsigned rt = insn & 7, rn = (insn >> 3) & 7, rm = (insn >> 6) & 7;
        native(false, false);
        switch ((insn >> 9) & 7) {
        case 0: t.st(t.reg(rn), t.reg(rm), 0, 0, t.reg(rt), 4); t.flush_check(next_pc); return;
        case 1: t.st(t.reg(rn), t.reg(rm), 0, 0, t.reg(rt), 2); t.flush_check(next_pc); return;
        case 2: t.st(t.reg(rn), t.reg(rm), 0, 0, t.reg(rt), 1); t.flush_check(next_pc); return;
        case 3: t.lds(t.reg(rt), t.reg(rn), t.reg(rm), 0, 0, 1); return;
        case 4: t.ld(t.reg(rt), t.reg(rn), t.reg(rm), 0, 0, 4); return;
        case 5: t.ld(t.reg(rt), t.reg(rn), t.reg(rm), 0, 0, 2); return;
        case 6: t.ld(t.reg(rt), t.reg(rn), t.reg(rm), 0, 0, 1); return;
        default: t.lds(t.reg(rt), t.reg(rn), t.reg(rm), 0, 0, 2); return;
        }
    }
    // load / store, immediate offset
    if ((op & 0x38) == 0x18 || (op & 0x3C) == 0x20) {
        const unsigned rt = insn & 7, rn = (insn >> 3) & 7;
        const unsigned imm5 = (insn >> 6) & 0x1F;
        const bool load = (insn >> 11) & 1;
        int size, scale;
        switch ((insn >> 12) & 0xF) {
            case 6: size = 4; scale = 4; break;
            case 7: size = 1; scale = 1; break;
            default: size = 2; scale = 2; break;
        }
        native(false, false);
        if (load) t.ld(t.reg(rt), t.reg(rn), t.s.zero, 0, imm5 * scale, size);
        else { t.st(t.reg(rn), t.s.zero, 0, imm5 * scale, t.reg(rt), size); t.flush_check(next_pc); }
        return;
    }
    // load / store, SP relative
    if ((op & 0x3C) == 0x24) {
        const unsigned rt = (insn >> 8) & 7;
        native(false, false);
        if ((insn >> 11) & 1) t.ld(t.reg(rt), t.reg(13), t.s.zero, 0, (insn & 0xFF) * 4, 4);
        else { t.st(t.reg(13), t.s.zero, 0, (insn & 0xFF) * 4, t.reg(rt), 4); t.flush_check(next_pc); }
        return;
    }
    // ADR / ADD (SP plus immediate)
    if ((op & 0x3C) == 0x28) {
        const unsigned rd = (insn >> 8) & 7;
        const uint32_t imm = (insn & 0xFF) * 4;
        native(false, false);
        if ((insn >> 11) & 1) t.bini(BinOp::Add, 4, t.reg(rd), t.reg(13), imm);
        else t.movi(t.reg(rd), static_cast<int64_t>(((pc + 4) & ~3u) + imm));
        return;
    }
    // miscellaneous
    if ((op & 0x3C) == 0x2C) {
        if ((insn & 0xFF00) == 0xB000) {   // ADD / SUB SP, immediate
            native(false, false);
            t.bini((insn & 0x80) ? BinOp::Sub : BinOp::Add, 4, t.reg(13), t.reg(13), (insn & 0x7F) * 4);
            return;
        }
        if ((insn & 0xF500) == 0xB100) {   // CBZ / CBNZ
            if (in_it) return;
            const unsigned rn = insn & 7;
            const uint32_t imm = (((insn >> 9) & 1) << 6) | (((insn >> 3) & 0x1F) << 1);
            native(false, false);
            info.term = true;
            Op& o = t.emit(OpKind::JccGenI);
            o.cc = static_cast<uint8_t>(((insn >> 11) & 1) ? Cond::Ne : Cond::Eq); o.sz = 4;
            o.a = t.reg(rn); o.d = 0;
            o.imm = static_cast<int64_t>(pc + 4 + imm); o.imm2 = static_cast<int64_t>(next_pc);
            return;
        }
        if ((insn & 0xFF00) == 0xB200) {   // SXTH SXTB UXTH UXTB
            native(false, false);
            const unsigned k = (insn >> 6) & 3;
            t.ext(t.reg(insn & 7), t.reg((insn >> 3) & 7), (k & 1) ? 1 : 2, k < 2, 4);
            return;
        }
        if ((insn & 0xFE00) == 0xB400) {   // PUSH
            native(false, false);
            block_transfer(t, 13, (insn & 0xFF) | (((insn >> 8) & 1) << 14), false, false, true, true, pc, next_pc);
            return;
        }
        if ((insn & 0xFE00) == 0xBC00) {   // POP
            const uint32_t list = (insn & 0xFF) | (((insn >> 8) & 1) << 15);
            if (list & 0x8000) info.term = true;
            native(false, false);
            block_transfer(t, 13, list, true, true, false, true, pc, next_pc);
            return;
        }
        if ((insn & 0xFF00) == 0xBA00) {   // REV REV16 REVSH
            native(false, false);
            const unsigned k = (insn >> 6) & 3;
            const int d = t.reg(insn & 7), m = t.reg((insn >> 3) & 7);
            if (k == 0) t.un(UnOp::Rev, d, m);
            else if (k == 1) t.un(UnOp::Rev16, d, m);
            else {
                const int x = t.newtmp();
                t.un(UnOp::Rev16, x, m);
                t.ext(d, x, 2, true, 4);
            }
            return;
        }
        if ((insn & 0xFF00) == 0xBF00 && (insn & 0xF) == 0) {   // NOP and the other hints
            native(false, false);
            return;
        }
        return;
    }
    // STM / LDM
    if ((op & 0x3C) == 0x30) {
        const unsigned rn = (insn >> 8) & 7;
        const uint32_t list = insn & 0xFF;
        if (list == 0) return;
        native(false, false);
        block_transfer(t, rn, list, (insn >> 11) & 1, true, false, true, pc, next_pc);
        return;
    }
    // conditional branch / SVC
    if ((op & 0x3C) == 0x34) {
        if (in_it) return;   // not allowed inside an IT block
        const unsigned cond = (insn >> 8) & 0xF;
        if (cond == 0xF) { info.effects = true; return; }   // SVC
        if (cond == 0xE) return;
        native(true, false);
        info.term = true;
        Op& o = t.emit(OpKind::A32Jcc);
        o.cc = static_cast<uint8_t>(cond);
        o.imm = static_cast<int64_t>(pc + 4 + static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(insn & 0xFF)) * 2));
        o.imm2 = static_cast<int64_t>(next_pc);
        return;
    }
    // B
    if ((op & 0x3E) == 0x38) {
        native(false, false);
        info.term = true;
        t.exit_to(pc + 4 + static_cast<uint32_t>((static_cast<int32_t>(static_cast<uint32_t>(insn) << 21) >> 21) * 2));
        return;
    }
}

// ---- 32-bit instructions ----------------------------------------------------------

// Thumb-2 data-processing opcodes, as the A32 encoding numbers them (0: no such).
const unsigned kDpMap[16] = {Cpu::kAND, Cpu::kBIC, Cpu::kORR, Cpu::kORN, Cpu::kEOR, 0, 0, 0,
                             Cpu::kADD, 0, Cpu::kADC, Cpu::kSBC, 0, Cpu::kSUB, Cpu::kRSB, 0};

void emit_t32(Tr& t, uint32_t insn, uint32_t pc, bool flags_live, Info& info) {
    const uint32_t next_pc = pc + 4;
    const unsigned op1 = (insn >> 27) & 3;
    const unsigned op2 = (insn >> 20) & 0x7F;
    auto native = [&](bool reads, bool writes_all) {
        info.native = true;
        info.reads = reads;
        info.writes_all = writes_all;
    };

    if (op1 == 2 && ((insn >> 15) & 1)) {   // branches and miscellaneous control
        const unsigned bits = (insn >> 12) & 7;
        const unsigned op = (insn >> 20) & 0x7F;
        if ((bits & 5) == 0) {
            if ((op & 0x38) == 0x38) return;   // MSR, MRS, hints, barriers
            // B (T3, conditional)
            const unsigned cond = (insn >> 22) & 0xF;
            if (cond >= 0xE) return;
            int32_t imm = static_cast<int32_t>((((insn >> 26) & 1) << 20) | (((insn >> 11) & 1) << 19) |
                                               (((insn >> 13) & 1) << 18) | (((insn >> 16) & 0x3F) << 12) |
                                               ((insn & 0x7FF) << 1));
            imm = (imm << 11) >> 11;
            native(true, false);
            info.term = true;
            Op& o = t.emit(OpKind::A32Jcc);
            o.cc = static_cast<uint8_t>(cond);
            o.imm = static_cast<int64_t>(pc + 4 + static_cast<uint32_t>(imm));
            o.imm2 = static_cast<int64_t>(next_pc);
            return;
        }
        const uint32_t s = (insn >> 26) & 1;
        const uint32_t j1 = (insn >> 13) & 1, j2 = (insn >> 11) & 1;
        const uint32_t i1 = ~(j1 ^ s) & 1, i2 = ~(j2 ^ s) & 1;
        int32_t imm = static_cast<int32_t>((s << 24) | (i1 << 23) | (i2 << 22) | (((insn >> 16) & 0x3FF) << 12) |
                                           ((insn & 0x7FF) << 1));
        imm = (imm << 7) >> 7;
        if ((bits & 5) == 5 || (bits & 5) == 1) {   // BL, B (T4)
            native(false, false);
            info.term = true;
            if ((bits & 5) == 5) t.movi(t.reg(14), static_cast<int64_t>(next_pc | 1));
            t.exit_to(pc + 4 + static_cast<uint32_t>(imm));
        }
        return;
    }

    if (op1 == 2) {   // data processing with an immediate
        const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF;
        const uint32_t imm12 = (((insn >> 26) & 1) << 11) | (((insn >> 12) & 7) << 8) | (insn & 0xFF);
        if ((op2 & 0x20) == 0) {   // modified immediate
            const unsigned op = (insn >> 21) & 0xF;
            const bool setflags = (insn >> 20) & 1;
            unsigned mapped = kDpMap[op];
            if (op == 2 && rn == 15) mapped = Cpu::kMOV;
            if (op == 3 && rn == 15) mapped = Cpu::kMVN;
            if (op == 1 && rd == 15) mapped = Cpu::kBIC;
            bool compare = false;
            if (rd == 15 && setflags) {
                switch (op) {
                    case 0: mapped = Cpu::kTST; compare = true; break;
                    case 4: mapped = Cpu::kTEQ; compare = true; break;
                    case 8: mapped = Cpu::kCMN; compare = true; break;
                    case 13: mapped = Cpu::kCMP; compare = true; break;
                    default: break;
                }
            }
            if (mapped == 0 || (rd == 15 && !compare)) return;
            const arm::ShiftResult sr = arm::thumb_expand_imm_c(imm12, false);
            const bool unchanged = (imm12 & 0xC00) == 0;
            native(mapped == Cpu::kADC || mapped == Cpu::kSBC || (setflags && is_logical(mapped)),
                   setflags && !is_logical(mapped));
            const bool fl = setflags && flags_live;
            if (compare && !fl) return;
            const int d = compare ? t.newtmp() : t.reg(rd);
            const int a = (mapped == Cpu::kMOV || mapped == Cpu::kMVN) ? t.s.zero : src(t, rn, pc);
            dp_imm(t, mapped, d, a, static_cast<int64_t>(sr.result), fl,
                   unchanged ? int64_t(-1) : int64_t(sr.result >> 31));
            return;
        }
        // plain binary immediate
        const unsigned op = (insn >> 20) & 0x1F;
        switch (op) {
        case 0x00: case 0x0A:   // ADDW, SUBW (ADR when Rn is the pc)
            if (rd == 15) return;
            native(false, false);
            if (rn == 15) {
                const uint32_t base = (pc + 4) & ~3u;
                t.movi(t.reg(rd), static_cast<int64_t>(op == 0 ? base + imm12 : base - imm12));
            } else {
                t.bini(op == 0 ? BinOp::Add : BinOp::Sub, 4, t.reg(rd), t.reg(rn), imm12);
            }
            return;
        case 0x04:   // MOVW
            if (rd == 15) return;
            native(false, false);
            t.movi(t.reg(rd), static_cast<int64_t>((((insn >> 16) & 0xF) << 12) | imm12));
            return;
        case 0x0C: {   // MOVT
            if (rd == 15) return;
            native(false, false);
            const uint32_t hi = ((((insn >> 16) & 0xF) << 12) | imm12) << 16;
            t.bini(BinOp::And, 4, t.reg(rd), t.reg(rd), 0xFFFF);
            t.bini(BinOp::Or, 4, t.reg(rd), t.reg(rd), hi);
            return;
        }
        case 0x14: case 0x1C: {   // SBFX / UBFX
            const unsigned lsb = (((insn >> 12) & 7) << 2) | ((insn >> 6) & 3);
            const unsigned width = (insn & 0x1F) + 1;
            if (rd == 15 || rn == 15 || lsb + width > 32) return;
            native(false, false);
            if (width == 32) { t.mov(t.reg(rd), t.reg(rn)); return; }
            const int x = t.newtmp();
            if (op == 0x1C) {
                t.bini(BinOp::Shr, 4, x, t.reg(rn), lsb);
                t.bini(BinOp::And, 4, t.reg(rd), x, (1ll << width) - 1);
            } else {
                t.bini(BinOp::Shl, 4, x, t.reg(rn), 32 - lsb - width);
                t.bini(BinOp::Sar, 4, t.reg(rd), x, 32 - width);
            }
            return;
        }
        case 0x16: {   // BFI / BFC
            const unsigned lsb = (((insn >> 12) & 7) << 2) | ((insn >> 6) & 3);
            const unsigned msb = insn & 0x1F;
            if (msb < lsb || rd == 15) return;
            const uint32_t mask = msb == 31 ? (0xFFFFFFFFu << lsb) : (((1u << (msb - lsb + 1)) - 1) << lsb);
            native(false, false);
            if (rn == 15) {
                t.bini(BinOp::And, 4, t.reg(rd), t.reg(rd), static_cast<int64_t>(~mask));
                return;
            }
            const int x = t.newtmp();
            t.bini(BinOp::Shl, 4, x, t.reg(rn), lsb);
            t.bini(BinOp::And, 4, x, x, mask);
            t.bini(BinOp::And, 4, t.reg(rd), t.reg(rd), static_cast<int64_t>(~mask));
            t.bin(BinOp::Or, 4, t.reg(rd), t.reg(rd), x);
            return;
        }
        default: return;
        }
    }

    if (op1 == 1) {
        if ((op2 & 0x64) == 0x00) {   // load / store multiple
            const unsigned kind = (insn >> 23) & 3;
            const bool load = (insn >> 20) & 1, w = (insn >> 21) & 1;
            const unsigned rn = (insn >> 16) & 0xF;
            const uint32_t list = insn & 0xFFFF;
            if ((kind != 1 && kind != 2) || rn == 15 || list == 0) return;
            if (load && (list & 0x8000)) info.term = true;
            native(false, false);
            block_transfer(t, rn, list, load, kind == 1, kind == 2, w, pc, next_pc);
            return;
        }
        if ((op2 & 0x60) == 0x20) {   // data processing, shifted register
            const unsigned op = (insn >> 21) & 0xF;
            const bool setflags = (insn >> 20) & 1;
            const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF, rm = insn & 0xF;
            const unsigned imm5 = (((insn >> 12) & 7) << 2) | ((insn >> 6) & 3);
            const unsigned type = (insn >> 4) & 3;
            unsigned mapped = kDpMap[op];
            if (op == 2 && rn == 15) mapped = Cpu::kMOV;
            if (op == 3 && rn == 15) mapped = Cpu::kMVN;
            bool compare = false;
            if (rd == 15 && setflags) {
                switch (op) {
                    case 0: mapped = Cpu::kTST; compare = true; break;
                    case 4: mapped = Cpu::kTEQ; compare = true; break;
                    case 8: mapped = Cpu::kCMN; compare = true; break;
                    case 13: mapped = Cpu::kCMP; compare = true; break;
                    default: break;
                }
            }
            if (mapped == 0 || (rd == 15 && !compare)) return;
            if (setflags && is_logical(mapped) && (type != 0 || imm5 != 0)) return;   // the shifter carry
            if (type == 3 && imm5 == 0) return;   // RRX
            native(mapped == Cpu::kADC || mapped == Cpu::kSBC || (setflags && is_logical(mapped)),
                   setflags && !is_logical(mapped));
            const bool fl = setflags && flags_live;
            if (compare && !fl) return;
            bool plain;
            const int b = shifted(t, rm, type, imm5, pc, &plain);
            const int d = compare ? t.newtmp() : t.reg(rd);
            const int a = (mapped == Cpu::kMOV || mapped == Cpu::kMVN) ? t.s.zero : src(t, rn, pc);
            dp_reg(t, mapped, d, a, b, fl);
            return;
        }
        if ((op2 & 0x64) == 0x04) info.effects = true;   // the exclusives and table branches: the monitor is not verified state
        return;
    }

    // op1 == 3: loads and stores of single items, register data processing, multiplies
    const unsigned op2b = (insn >> 20) & 0x7F;
    if ((op2b & 0x71) == 0x00 || (op2b & 0x67) == 0x01 || (op2b & 0x67) == 0x03 || (op2b & 0x67) == 0x05) {
        const bool sign = (insn >> 24) & 1;
        const bool imm12_form = (insn >> 23) & 1;
        const unsigned size2 = (insn >> 21) & 3;
        const bool load = (insn >> 20) & 1;
        const unsigned rn = (insn >> 16) & 0xF, rt = (insn >> 12) & 0xF;
        if (size2 == 3) return;
        const int size = size2 == 0 ? 1 : (size2 == 1 ? 2 : 4);
        if (load && rt == 15) return;   // a hint, or a load into the pc
        if (rn == 15) {   // literal
            const uint32_t base = (pc + 4) & ~3u;
            const uint32_t imm = insn & 0xFFF;
            const uint32_t addr = imm12_form ? base + imm : base - imm;
            native(false, false);
            if (load) {
                if (sign) t.lds(t.reg(rt), t.s.zero, t.s.zero, 0, addr, size);
                else t.ld(t.reg(rt), t.s.zero, t.s.zero, 0, addr, size);
            } else {
                t.st(t.s.zero, t.s.zero, 0, addr, src(t, rt, pc), size);
                t.flush_check(next_pc);
            }
            return;
        }
        int addr_slot = t.reg(rn), idx_slot = t.s.zero, idx_shift = 0;
        int64_t addr_imm = 0;
        bool wb = false, rn_plus = true;
        int64_t wb_off = 0;
        if (imm12_form) {
            addr_imm = insn & 0xFFF;
        } else if ((insn >> 11) & 1) {   // the 8-bit P/U/W form
            const bool p = (insn >> 10) & 1, u = (insn >> 9) & 1, w = (insn >> 8) & 1;
            const int64_t imm = insn & 0xFF;
            const int64_t off = u ? imm : -imm;
            if (p) addr_imm = off;
            wb = w && !(load && rt == rn);
            wb_off = off;
            (void)rn_plus;
        } else {   // register offset, LSL #imm2
            const unsigned rm = insn & 0xF;
            if (rm == 15) return;
            idx_slot = t.reg(rm);
            idx_shift = (insn >> 4) & 3;
        }
        native(false, false);
        if (load) {
            if (sign) t.lds(t.reg(rt), addr_slot, idx_slot, idx_shift, addr_imm, size);
            else t.ld(t.reg(rt), addr_slot, idx_slot, idx_shift, addr_imm, size);
            if (wb) t.bini(BinOp::Add, 4, t.reg(rn), t.reg(rn), wb_off);
        } else {
            t.st(addr_slot, idx_slot, idx_shift, addr_imm, src(t, rt, pc), size);
            if (wb) t.bini(BinOp::Add, 4, t.reg(rn), t.reg(rn), wb_off);
            t.flush_check(next_pc);
        }
        return;
    }
    if ((op2b & 0x70) == 0x20) {   // data processing, register
        const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF, rm = insn & 0xF;
        const unsigned op1b = (insn >> 20) & 0xF, op2c = (insn >> 4) & 0xF;
        if (rd == 15 || rm == 15) return;
        if (op2c == 0 && (op1b & 8) == 0) {   // LSL / LSR / ASR / ROR (register)
            if (rn == 15 || ((op1b & 1) && flags_live)) return;
            native(false, false);
            Op& o = t.emit(OpKind::A32Shift);
            o.cc = static_cast<uint8_t>((op1b >> 1) & 3); o.d = t.reg(rd); o.a = t.reg(rn); o.b = t.reg(rm);
            return;
        }
        if ((op2c & 8) && (op1b & 8) == 0) {   // the extends
            if (op1b == 2 || op1b == 3 || op1b > 5) return;
            native(false, false);
            const unsigned rot = ((insn >> 4) & 3) * 8;
            int x = t.reg(rm);
            if (rot) { x = t.newtmp(); t.bini(BinOp::Ror, 4, x, t.reg(rm), rot); }
            const int w = (op1b == 0 || op1b == 1) ? 2 : 1;
            const bool sgn = op1b == 0 || op1b == 4;
            if (rn == 15) {
                t.ext(t.reg(rd), x, w, sgn, 4);
            } else {   // the accumulate forms
                const int e = t.newtmp();
                t.ext(e, x, w, sgn, 4);
                t.bin(BinOp::Add, 4, t.reg(rd), e, t.reg(rn));
            }
            return;
        }
        if (rn == 15) return;
        if ((op1b & 0xC) == 0x8 && (op2c & 0xC) == 0x8) {   // REV, REV16, RBIT, REVSH, CLZ
            switch (((op1b & 3) << 2) | (op2c & 3)) {
            case 0x4: native(false, false); t.un(UnOp::Rev, t.reg(rd), t.reg(rm)); return;
            case 0x5: native(false, false); t.un(UnOp::Rev16, t.reg(rd), t.reg(rm)); return;
            case 0x6: native(false, false); t.un(UnOp::Rbit, t.reg(rd), t.reg(rm)); return;
            case 0x7: {
                native(false, false);
                const int x = t.newtmp();
                t.un(UnOp::Rev16, x, t.reg(rm));
                t.ext(t.reg(rd), x, 2, true, 4);
                return;
            }
            case 0xC: native(false, false); t.un(UnOp::Clz, t.reg(rd), t.reg(rm)); return;
            default: return;
            }
        }
        return;
    }
    if ((op2b & 0x78) == 0x30) {   // multiply, multiply accumulate
        const unsigned o1 = (insn >> 20) & 7, o2 = (insn >> 4) & 3;
        const unsigned rn = (insn >> 16) & 0xF, ra = (insn >> 12) & 0xF;
        const unsigned rd = (insn >> 8) & 0xF, rm = insn & 0xF;
        if (o1 != 0 || rd == 15 || rn == 15 || rm == 15) return;
        if (o2 == 0) {   // MUL / MLA
            native(false, false);
            if (ra == 15) { t.bin(BinOp::Mul, 4, t.reg(rd), t.reg(rn), t.reg(rm)); return; }
            const int p = t.newtmp();
            t.bin(BinOp::Mul, 4, p, t.reg(rn), t.reg(rm));
            t.bin(BinOp::Add, 4, t.reg(rd), p, t.reg(ra));
        } else if (o2 == 1 && ra != 15) {   // MLS
            native(false, false);
            const int p = t.newtmp();
            t.bin(BinOp::Mul, 4, p, t.reg(rn), t.reg(rm));
            t.bin(BinOp::Sub, 4, t.reg(rd), t.reg(ra), p);
        }
        return;
    }
    if ((op2b & 0x78) == 0x38) {   // long multiply, divide
        const unsigned o1 = (insn >> 20) & 7;
        const unsigned rn = (insn >> 16) & 0xF, rdlo = (insn >> 12) & 0xF;
        const unsigned rdhi = (insn >> 8) & 0xF, rm = insn & 0xF;
        if (rn == 15 || rm == 15 || rdhi == 15) return;
        if (o1 == 1 || o1 == 3) {   // SDIV / UDIV
            native(false, false);
            t.bin(o1 == 1 ? BinOp::SDiv0 : BinOp::UDiv0, 4, t.reg(rdhi), t.reg(rn), t.reg(rm));
            return;
        }
        if ((o1 == 0 || o1 == 2) && rdlo != 15 && rdlo != rdhi) {   // SMULL / UMULL
            native(false, false);
            const bool sgn = o1 == 0;
            const int na = t.newtmp(), nb = t.newtmp(), p = t.newtmp(), hi = t.newtmp();
            t.ext(na, t.reg(rn), 4, sgn, 8);
            t.ext(nb, t.reg(rm), 4, sgn, 8);
            t.bin(BinOp::Mul, 8, p, na, nb);
            t.bini(BinOp::Shr, 8, hi, p, 32);
            t.ext(t.reg(rdlo), p, 4, false, 8);
            t.mov(t.reg(rdhi), hi);
        }
        return;
    }
}

// Whether an instruction left to the interpreter might change the pc.
bool may_branch_t16(uint16_t insn) {
    if ((insn & 0xF800) == 0xE000) return true;                  // B
    if ((insn & 0xF000) == 0xD000) return true;                  // conditional branch, SVC
    if ((insn & 0xFF00) == 0x4700) return true;                  // BX, BLX
    if ((insn & 0xFC00) == 0x4400) return (insn & 0x87) == 0x87; // ADD / MOV to the pc (CMP does not write)
    if ((insn & 0xFE00) == 0xBC00) return (insn & 0x100) != 0;   // POP {..., pc}
    if ((insn & 0xF500) == 0xB100) return true;                  // CBZ / CBNZ
    return false;
}
bool may_branch_t32(uint32_t insn) {
    const uint32_t hw1 = insn >> 16, hw2 = insn & 0xFFFF;
    if ((hw1 & 0xF800) == 0xF000 && (hw2 & 0x8000)) return true;   // branches, BL, BLX
    if ((hw2 >> 12) == 0xF) return true;                           // a load into the pc
    if (((hw2 >> 8) & 0xF) == 0xF && (hw1 & 0xFE00) == 0xEA00) return true;   // data processing to the pc
    if ((hw1 & 0xFE00) == 0xE800 && (hw2 & 0x8000)) return true;   // LDM / POP with the pc
    if ((hw1 & 0xFFF0) == 0xE8D0 && (hw2 & 0xFFE0) == 0xF000) return true;   // TBB / TBH
    if (((hw2 >> 8) & 0xF) == 0xF && (hw1 & 0xF800) == 0xF000) return true;  // immediate forms to the pc
    return false;
}

}  // namespace

bool translate_thumb_block(GCpu& cpu, const GuestSlots& slots, uint64_t pc, IrBlock& out) {
    const uint8_t* end = nullptr;
    const uint8_t* p0 = nullptr;
    try {
        p0 = cpu.mem().code_window(pc, &end);
    } catch (...) {
        return false;
    }
    const uint64_t to_granule_end = ((pc >> kCodeBits) + 1) << kCodeBits;
    uint64_t room = to_granule_end - pc;
    if (static_cast<uint64_t>(end - p0) < room) room = static_cast<uint64_t>(end - p0);
    if (room < 2) return false;

    struct Plan {
        uint32_t w;
        uint8_t len;
        bool forced;        // part of an IT block with something in it the translator does not do: interpreted
        bool it_head;       // the IT instruction of a translated IT block: nothing to run
        bool in_it;         // a member of a translated IT block, which runs only if `itcond` holds
        uint8_t itcond;
        Info info;
    };
    std::vector<Plan> plans;
    size_t off = 0;
    auto fetch = [&](size_t at, Plan& pl) {
        uint16_t hw1;
        std::memcpy(&hw1, p0 + at, 2);
        pl.len = (hw1 & 0xF800) >= 0xE800 ? 4 : 2;
        if (at + pl.len > room) return false;
        if (pl.len == 4) {
            uint16_t hw2;
            std::memcpy(&hw2, p0 + at + 2, 2);
            pl.w = (static_cast<uint32_t>(hw1) << 16) | hw2;
        } else {
            pl.w = hw1;
        }
        return true;
    };
    // Pass 1: what each instruction is.
    while (plans.size() < 64) {
        Plan pl{};
        if (!fetch(off, pl)) break;
        const uint32_t cur = static_cast<uint32_t>(pc) + static_cast<uint32_t>(off);
        if (pl.len == 2 && (pl.w & 0xFF00) == 0xBF00 && (pl.w & 0xF) != 0) {
            // IT: the instruction and the ones it covers
            unsigned n = 0;
            while (!((pl.w >> n) & 1)) ++n;
            const unsigned count = 4 - n;
            const unsigned firstcond = (pl.w >> 4) & 0xF, mask = pl.w & 0xF;
            std::vector<Plan> group;
            pl.it_head = true;
            pl.info.native = true;
            pl.info.reads = true;
            group.push_back(pl);
            size_t at = off + pl.len;
            bool ok = true, all_native = true;
            for (unsigned k = 0; k < count; ++k) {
                Plan m{};
                if (!fetch(at, m)) { ok = false; break; }
                m.in_it = true;
                // The condition of the k-th instruction: firstcond, with the low bit taken
                // from the mask for all but the first.
                m.itcond = static_cast<uint8_t>(k == 0 ? firstcond : ((firstcond & 0xE) | ((mask >> (3 - (k - 1))) & 1)));
                // Can it be translated?
                IrBlock scratch;
                Tr dry(slots, scratch);
                dry.pcoff = 4;
                const uint32_t mcur = static_cast<uint32_t>(pc) + static_cast<uint32_t>(at);
                if (m.len == 2) emit_t16(dry, static_cast<uint16_t>(m.w), mcur, true, m.info, true);
                else emit_t32(dry, m.w, mcur, true, m.info);
                if (!m.info.native) all_native = false;
                group.push_back(m);
                at += m.len;
            }
            if (!ok) break;   // the block ends before the IT; the next one starts there
            Plan& last = group.back();
            const bool ends = last.info.native ? last.info.term
                              : (last.len == 2 ? may_branch_t16(static_cast<uint16_t>(last.w)) : may_branch_t32(last.w));
            if (all_native) {
                // Conditional execution can skip any of them, so none certainly writes the flags,
                // and each may read them.  A branch ending the group falls through when it is
                // skipped: the block's closing exit covers that.
                for (Plan& g : group) {
                    g.info.reads = true;
                    g.info.writes_all = false;
                    g.info.term = false;
                }
            } else {
                for (Plan& g : group) { g.forced = true; g.in_it = false; g.it_head = false; g.info = Info(); }
                group.back().info.term = ends;
            }
            for (const Plan& g : group) plans.push_back(g);
            off = at;
            if (ends) break;
            continue;
        }
        IrBlock scratch;
        Tr dry(slots, scratch);
        dry.pcoff = 4;
        if (pl.len == 2) emit_t16(dry, static_cast<uint16_t>(pl.w), cur, true, pl.info, false);
        else emit_t32(dry, pl.w, cur, true, pl.info);
        if (!pl.info.native) {
            if (pl.len == 2 ? may_branch_t16(static_cast<uint16_t>(pl.w)) : may_branch_t32(pl.w)) pl.info.term = true;
        }
        plans.push_back(pl);
        off += pl.len;
        if (pl.info.term) break;
    }
    if (plans.empty()) return false;
    const size_t n = plans.size();

    // Which flag writes can anyone see?  At the end of the block, everyone can.
    std::vector<char> live_after(n);
    {
        bool live = true;
        for (size_t i = n; i-- > 0;) {
            live_after[i] = live;
            if (plans[i].it_head) { live = true; continue; }
            if (plans[i].forced || !plans[i].info.native) { live = true; continue; }
            if (plans[i].info.writes_all) live = false;
            if (plans[i].info.reads) live = true;
        }
    }

    out.start = pc | 1;
    Tr t(slots, out);
    t.pcoff = 4;
    uint32_t cur = static_cast<uint32_t>(pc);
    for (size_t i = 0; i < n; ++i) {
        const Plan& pl = plans[i];
        t.cur_off = cur - static_cast<uint32_t>(pc);
        t.tmp_next = 1;
        t.stored = false;
        t.last = i + 1 == n;
        const uint32_t next = cur + pl.len;
        Info info;
        const size_t before = out.ops.size();
        ++out.insns;
        size_t skip_at = static_cast<size_t>(-1);
        if (pl.it_head) {
            info.native = true;   // nothing to run: the conditions are in the ops that follow
        } else if (!pl.forced) {
            if (pl.in_it && pl.itcond != 0xE) {
                Op& s = t.emit(OpKind::A32Skip);
                s.cc = pl.itcond;
                skip_at = out.ops.size() - 1;
            }
            if (pl.len == 2) emit_t16(t, static_cast<uint16_t>(pl.w), cur, live_after[i] != 0, info, pl.in_it);
            else emit_t32(t, pl.w, cur, live_after[i] != 0, info);
            if (skip_at != static_cast<size_t>(-1)) out.ops[skip_at].imm = static_cast<int64_t>(out.ops.size() - skip_at - 1);
        }
        if (!info.native) {
            --out.insns;
            out.ops.resize(before);
            Op& o = t.emit(OpKind::Interp);
            o.imm = static_cast<int64_t>(cur);
            o.imm2 = static_cast<int64_t>(next | 1);
            o.c = static_cast<int32_t>(pl.len == 2 ? (pl.w >> 8) & 0xFF : 0x100 | ((pl.w >> 20) & 0xFF));   // for the statistics
            if (pl.info.effects || info.effects) out.effects = true;
        }
        cur = next;
    }
    out.guest_bytes = cur - static_cast<uint32_t>(pc);
    if (!(plans.back().info.native && plans.back().info.term)) {
        t.cur_off = cur - static_cast<uint32_t>(pc);
        Op& o = t.emit(OpKind::Exit);
        o.imm = static_cast<int64_t>(cur);
    }
    return out.insns > 0;
}

}  // namespace jit
