// 32-bit ARM frontend: guest words -> IR.  ARM state only so far; Thumb code runs in
// the interpreter.
//
// Every A32 instruction is four bytes.  The decoding follows cpu.cpp group for group
// (it is the specification this has to agree with - the verifier compares the two).
// Whatever is not translated becomes an `Interp` op, which runs that one instruction
// through the interpreter: all of VFP and NEON, the exclusives, status-register
// moves, saturating and parallel arithmetic, and so on.  Coverage grows one encoding
// at a time.
//
// Registers are slots 0..14 (r0..r14).  R15 reads as the instruction's address plus 8,
// which is known when translating, so a read of it is a constant in a temporary.  A
// write to R15 is a branch and ends the block.
//
// Almost every instruction is conditional.  A conditional one is preceded by an
// `A32ExitIf`: when the condition fails the block is left at the next instruction,
// which is simply what skipping it means.
#include "arm_tr.h"

namespace jit {

namespace {

using arm::Cpu;

// ---- the translator proper ------------------------------------------------------

void emit_body(Tr& t, uint32_t insn, uint64_t pc64, bool flags_live, Info& info);

// Translates one instruction.  Emits IR and returns with info.native set, or emits
// nothing and leaves info.native clear (the caller then emits an Interp op).
void emit_insn(Tr& t, uint32_t insn, uint64_t pc64, bool flags_live, Info& info) {
    t.pc_tmp = -1;
    emit_body(t, insn, pc64, flags_live, info);
    if (info.native && t.pc_tmp >= 0) {   // a data-processing instruction wrote R15
        info.term = true;
        Op& o = t.emit(OpKind::A32Bx);
        o.a = t.pc_tmp;
    }
}

void emit_body(Tr& t, uint32_t insn, uint64_t pc64, bool flags_live, Info& info) {
    const uint32_t pc = static_cast<uint32_t>(pc64);
    const uint32_t next_pc = pc + 4;
    const unsigned cond = insn >> 28;
    if (cond == 0xF) { info.effects = true; return; }   // the unconditional space: BLX, NEON, barriers (CLREX)

    // Marks the instruction as translated and, when it is conditional, emits the test
    // that skips it.
    auto native = [&](bool reads, bool writes_all) {
        info.native = true;
        info.reads = reads || cond != 0xE;
        info.writes_all = writes_all && cond == 0xE;
        if (cond != 0xE) {
            Op& o = t.emit(OpKind::A32ExitIf);
            o.cc = static_cast<uint8_t>(cond);
            o.imm = static_cast<int64_t>(next_pc);
        }
    };

    // A load, a store, or a plain move leaves the flags alone: the interpreted ones
    // among them should not keep earlier flag writes alive.
    switch ((insn >> 25) & 7) {
    // ---- data processing, register forms and the groups hiding in them -------------
    case 0: {
        const unsigned op1 = (insn >> 20) & 0x1F, op2 = (insn >> 4) & 0xF;
        const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 12) & 0xF, rm = insn & 0xF;

        if (op2 == 0x9) {
            if ((op1 & 0x10) != 0) { info.reads = false; info.effects = true; return; }   // the exclusives and SWP: the monitor is not in the state the verifier compares
            // multiplies
            const unsigned op = (insn >> 21) & 7;
            if ((insn >> 20) & 1) return;                            // the flag-setting forms
            const unsigned mrd = (insn >> 16) & 0xF, ra = (insn >> 12) & 0xF, ms = (insn >> 8) & 0xF, mn = insn & 0xF;
            if (mrd == 15 || ms == 15 || mn == 15) return;
            switch (op) {
            case 0:   // MUL
                native(false, false);
                t.bin(BinOp::Mul, 4, t.reg(mrd), t.reg(mn), t.reg(ms));
                return;
            case 1: case 3: {   // MLA, MLS
                if (ra == 15) return;
                native(false, false);
                const int p = t.newtmp();
                t.bin(BinOp::Mul, 4, p, t.reg(mn), t.reg(ms));
                if (op == 1) t.bin(BinOp::Add, 4, t.reg(mrd), p, t.reg(ra));
                else t.bin(BinOp::Sub, 4, t.reg(mrd), t.reg(ra), p);
                return;
            }
            case 4: case 6: case 5: case 7: {   // UMULL SMULL UMLAL SMLAL: RdLo = ra, RdHi = mrd
                if (ra == 15 || ra == mrd) return;
                native(false, false);
                const bool sgn = op >= 6;
                const int na = t.newtmp(), nb = t.newtmp(), p = t.newtmp();
                t.ext(na, t.reg(mn), 4, sgn, 8);
                t.ext(nb, t.reg(ms), 4, sgn, 8);
                t.bin(BinOp::Mul, 8, p, na, nb);
                if (op == 5 || op == 7) {
                    const int hi = t.newtmp(), acc = t.newtmp();
                    t.bini(BinOp::Shl, 8, hi, t.reg(mrd), 32);
                    t.bin(BinOp::Or, 8, acc, hi, t.reg(ra));
                    t.bin(BinOp::Add, 8, p, p, acc);
                }
                const int hi2 = t.newtmp();
                t.bini(BinOp::Shr, 8, hi2, p, 32);
                t.ext(t.reg(ra), p, 4, false, 8);
                t.mov(t.reg(mrd), hi2);
                return;
            }
            default: return;
            }
        }
        if ((op2 & 0x9) == 0x9) {   // halfword and signed-byte loads and stores
            const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
            const bool imm = (insn >> 22) & 1, w = (insn >> 21) & 1, load = (insn >> 20) & 1;
            const unsigned kind = (insn >> 5) & 3;
            if (kind == 0) return;
            if (!load && kind != 1) return;                          // LDRD / STRD
            if (load && rd == 15) return;
            if (!imm && rm == 15) return;
            info.reads = false;
            native(false, false);
            const int sz = (kind == 2) ? 1 : 2;
            const bool sgn = load && kind >= 2;
            const bool wb = (!p || w) && rn != 15;
            // the base: R15 reads as pc + 8
            const int base = src(t, rn, pc);
            int addr_slot = base;
            int64_t addr_imm = 0;
            const int64_t off_imm = imm ? static_cast<int64_t>(((insn >> 4) & 0xF0) | (insn & 0xF)) : 0;
            // A register offset is added before the access: a load into Rm must not
            // change the written-back address.
            int ea = -1;
            if (imm) {
                if (p) addr_imm = u ? off_imm : -off_imm;
            } else {
                ea = t.newtmp();
                t.bin(u ? BinOp::Add : BinOp::Sub, 4, ea, base, t.reg(rm));
                if (p) addr_slot = ea;
            }
            auto writeback = [&]() {
                if (imm) t.bini(u ? BinOp::Add : BinOp::Sub, 4, t.reg(rn), t.reg(rn), off_imm);
                else t.mov(t.reg(rn), ea);
            };
            if (load) {
                if (sgn) t.lds(t.reg(rd), addr_slot, t.s.zero, 0, addr_imm, sz);
                else t.ld(t.reg(rd), addr_slot, t.s.zero, 0, addr_imm, sz);
                if (wb && rd != rn) writeback();
            } else {
                t.st(addr_slot, t.s.zero, 0, addr_imm, src(t, rd, pc), sz);
                if (wb) writeback();
                t.flush_check(next_pc);
            }
            return;
        }
        if ((op1 & 0x19) == 0x10) {   // the miscellaneous group
            if ((op2 & 0x8) != 0) return;   // halfword multiplies
            const unsigned op = (insn >> 21) & 3, kind = (insn >> 4) & 7;
            if (kind == 1 && op == 1 && rm != 15) {   // BX
                native(false, false);
                info.term = true;
                Op& o = t.emit(OpKind::A32Bx);
                o.a = t.reg(rm);
                return;
            }
            if (kind == 3 && op == 1 && rm != 15) {   // BLX (register)
                native(false, false);
                info.term = true;
                const int target = t.newtmp();
                t.mov(target, t.reg(rm));
                t.movi(t.reg(14), static_cast<int64_t>(next_pc));
                Op& o = t.emit(OpKind::A32Bx);
                o.a = target;
                return;
            }
            if (kind == 1 && op == 3 && rd != 15 && rm != 15) {   // CLZ
                native(false, false);
                t.un(UnOp::Clz, t.reg(rd), t.reg(rm));
                return;
            }
            return;
        }
        // data-processing register: shifted by an immediate (bit 4 clear) or by a register
        const bool regshift = (op2 & 1) != 0;
        const unsigned op = (insn >> 21) & 0xF;
        const bool setflags = (insn >> 20) & 1;
        if (!is_compare(op) && rd == 15 && setflags) return;
        bool plain;
        if (setflags && is_logical(op)) {
            // the shifter carry matters: only the unshifted form is translated
            if (regshift || ((insn >> 5) & 3) != 0 || ((insn >> 7) & 0x1F) != 0) return;
        }
        if (!regshift && ((insn >> 5) & 3) == 3 && ((insn >> 7) & 0x1F) == 0) return;   // RRX
        native(reads_flags(op, setflags), setflags && !is_logical(op));
        int bslot;
        if (regshift) {
            const int v = src(t, rm, pc), amt = src(t, (insn >> 8) & 0xF, pc);
            bslot = t.newtmp();
            Op& o = t.emit(OpKind::A32Shift);
            o.cc = static_cast<uint8_t>((insn >> 5) & 3); o.d = bslot; o.a = v; o.b = amt;
        } else {
            bslot = shifted(t, rm, (insn >> 5) & 3, (insn >> 7) & 0x1F, pc, &plain);
        }
        const int aslot = (op == Cpu::kMOV || op == Cpu::kMVN) ? t.s.zero : src(t, rn, pc);
        const bool fl = setflags && flags_live;
        const int dslot = is_compare(op) ? t.newtmp() : dreg(t, rd);
        if (is_compare(op) && !fl) return;
        // plain operations without flags use the generic ops
        if (!fl && op != Cpu::kADC && op != Cpu::kSBC && op != Cpu::kRSC) {
            switch (op) {
            case Cpu::kAND: t.bin(BinOp::And, 4, dslot, aslot, bslot); return;
            case Cpu::kEOR: t.bin(BinOp::Xor, 4, dslot, aslot, bslot); return;
            case Cpu::kORR: t.bin(BinOp::Or, 4, dslot, aslot, bslot); return;
            case Cpu::kBIC: t.bin(BinOp::Andn, 4, dslot, aslot, bslot); return;
            case Cpu::kSUB: t.bin(BinOp::Sub, 4, dslot, aslot, bslot); return;
            case Cpu::kRSB: t.bin(BinOp::Sub, 4, dslot, bslot, aslot); return;
            case Cpu::kADD: t.bin(BinOp::Add, 4, dslot, aslot, bslot); return;
            case Cpu::kMOV: t.mov(dslot, bslot); return;
            case Cpu::kMVN: t.bini(BinOp::Xor, 4, dslot, bslot, 0xFFFFFFFFll); return;
            default: break;
            }
        }
        Op& o = t.emit(OpKind::A32Dp);
        o.cc = static_cast<uint8_t>(op); o.d = dslot; o.a = aslot; o.b = bslot;
        o.imm2 = -1;
        if (fl) o.flags |= kFlagEager;
        return;
    }

    // ---- data processing, immediate forms -----------------------------------------
    case 1: {
        const unsigned op1 = (insn >> 20) & 0x1F;
        const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 12) & 0xF;
        const uint32_t imm12 = insn & 0xFFF;
        if (op1 == 0x10) {   // MOVW
            if (rd == 15) return;
            native(false, false);
            t.movi(t.reg(rd), static_cast<int64_t>((((insn >> 16) & 0xF) << 12) | imm12));
            return;
        }
        if (op1 == 0x14) {   // MOVT
            if (rd == 15) return;
            native(false, false);
            const uint32_t hi = ((((insn >> 16) & 0xF) << 12) | imm12) << 16;
            t.bini(BinOp::And, 4, t.reg(rd), t.reg(rd), 0xFFFF);
            t.bini(BinOp::Or, 4, t.reg(rd), t.reg(rd), hi);
            return;
        }
        if ((op1 & 0x1B) == 0x12) {   // MSR (immediate) and the hints
            if (rn == 0 && ((insn >> 22) & 1) == 0) { native(false, false); return; }   // NOP, YIELD...
            return;
        }
        const unsigned op = (insn >> 21) & 0xF;
        const bool setflags = (insn >> 20) & 1;
        if (!is_compare(op) && rd == 15 && setflags) return;
        const arm::ShiftResult sr = arm::expand_imm_c(imm12, false);
        const unsigned rot = ((imm12 >> 8) & 0xF) * 2;
        native(reads_flags(op, setflags), setflags && !is_logical(op));
        const bool fl = setflags && flags_live;
        if (is_compare(op) && !fl) return;
        const int dslot = is_compare(op) ? t.newtmp() : dreg(t, rd);
        const int64_t imm = static_cast<int64_t>(sr.result);
        if (!fl && op != Cpu::kADC && op != Cpu::kSBC && op != Cpu::kRSC && op != Cpu::kRSB) {
            const int aslot = src(t, rn, pc);
            switch (op) {
            case Cpu::kAND: t.bini(BinOp::And, 4, dslot, aslot, imm); return;
            case Cpu::kEOR: t.bini(BinOp::Xor, 4, dslot, aslot, imm); return;
            case Cpu::kORR: t.bini(BinOp::Or, 4, dslot, aslot, imm); return;
            case Cpu::kBIC: t.bini(BinOp::Andn, 4, dslot, aslot, imm); return;
            case Cpu::kSUB: t.bini(BinOp::Sub, 4, dslot, aslot, imm); return;
            case Cpu::kADD: t.bini(BinOp::Add, 4, dslot, aslot, imm); return;
            case Cpu::kMOV: t.movi(dslot, imm); return;
            case Cpu::kMVN: t.movi(dslot, static_cast<int64_t>(~sr.result)); return;
            default: break;
            }
        }
        Op& o = t.emit(OpKind::A32DpI);
        o.cc = static_cast<uint8_t>(op); o.d = dslot;
        o.a = (op == Cpu::kMOV || op == Cpu::kMVN) ? t.s.zero : src(t, rn, pc);
        o.imm = imm;
        // the carry the immediate produces: unchanged when there is no rotation
        o.imm2 = rot == 0 ? int64_t(-1) : int64_t(sr.result >> 31);
        if (fl) o.flags |= kFlagEager;
        return;
    }

    // ---- single data transfer: immediate offset, and register offset ---------------
    case 2:
    case 3: {
        if (((insn >> 25) & 7) == 3 && (insn & 0x10)) break;   // media instructions
        const bool reg_offset = (insn >> 25) & 1;
        const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
        const bool byte = (insn >> 22) & 1, w = (insn >> 21) & 1, load = (insn >> 20) & 1;
        const unsigned rn = (insn >> 16) & 0xF, rt = (insn >> 12) & 0xF;
        if (load && rt == 15) return;
        const unsigned rm = insn & 0xF;
        int off_slot = -1;
        int64_t off_imm = 0;
        info.reads = false;
        if (!reg_offset) {
            off_imm = insn & 0xFFF;
        } else {
            if (rm == 15) return;
            if ((((insn >> 5) & 3) == 3) && ((insn >> 7) & 0x1F) == 0) return;   // RRX
        }
        native(false, false);
        const int sz = byte ? 1 : 4;
        const bool wb = (!p || w) && rn != 15 && !(load && rt == rn);
        if (reg_offset) {
            bool plain;
            off_slot = shifted(t, rm, (insn >> 5) & 3, (insn >> 7) & 0x1F, pc, &plain);
        }
        int addr_slot, ea = -1;
        int64_t addr_imm = 0;
        if (rn == 15) {
            // a literal: the base is the aligned pc, and nothing is written back
            const uint32_t base = (pc + 8) & ~3u;
            if (reg_offset) {
                const int b = t.newtmp();
                t.movi(b, base);
                ea = t.newtmp();
                t.bin(u ? BinOp::Add : BinOp::Sub, 4, ea, b, off_slot);
                addr_slot = p ? ea : b;
            } else {
                addr_slot = t.s.zero;
                addr_imm = p ? static_cast<int64_t>(u ? base + off_imm : base - off_imm) : static_cast<int64_t>(base);
            }
        } else if (!reg_offset) {
            addr_slot = t.reg(rn);
            addr_imm = p ? (u ? off_imm : -off_imm) : 0;
        } else {
            // the offset is added before the access: a load into Rm must not change
            // the address that is written back
            ea = t.newtmp();
            t.bin(u ? BinOp::Add : BinOp::Sub, 4, ea, t.reg(rn), off_slot);
            addr_slot = p ? ea : t.reg(rn);
        }
        auto writeback = [&]() {
            if (!wb) return;
            if (!reg_offset) t.bini(u ? BinOp::Add : BinOp::Sub, 4, t.reg(rn), t.reg(rn), off_imm);
            else t.mov(t.reg(rn), ea);
        };
        if (load) {
            t.ld(t.reg(rt), addr_slot, t.s.zero, 0, addr_imm, sz);
            writeback();
        } else {
            t.st(addr_slot, t.s.zero, 0, addr_imm, src(t, rt, pc), sz);
            writeback();
            t.flush_check(next_pc);
        }
        return;
    }
    default: break;
    }

    // ---- the rest: media instructions, block transfers, branches, system -----------
    switch ((insn >> 25) & 7) {
    case 3: {   // media
        const unsigned op1 = (insn >> 20) & 0x1F;
        const unsigned op2 = (insn >> 5) & 7;
        const unsigned rd = (insn >> 12) & 0xF, rn = (insn >> 16) & 0xF, rm = insn & 0xF;
        const unsigned bfop = (insn >> 21) & 0xF;
        if ((bfop == 13 || bfop == 15) && (op2 & 3) == 2) {   // SBFX / UBFX
            const unsigned lsb = (insn >> 7) & 0x1F, width = ((insn >> 16) & 0x1F) + 1;
            if (rd == 15 || rm == 15 || lsb + width > 32) return;
            native(false, false);
            if (bfop == 15) {
                if (width == 32) { t.mov(t.reg(rd), t.reg(rm)); return; }
                const int x = t.newtmp();
                t.bini(BinOp::Shr, 4, x, t.reg(rm), lsb);
                t.bini(BinOp::And, 4, t.reg(rd), x, (1ll << width) - 1);
            } else {
                if (width == 32) { t.mov(t.reg(rd), t.reg(rm)); return; }
                const int x = t.newtmp();
                t.bini(BinOp::Shl, 4, x, t.reg(rm), 32 - lsb - width);
                t.bini(BinOp::Sar, 4, t.reg(rd), x, 32 - width);
            }
            return;
        }
        if (bfop == 14 && (op2 & 3) == 0) {   // BFC / BFI
            const unsigned lsb = (insn >> 7) & 0x1F, msb = (insn >> 16) & 0x1F;
            if (msb < lsb || rd == 15 || rm == 15) return;
            const uint32_t mask = msb == 31 ? (0xFFFFFFFFu << lsb) : (((1u << (msb - lsb + 1)) - 1) << lsb);
            native(false, false);
            if (rn == 15) {   // BFC
                t.bini(BinOp::And, 4, t.reg(rd), t.reg(rd), static_cast<int64_t>(~mask));
                return;
            }
            const int x = t.newtmp();
            t.bini(BinOp::Shl, 4, x, t.reg(rm), lsb);
            t.bini(BinOp::And, 4, x, x, mask);
            t.bini(BinOp::And, 4, t.reg(rd), t.reg(rd), static_cast<int64_t>(~mask));
            t.bin(BinOp::Or, 4, t.reg(rd), t.reg(rd), x);
            return;
        }
        if (op1 == 0x1F && op2 == 7) return;
        if ((op1 & 0x18) == 0x10) {   // multiplies and divides
            const unsigned rdhi = (insn >> 16) & 0xF;
            const unsigned rs = (insn >> 8) & 0xF;
            if ((op1 & 7) == 1 || (op1 & 7) == 3) {   // SDIV / UDIV
                if (rdhi == 15 || rm == 15 || rs == 15) return;
                native(false, false);
                t.bin((op1 & 7) == 1 ? BinOp::SDiv0 : BinOp::UDiv0, 4, t.reg(rdhi), t.reg(rm), t.reg(rs));
            }
            return;
        }
        if ((op1 & 0x18) == 0x18 && op2 == 0) return;   // USAD8
        if ((op1 & 0x18) == 0) return;                  // parallel arithmetic
        // packing, extension, reversal
        const unsigned rot = ((insn >> 10) & 3) * 8;
        const unsigned sub = (insn >> 20) & 7;
        if (op2 == 3) {   // the extend family
            if (sub != 2 && sub != 3 && sub != 6 && sub != 7) return;
            if (rd == 15 || rm == 15) return;
            native(false, false);
            int x = t.reg(rm);
            if (rot) {
                x = t.newtmp();
                t.bini(BinOp::Ror, 4, x, t.reg(rm), rot);
            }
            const bool sgn = sub == 2 || sub == 3;
            const int w = (sub == 2 || sub == 6) ? 1 : 2;
            if (rn == 15) {
                t.ext(t.reg(rd), x, w, sgn, 4);
            } else {
                const int e = t.newtmp();
                t.ext(e, x, w, sgn, 4);
                t.bin(BinOp::Add, 4, t.reg(rd), e, t.reg(rn));
            }
            return;
        }
        if (sub == 0) return;   // PKH, SEL
        if ((op2 & 1) == 0 && (sub == 2 || sub == 3 || sub == 6 || sub == 7)) return;   // SSAT, USAT
        if ((sub == 3 || sub == 7) && rd != 15 && rm != 15) {
            if (op2 == 1) {
                native(false, false);
                t.un(sub == 7 ? UnOp::Rbit : UnOp::Rev, t.reg(rd), t.reg(rm));
                return;
            }
            if (op2 == 5) {
                native(false, false);
                if (sub == 7) {   // REVSH
                    const int x = t.newtmp();
                    t.un(UnOp::Rev16, x, t.reg(rm));
                    t.ext(t.reg(rd), x, 2, true, 4);
                } else {
                    t.un(UnOp::Rev16, t.reg(rd), t.reg(rm));
                }
                return;
            }
        }
        return;
    }

    case 4: {   // LDM / STM
        const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
        const bool s = (insn >> 22) & 1, w = (insn >> 21) & 1, load = (insn >> 20) & 1;
        const unsigned rn = (insn >> 16) & 0xF;
        const uint32_t list = insn & 0xFFFF;
        if (s || list == 0 || rn == 15) return;
        if (load && (list & 0x8000)) info.term = true;
        info.reads = false;
        native(false, false);
        block_transfer(t, rn, list, load, u, p, w, pc, next_pc);
        return;
    }

    case 5: {   // B / BL
        const int32_t imm = (static_cast<int32_t>(insn << 8) >> 8) * 4;
        const uint32_t target = pc + 8 + static_cast<uint32_t>(imm);
        const bool link = (insn >> 24) & 1;
        info.term = true;
        if (cond == 0xE || link) {
            native(false, false);
            if (link) t.movi(t.reg(14), static_cast<int64_t>(next_pc));
            t.exit_to(target);
            return;
        }
        info.native = true;
        info.reads = true;
        Op& o = t.emit(OpKind::A32Jcc);
        o.cc = static_cast<uint8_t>(cond);
        o.imm = static_cast<int64_t>(target);
        o.imm2 = static_cast<int64_t>(next_pc);
        return;
    }

    case 6: case 7:   // coprocessors, VFP, SVC
        if (((insn >> 24) & 0xF) == 0xF) info.effects = true;
        return;

    default:
        return;
    }
}

// Whether an instruction the translator left to the interpreter might change the pc.
bool may_branch(uint32_t insn) {
    if ((insn >> 28) == 0xF) return (insn >> 25) == 0x7D;   // BLX (immediate)
    switch ((insn >> 25) & 7) {
        case 0: case 1: case 2: case 3: return ((insn >> 12) & 0xF) == 0xF || ((insn & 0x0FFFFF00u) == 0x012FFF00u);
        case 4: return (insn & 0x8000) != 0;
        case 5: return true;
        default: return false;
    }
}

}  // namespace

bool translate_thumb_block(GCpu& cpu, const GuestSlots& slots, uint64_t pc, IrBlock& out);

bool translate_block(GCpu& cpu, const GuestSlots& slots, uint64_t pc, IrBlock& out) {
    if (pc & 1) return translate_thumb_block(cpu, slots, pc & ~1ull, out);   // Thumb
    if (pc & 3) return false;   // misaligned
    const uint8_t* end = nullptr;
    const uint8_t* p0 = nullptr;
    try {
        p0 = cpu.mem().code_window(pc, &end);
    } catch (...) {
        return false;
    }
    // A block stays inside one 4 KiB granule, which is what invalidation tracks.
    const uint64_t to_granule_end = ((pc >> kCodeBits) + 1) << kCodeBits;
    uint64_t room = to_granule_end - pc;
    if (static_cast<uint64_t>(end - p0) < room) room = static_cast<uint64_t>(end - p0);
    const size_t max_insns = static_cast<size_t>(room / 4) < 64 ? static_cast<size_t>(room / 4) : 64;
    if (max_insns == 0) return false;

    struct Plan {
        uint32_t w;
        Info info;
    };
    std::vector<Plan> plans;
    // Pass 1: what each instruction is.
    for (size_t i = 0; i < max_insns; ++i) {
        Plan pl;
        std::memcpy(&pl.w, p0 + i * 4, 4);
        IrBlock scratch;
        Tr dry(slots, scratch);
        emit_insn(dry, pl.w, pc + i * 4, true, pl.info);
        if (!pl.info.native && may_branch(pl.w)) pl.info.term = true;
        plans.push_back(pl);
        if (pl.info.term) break;
    }
    const size_t n = plans.size();

    // Which flag writes can anyone see?  At the end of the block, everyone can.
    std::vector<char> live_after(n);
    {
        bool live = true;
        for (size_t i = n; i-- > 0;) {
            live_after[i] = live;
            if (plans[i].info.writes_all) live = false;
            if (plans[i].info.reads) live = true;
        }
    }

    out.start = pc;
    Tr t(slots, out);
    uint64_t cur = pc;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t w = plans[i].w;
        t.cur_off = static_cast<uint32_t>(cur - pc);
        t.tmp_next = 1;
        t.stored = false;
        t.last = i + 1 == n;
        const uint64_t next = cur + 4;
        Info info;
        const size_t before = out.ops.size();
        ++out.insns;
        emit_insn(t, w, cur, live_after[i] != 0, info);
        if (!info.native) {
            --out.insns;
            out.ops.resize(before);   // nothing should have been emitted; be sure
            Op& o = t.emit(OpKind::Interp);
            o.imm = static_cast<int64_t>(cur);
            o.imm2 = static_cast<int64_t>(next);
            o.c = static_cast<int32_t>(((w >> 20) & 0xFF) | (((w >> 4) & 1) << 8));   // for the statistics: bits 27..20 and 4
            if (plans[i].info.effects || info.effects) out.effects = true;
        }
        cur = next;
    }
    out.guest_bytes = static_cast<uint32_t>(cur - pc);
    if (!(plans.back().info.native && plans.back().info.term)) {
        t.cur_off = static_cast<uint32_t>(cur - pc);
        Op& o = t.emit(OpKind::Exit);
        o.imm = static_cast<int64_t>(cur);
    }
    return out.insns > 0;
}

}  // namespace jit

