// The T32 (Thumb and Thumb-2) interpreter.
//
// Thumb is not a subset of ARM with shorter encodings; it is a second instruction
// set that happens to run on the same registers, and three of its properties have
// no A32 equivalent at all:
//
//   - **Instructions are two bytes or four**, and which one is decided by the top
//     five bits of the first halfword. A decoder that guesses wrong does not fail
//     -- it resumes in the middle of an instruction and executes the second half of
//     one thing as the first half of another.
//   - **Reading R15 gives the current address + 4**, not + 8, and that holds for a
//     four-byte Thumb instruction too. `pc_val()` is where that lives, and it is
//     the reason nothing here computes a PC-relative address by hand.
//   - **Conditional execution is carried out of band.** Sixteen-bit Thumb has no
//     condition field, so `IT` puts up to four conditions into ITSTATE and the
//     following instructions consume them one at a time. The condition is therefore
//     evaluated *here*, in `exec_thumb`, before the decoder is ever reached -- not
//     inside individual handlers, where forgetting it in one of them would silently
//     execute one instruction that should have been skipped.
//
// Everything that actually computes -- the ALU, the shifter, loads, stores, the
// register lists -- is shared with the A32 file. Only the encodings live here.
#include "cpu.h"
#include <cstdio>

namespace arm {

void Cpu::exec_thumb() {
    const uint16_t hw1 = mem_.read<uint16_t>(pc);
    // 0b11101, 0b11110 and 0b11111 introduce a 32-bit instruction; everything else
    // is 16 bits. This is the whole rule, and getting it wrong desynchronises the
    // instruction stream rather than reporting anything.
    const bool is32 = (hw1 & 0xF800) >= 0xE800;
    uint32_t insn;
    if (is32) {
        insn = (static_cast<uint32_t>(hw1) << 16) | mem_.read<uint16_t>(pc + 2);
        pc += 4;
    } else {
        insn = hw1;
        pc += 2;
    }

    // ITSTATE is consumed before the instruction runs and advanced whether or not
    // it does, because a skipped instruction still uses up its slot in the block.
    // `in_it` remembers what the answer was *for this instruction*, because the
    // advance has already changed it -- see the note on the member.
    in_it = in_it_block();
    if (in_it) {
        const unsigned cond = it_cond();
        it_advance();
        if (!cond_holds(cond)) return;
    }
    if (is32) t32(insn); else t16(static_cast<uint16_t>(insn));
}

// ---------------------------------------------------------------------------
// A6.2 16-bit Thumb instruction encoding
// ---------------------------------------------------------------------------

void Cpu::t16(uint16_t insn) {
    const unsigned op = (insn >> 10) & 0x3F;

    // A6.2.1 Shift (immediate), add, subtract, move, and compare.
    //
    // The mask is 0x38, not 0x30. Bits 15..11 of `00xxx` are the shifts and the
    // three-operand adds; `001xx` is the eight-bit immediate group, and it shares
    // the top two bits with them. Catching both with one test sends
    // `subs r3, #4` into the shift decoder, where it becomes a rotate -- an
    // instruction that computes a plausible number and never terminates the loop
    // it was counting.
    if ((op & 0x38) == 0) {
        const unsigned opc = (insn >> 9) & 0x1F;
        const unsigned rd = insn & 7, rn = (insn >> 3) & 7;
        // The setflags rule for the whole of 16-bit Thumb: flags are written unless
        // the instruction is inside an IT block. A compiler relies on it -- `adds`
        // and `add` are the same encoding outside and inside an IT.
        const bool sf = !in_it;
        if ((opc & 0x1C) != 0x0C) {                       // the three shifts
            const unsigned type = (opc >> 2) & 3;
            const unsigned imm5 = (insn >> 6) & 0x1F;
            unsigned stype, amt;
            decode_imm_shift(type, imm5, &stype, &amt);
            const ShiftResult s = stype == kRRX ? rrx_c(reg(rn), c)
                                                : shift_c(reg(rn), stype, amt, c);
            r[rd] = s.result;
            if (sf) { set_nz(s.result); c = s.carry; }
            return;
        }
        switch (opc & 3) {
            case 0: dp_op(kADD, rd, reg(rn), reg((insn >> 6) & 7), sf, c); return;
            case 1: dp_op(kSUB, rd, reg(rn), reg((insn >> 6) & 7), sf, c); return;
            case 2: dp_op(kADD, rd, reg(rn), (insn >> 6) & 7, sf, c); return;
            default: dp_op(kSUB, rd, reg(rn), (insn >> 6) & 7, sf, c); return;
        }
    }
    if ((op & 0x38) == 0x08) {                            // MOV/CMP/ADD/SUB immediate
        const unsigned rd = (insn >> 8) & 7;
        const uint32_t imm = insn & 0xFF;
        const bool sf = !in_it;
        switch ((insn >> 11) & 3) {
            case 0: dp_op(kMOV, rd, 0, imm, sf, c); return;
            case 1: dp_op(kCMP, rd, reg(rd), imm, true, c); return;
            case 2: dp_op(kADD, rd, reg(rd), imm, sf, c); return;
            default: dp_op(kSUB, rd, reg(rd), imm, sf, c); return;
        }
    }

    // A6.2.2 Data-processing.
    if (op == 0x10) {
        const unsigned opc = (insn >> 6) & 0xF;
        const unsigned rd = insn & 7, rm = (insn >> 3) & 7;
        const bool sf = !in_it;
        switch (opc) {
            case 0x0: dp_op(kAND, rd, reg(rd), reg(rm), sf, c); return;
            case 0x1: dp_op(kEOR, rd, reg(rd), reg(rm), sf, c); return;
            case 0x2: case 0x3: case 0x4: case 0x7: {     // LSL/LSR/ASR/ROR by register
                static const unsigned kType[16] = {0,0,kLSL,kLSR,kASR,0,0,kROR};
                const ShiftResult s = shift_c(reg(rd), kType[opc], reg(rm) & 0xFF, c);
                r[rd] = s.result;
                if (sf) { set_nz(s.result); c = s.carry; }
                return;
            }
            case 0x5: dp_op(kADC, rd, reg(rd), reg(rm), sf, c); return;
            case 0x6: dp_op(kSBC, rd, reg(rd), reg(rm), sf, c); return;
            case 0x8: dp_op(kTST, rd, reg(rd), reg(rm), true, c); return;
            case 0x9: dp_op(kRSB, rd, reg(rm), 0, sf, c); return;    // RSBS Rd, Rm, #0
            case 0xA: dp_op(kCMP, rd, reg(rd), reg(rm), true, c); return;
            case 0xB: dp_op(kCMN, rd, reg(rd), reg(rm), true, c); return;
            case 0xC: dp_op(kORR, rd, reg(rd), reg(rm), sf, c); return;
            case 0xD: {                                   // MUL
                const uint32_t res = reg(rd) * reg(rm);
                r[rd] = res;
                if (sf) set_nz(res);
                return;
            }
            case 0xE: dp_op(kBIC, rd, reg(rd), reg(rm), sf, c); return;
            default: dp_op(kMVN, rd, 0, reg(rm), sf, c); return;
        }
    }

    // A6.2.3 Special data instructions and branch and exchange. The one place
    // 16-bit Thumb can name a high register, which is why BX and the frame-pointer
    // moves live here.
    if (op == 0x11) {
        const unsigned opc = (insn >> 8) & 3;
        const unsigned rm = (insn >> 3) & 0xF;
        const unsigned rd = (insn & 7) | (((insn >> 7) & 1) << 3);
        switch (opc) {
            // `add pc, rX` and `mov pc, rX` are data-processing writes, so they do
            // *not* interwork in Thumb state -- see set_reg_alu().
            case 0: set_reg_alu(rd, reg(rd) + reg(rm)); return;      // ADD (no flags)
            case 1: dp_op(kCMP, 0, reg(rd), reg(rm), true, c); return;
            case 2: set_reg_alu(rd, reg(rm)); return;                // MOV
            default:
                if ((insn >> 7) & 1) {                               // BLX
                    const uint32_t target = reg(rm);
                    set_lr((cur_pc() + 2) | 1);
                    bx_write_pc(target);
                } else {
                    bx_write_pc(reg(rm));                            // BX
                }
                return;
        }
    }

    if ((op & 0x3E) == 0x12) {                            // LDR (literal)
        const unsigned rt = (insn >> 8) & 7;
        const uint32_t addr = (pc_val() & ~3u) + (insn & 0xFF) * 4;
        do_load(rt, addr, 4, false);
        return;
    }

    // A6.2.4 Load/store single data item.
    if ((op & 0x3C) == 0x14) {                            // register offset
        const unsigned rt = insn & 7, rn = (insn >> 3) & 7, rm = (insn >> 6) & 7;
        const uint32_t addr = reg(rn) + reg(rm);
        switch ((insn >> 9) & 7) {
            case 0: do_store(rt, addr, 4); return;
            case 1: do_store(rt, addr, 2); return;
            case 2: do_store(rt, addr, 1); return;
            case 3: do_load(rt, addr, 1, true); return;
            case 4: do_load(rt, addr, 4, false); return;
            case 5: do_load(rt, addr, 2, false); return;
            case 6: do_load(rt, addr, 1, false); return;
            default: do_load(rt, addr, 2, true); return;
        }
    }
    if ((op & 0x38) == 0x18 || (op & 0x3C) == 0x20) {     // immediate offset
        const unsigned rt = insn & 7, rn = (insn >> 3) & 7;
        const unsigned imm5 = (insn >> 6) & 0x1F;
        const bool load = (insn >> 11) & 1;
        unsigned size;
        uint32_t scale;
        switch ((insn >> 12) & 0xF) {
            case 6: size = 4; scale = 4; break;
            case 7: size = 1; scale = 1; break;
            default: size = 2; scale = 2; break;          // 0b1000
        }
        const uint32_t addr = reg(rn) + imm5 * scale;
        if (load) do_load(rt, addr, size, false); else do_store(rt, addr, size);
        return;
    }
    if ((op & 0x3C) == 0x24) {                            // LDR/STR (SP relative)
        const unsigned rt = (insn >> 8) & 7;
        const uint32_t addr = sp() + (insn & 0xFF) * 4;
        if ((insn >> 11) & 1) do_load(rt, addr, 4, false); else do_store(rt, addr, 4);
        return;
    }
    if ((op & 0x3C) == 0x28) {                            // ADR / ADD (SP plus immediate)
        const unsigned rd = (insn >> 8) & 7;
        const uint32_t imm = (insn & 0xFF) * 4;
        r[rd] = ((insn >> 11) & 1) ? sp() + imm : (pc_val() & ~3u) + imm;
        return;
    }

    // A6.2.5 Miscellaneous 16-bit instructions.
    if ((op & 0x3C) == 0x2C) {
        if ((insn & 0xFF00) == 0xB000) {                  // ADD/SUB SP, immediate
            const uint32_t imm = (insn & 0x7F) * 4;
            set_sp((insn & 0x80) ? sp() - imm : sp() + imm);
            return;
        }
        if ((insn & 0xF500) == 0xB100) {                  // CBZ / CBNZ
            const unsigned rn = insn & 7;
            const uint32_t imm = (((insn >> 9) & 1) << 6) | (((insn >> 3) & 0x1F) << 1);
            const bool nonzero = (insn >> 11) & 1;
            if ((reg(rn) != 0) == nonzero) branch_write_pc(pc_val() + imm);
            return;
        }
        if ((insn & 0xFF00) == 0xB200) {                  // SXTH / SXTB / UXTH / UXTB
            const unsigned rd = insn & 7, rm = (insn >> 3) & 7;
            const uint32_t x = reg(rm);
            switch ((insn >> 6) & 3) {
                case 0: r[rd] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(x))); return;
                case 1: r[rd] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(x))); return;
                case 2: r[rd] = x & 0xFFFF; return;
                default: r[rd] = x & 0xFF; return;
            }
        }
        if ((insn & 0xFE00) == 0xB400) {                  // PUSH
            const uint32_t list = (insn & 0xFF) | (((insn >> 8) & 1) << 14);
            do_stm(13, list, false, true, true);
            return;
        }
        if ((insn & 0xFE00) == 0xBC00) {                  // POP
            const uint32_t list = (insn & 0xFF) | (((insn >> 8) & 1) << 15);
            do_ldm(13, list, true, false, true, false);
            return;
        }
        if ((insn & 0xFF00) == 0xBA00) {                  // REV / REV16 / REVSH
            const unsigned rd = insn & 7, rm = (insn >> 3) & 7;
            const uint32_t x = reg(rm);
            switch ((insn >> 6) & 3) {
                case 0: r[rd] = (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24); return;
                case 1: r[rd] = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu); return;
                default:
                    r[rd] = static_cast<uint32_t>(static_cast<int32_t>(
                        static_cast<int16_t>(((x & 0xFF) << 8) | ((x >> 8) & 0xFF))));
                    return;
            }
        }
        if ((insn & 0xFF00) == 0xBF00) {                  // IT and the hints
            const unsigned mask = insn & 0xF;
            if (mask) { itstate = insn & 0xFF; return; }  // IT: firstcond:mask, as encoded
            return;                                       // NOP / YIELD / WFE / WFI / SEV
        }
        if ((insn & 0xFF00) == 0xBE00) fail("BKPT", insn);
        if ((insn & 0xFFE8) == 0xB660) return;            // CPS
        if ((insn & 0xFFF7) == 0xB650) return;            // SETEND
        undefined(insn);
        return;
    }

    if ((op & 0x3C) == 0x30) {                            // STM / LDM
        const unsigned rn = (insn >> 8) & 7;
        const uint32_t list = insn & 0xFF;
        if ((insn >> 11) & 1) {
            // LDM writes back unless the base was loaded, which do_ldm already
            // decides -- so the flag is simply "the encoding has writeback".
            do_ldm(rn, list, true, false, true, false);
        } else {
            do_stm(rn, list, true, false, true);
        }
        return;
    }

    if ((op & 0x3C) == 0x34) {                            // conditional branch / SVC
        const unsigned cond = (insn >> 8) & 0xF;
        if (cond == 0xF) {                                // SVC
            if (on_svc && on_svc(insn & 0xFF)) return;
            halted = true;
            return;
        }
        if (cond == 0xE) { undefined(insn); return; }     // permanently UNDEFINED
        if (cond_holds(cond))
            branch_write_pc(pc_val() + (static_cast<int32_t>(static_cast<int8_t>(insn & 0xFF)) * 2));
        return;
    }
    if ((op & 0x3E) == 0x38) {                            // B (unconditional)
        const int32_t imm = (static_cast<int32_t>(insn << 21) >> 21) * 2;   // sign-extend imm11
        branch_write_pc(pc_val() + imm);
        return;
    }
    undefined(insn);
}

// ---------------------------------------------------------------------------
// A6.3 32-bit Thumb instruction encoding
// ---------------------------------------------------------------------------

void Cpu::t32(uint32_t insn) {
    const unsigned op1 = (insn >> 27) & 3;
    const unsigned op2 = (insn >> 20) & 0x7F;

    switch (op1) {
        case 1:
            if ((op2 & 0x64) == 0x00) { t32_load_store_multiple(insn); return; }
            if ((op2 & 0x64) == 0x04) { t32_load_store_dual(insn); return; }
            if ((op2 & 0x60) == 0x20) { t32_dp_shifted_reg(insn); return; }
            a32_coproc(insn);                             // 1xxxxxx: coprocessor / VFP / NEON
            return;
        case 2:
            if ((insn >> 15) & 1) { t32_branch_misc(insn); return; }
            if ((op2 & 0x20) == 0) t32_dp_modified_imm(insn); else t32_dp_plain_imm(insn);
            return;
        default:
            if ((op2 & 0x71) == 0x00) { t32_store_single(insn); return; }
            if ((op2 & 0x67) == 0x01) { t32_load_byte(insn); return; }
            if ((op2 & 0x67) == 0x03) { t32_load_half(insn); return; }
            if ((op2 & 0x67) == 0x05) { t32_load_word(insn); return; }
            if ((op2 & 0x70) == 0x10) {                   // Advanced SIMD element/structure
                if (exec_neon_ldst(insn)) return;
                undefined(insn);
                return;
            }
            if ((op2 & 0x70) == 0x20) { t32_dp_register(insn); return; }
            if ((op2 & 0x78) == 0x30) { t32_multiply(insn); return; }
            if ((op2 & 0x78) == 0x38) { t32_long_multiply(insn); return; }
            a32_coproc(insn);
            return;
    }
}

// A6.3.1 Data-processing (modified immediate).
void Cpu::t32_dp_modified_imm(uint32_t insn) {
    const unsigned op = (insn >> 21) & 0xF;
    const bool setflags = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF;
    const uint32_t imm12 = (((insn >> 26) & 1) << 11) | (((insn >> 12) & 7) << 8) | (insn & 0xFF);
    const ShiftResult s = thumb_expand_imm_c(imm12, c);
    // Thumb-2 renumbers the data-processing opcodes and adds ORN; the table maps
    // its four-bit field onto the A32 numbering the shared executor uses.
    static const unsigned kMap[16] = {kAND, kBIC, kORR, kORN, kEOR, 0, 0, 0,
                                      kADD, 0, kADC, kSBC, 0, kSUB, kRSB, 0};
    unsigned mapped = kMap[op];
    if (op == 2 && rn == 15) mapped = kMOV;               // ORR with Rn==15 is MOV
    if (op == 3 && rn == 15) mapped = kMVN;               // ORN with Rn==15 is MVN
    if (op == 1 && rd == 15) mapped = kBIC;
    // The compare forms are spelled as "write to R15 with S set", which the shared
    // executor would take as a branch -- so they are turned back into the real
    // opcode here, where the encoding still says which one it is.
    if (rd == 15 && setflags) {
        switch (op) {
            case 0: dp_op(kTST, 0, reg(rn), s.result, true, s.carry); return;
            case 4: dp_op(kTEQ, 0, reg(rn), s.result, true, s.carry); return;
            case 8: dp_op(kCMN, 0, reg(rn), s.result, true, s.carry); return;
            case 13: dp_op(kCMP, 0, reg(rn), s.result, true, s.carry); return;
            default: break;
        }
    }
    dp_op(mapped, rd, reg(rn), s.result, setflags, s.carry);
}

// A6.3.3 Data-processing (plain binary immediate): the 12- and 16-bit immediates
// that do not go through the modified-immediate encoding, plus the bitfield and
// saturating instructions.
void Cpu::t32_dp_plain_imm(uint32_t insn) {
    const unsigned op = (insn >> 20) & 0x1F;
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF;
    const uint32_t i = (insn >> 26) & 1, imm3 = (insn >> 12) & 7, imm8 = insn & 0xFF;
    const uint32_t imm12 = (i << 11) | (imm3 << 8) | imm8;

    switch (op) {
        case 0x00:                                        // ADDW / ADR
            if (rn == 15) r[rd] = (pc_val() & ~3u) + imm12;
            else r[rd] = reg(rn) + imm12;
            return;
        case 0x0A:                                        // SUBW / ADR (negative)
            if (rn == 15) r[rd] = (pc_val() & ~3u) - imm12;
            else r[rd] = reg(rn) - imm12;
            return;
        case 0x04:                                        // MOVW
            r[rd] = (((insn >> 16) & 0xF) << 12) | imm12;
            return;
        case 0x0C:                                        // MOVT
            r[rd] = (r[rd] & 0xFFFF) | ((((insn >> 16) & 0xF) << 12 | imm12) << 16);
            return;
        case 0x10: case 0x12: {                           // SSAT / SSAT16
            const unsigned sat = (insn & 0x1F) + 1;
            const unsigned shift = (imm3 << 2) | ((insn >> 6) & 3);
            const bool asr = (insn >> 21) & 1;
            const int64_t val = static_cast<int32_t>(
                shift_c(reg(rn), asr ? kASR : kLSL, shift, c).result);
            const int64_t hi = (1ll << (sat - 1)) - 1, lo = -(1ll << (sat - 1));
            if (val > hi) { q = true; r[rd] = static_cast<uint32_t>(hi); }
            else if (val < lo) { q = true; r[rd] = static_cast<uint32_t>(lo); }
            else r[rd] = static_cast<uint32_t>(val);
            return;
        }
        case 0x18: case 0x1A: {                           // USAT / USAT16
            const unsigned sat = insn & 0x1F;
            const unsigned shift = (imm3 << 2) | ((insn >> 6) & 3);
            const bool asr = (insn >> 21) & 1;
            const int64_t val = static_cast<int32_t>(
                shift_c(reg(rn), asr ? kASR : kLSL, shift, c).result);
            const int64_t hi = (1ll << sat) - 1;
            if (val > hi) { q = true; r[rd] = static_cast<uint32_t>(hi); }
            else if (val < 0) { q = true; r[rd] = 0; }
            else r[rd] = static_cast<uint32_t>(val);
            return;
        }
        case 0x14: case 0x1C: {                           // SBFX / UBFX
            const unsigned lsb = (imm3 << 2) | ((insn >> 6) & 3);
            const unsigned width = (insn & 0x1F) + 1;
            uint32_t val = reg(rn);
            val = width == 32 ? (val >> lsb) : ((val >> lsb) & ((1u << width) - 1));
            if (op == 0x14 && width < 32 && (val & (1u << (width - 1))))
                val |= ~((1u << width) - 1);
            r[rd] = val;
            return;
        }
        case 0x16: {                                      // BFI / BFC
            const unsigned lsb = (imm3 << 2) | ((insn >> 6) & 3);
            const unsigned msb = insn & 0x1F;
            if (msb < lsb) { undefined(insn); return; }
            const uint32_t mask = msb == 31 ? (0xFFFFFFFFu << lsb)
                                            : (((1u << (msb - lsb + 1)) - 1) << lsb);
            const uint32_t src = rn == 15 ? 0 : reg(rn);
            r[rd] = (r[rd] & ~mask) | ((src << lsb) & mask);
            return;
        }
        default: undefined(insn); return;
    }
}

// A6.3.4 Branches and miscellaneous control.
void Cpu::t32_branch_misc(uint32_t insn) {
    const unsigned op = (insn >> 20) & 0x7F;
    const unsigned op1 = (insn >> 12) & 7;

    if ((op1 & 5) == 0) {                                 // the non-branch corner
        if ((op & 0x7E) == 0x38) {                        // MSR
            return;                                       // no privileged state here
        }
        if (op == 0x3B) {                                 // DSB / DMB / ISB / CLREX
            if (((insn >> 4) & 0xF) == 2) clear_exclusive();
            return;
        }
        if (op == 0x3A) return;                           // CPS and the hints
        if ((op & 0x7E) == 0x3E) {                        // MRS
            set_reg((insn >> 8) & 0xF, cpsr());
            return;
        }
    }

    if ((op1 & 5) == 0 && (op & 0x38) != 0x38) {          // B (T3, conditional)
        const unsigned cond = (insn >> 22) & 0xF;
        const int32_t imm = static_cast<int32_t>(
            (((insn >> 26) & 1) << 20) | (((insn >> 11) & 1) << 19) |
            (((insn >> 13) & 1) << 18) | (((insn >> 16) & 0x3F) << 12) |
            ((insn & 0x7FF) << 1));
        const int32_t simm = (imm << 11) >> 11;           // sign-extend at bit 20
        if (cond_holds(cond)) branch_write_pc(pc_val() + simm);
        return;
    }

    // B (T4), BL and BLX share one shape: S, two J bits that are stored inverted
    // relative to S, and two immediate fields. The inversion is what lets the
    // 24-bit range be encoded without a discontinuity, and getting it wrong
    // produces a branch that is correct for forward targets and wrong for backward
    // ones -- which looks like a miscompiled loop.
    const uint32_t s = (insn >> 26) & 1;
    const uint32_t j1 = (insn >> 13) & 1, j2 = (insn >> 11) & 1;
    const uint32_t i1 = ~(j1 ^ s) & 1, i2 = ~(j2 ^ s) & 1;
    const uint32_t imm10 = (insn >> 16) & 0x3FF, imm11 = insn & 0x7FF;
    int32_t imm = static_cast<int32_t>((s << 24) | (i1 << 23) | (i2 << 22) |
                                       (imm10 << 12) | (imm11 << 1));
    imm = (imm << 7) >> 7;                                // sign-extend at bit 24

    // Which of the three it is comes from two bits of op1, not from its value:
    // bit 2 says "link" and bit 0 says "same instruction set". Testing for the
    // exact values 5 and 7 catches half of each -- `bl` encoded with op1 = 111
    // then executes as a plain branch, which does not fault: the callee runs
    // correctly and returns through whatever was already in LR.
    if ((op1 & 5) == 5) {                                 // BL
        set_lr((cur_pc() + 4) | 1);
        branch_write_pc(pc_val() + imm);
        return;
    }
    if ((op1 & 5) == 1) {                                 // B (T4)
        branch_write_pc(pc_val() + imm);
        return;
    }
    if ((op1 & 5) == 4) {                                 // BLX (immediate), to ARM state
        const uint32_t imm10l = (insn >> 1) & 0x3FF;
        int32_t bimm = static_cast<int32_t>((s << 24) | (i1 << 23) | (i2 << 22) |
                                            (imm10 << 12) | (imm10l << 2));
        bimm = (bimm << 7) >> 7;
        // Align(PC, 4) with the PC read **in Thumb state** -- cur + 4. Clearing
        // `thumb` first makes pc_val() answer cur + 8 and the branch lands four
        // bytes into the callee, which is a real instruction and simply runs.
        const uint32_t target = ((pc_val() & ~3u) + bimm) & ~3u;
        set_lr((cur_pc() + 4) | 1);
        thumb = false;
        pc = target;
        return;
    }
    undefined(insn);
}

// A6.3.5 Load/store multiple, and the two forms that share its encoding.
void Cpu::t32_load_store_multiple(uint32_t insn) {
    const unsigned op = (insn >> 23) & 3;
    const bool load = (insn >> 20) & 1, w = (insn >> 21) & 1;
    const unsigned rn = (insn >> 16) & 0xF;
    const uint32_t list = insn & 0xFFFF;
    if (op == 1) {                                        // IA (POP.W when Rn==13)
        if (load) do_ldm(rn, list, true, false, w, false);
        else do_stm(rn, list, true, false, w);
        return;
    }
    if (op == 2) {                                        // DB (PUSH.W when Rn==13)
        if (load) do_ldm(rn, list, false, true, w, false);
        else do_stm(rn, list, false, true, w);
        return;
    }
    undefined(insn);                                      // SRS / RFE: privileged
}

// A6.3.6 Load/store dual, load/store exclusive, table branch.
void Cpu::t32_load_store_dual(uint32_t insn) {
    const unsigned op1 = (insn >> 23) & 3, op2 = (insn >> 20) & 3;
    const unsigned rn = (insn >> 16) & 0xF, rt = (insn >> 12) & 0xF;
    const unsigned rd = (insn >> 8) & 0xF;

    if (op1 == 0 && op2 == 0) {                           // STREX
        const uint32_t addr = reg(rn) + (insn & 0xFF) * 4;
        if (take_exclusive(addr)) { mem_.write<uint32_t>(addr, reg(rt)); r[rd] = 0; }
        else r[rd] = 1;
        return;
    }
    if (op1 == 0 && op2 == 1) {                           // LDREX
        const uint32_t addr = reg(rn) + (insn & 0xFF) * 4;
        set_exclusive(addr);
        r[rt] = mem_.read<uint32_t>(addr);
        return;
    }
    // op1 == 01 is the whole of the byte/half/dual exclusive family *and* the table
    // branches, split by op2: 00 stores, 01 loads. The word-sized STREX/LDREX above
    // are op1 == 00, which is a different pair of bits -- reading op1 as one bit, or
    // looking for these under op1 == 00, leaves every sub-word atomic decoding as an
    // LDRD. That does not fault: it loads a register pair from the lock's address,
    // and the compare-and-swap loop that owns the lock spins forever on a value it
    // can never store.
    if (op1 == 1 && op2 == 0) {                           // STREXB / STREXH / STREXD
        const unsigned kind = (insn >> 4) & 0xF;
        const uint32_t addr = reg(rn);
        const unsigned rdd = insn & 0xF;                  // Rd is the low nibble here
        if (kind == 4 || kind == 5 || kind == 7) {
            if (take_exclusive(addr)) {
                if (kind == 4) mem_.write<uint8_t>(addr, static_cast<uint8_t>(reg(rt)));
                else if (kind == 5) mem_.write<uint16_t>(addr, static_cast<uint16_t>(reg(rt)));
                else {
                    mem_.write<uint32_t>(addr, reg(rt));
                    mem_.write<uint32_t>(addr + 4, reg(rd));
                }
                r[rdd] = 0;
            } else r[rdd] = 1;
            return;
        }
        undefined(insn);
        return;
    }
    if (op1 == 1 && op2 == 1) {                           // TBB/TBH, LDREXB/H/D
        const unsigned kind = (insn >> 4) & 0xF;
        const uint32_t addr = reg(rn);
        switch (kind) {
            case 0: case 1: {                             // TBB / TBH
                const unsigned rm = insn & 0xF;
                const bool halfword = kind == 1;
                const uint32_t base = rn == 15 ? pc_val() : reg(rn);
                const uint32_t idx = reg(rm);
                const uint32_t off = halfword ? mem_.read<uint16_t>(base + idx * 2)
                                              : mem_.read<uint8_t>(base + idx);
                branch_write_pc(pc_val() + off * 2);
                return;
            }
            case 4: set_exclusive(addr); r[rt] = mem_.read<uint8_t>(addr); return;   // LDREXB
            case 5: set_exclusive(addr); r[rt] = mem_.read<uint16_t>(addr); return;  // LDREXH
            case 7:                                                                  // LDREXD
                set_exclusive(addr);
                r[rt] = mem_.read<uint32_t>(addr);
                r[rd] = mem_.read<uint32_t>(addr + 4);
                return;
            default: undefined(insn); return;
        }
    }

    // LDRD / STRD (immediate). P/U/W in bits 24/23/21, an 8-bit word offset.
    const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1, w = (insn >> 21) & 1;
    const bool load = (insn >> 20) & 1;
    const unsigned rt2 = (insn >> 8) & 0xF;
    const uint32_t imm = (insn & 0xFF) * 4;
    const uint32_t base = rn == 15 ? (pc_val() & ~3u) : reg(rn);
    const uint32_t offset_addr = u ? base + imm : base - imm;
    const uint32_t addr = p ? offset_addr : base;
    if (load) {
        r[rt] = mem_.read<uint32_t>(addr);
        r[rt2] = mem_.read<uint32_t>(addr + 4);
    } else {
        mem_.write<uint32_t>(addr, reg(rt));
        mem_.write<uint32_t>(addr + 4, reg(rt2));
    }
    if (w && rn != 15) r[rn] = offset_addr;
}

// A6.3.11 Data-processing (shifted register).
void Cpu::t32_dp_shifted_reg(uint32_t insn) {
    const unsigned op = (insn >> 21) & 0xF;
    const bool setflags = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF, rm = insn & 0xF;
    const unsigned imm5 = (((insn >> 12) & 7) << 2) | ((insn >> 6) & 3);
    unsigned stype, amt;
    decode_imm_shift((insn >> 4) & 3, imm5, &stype, &amt);
    const ShiftResult s = stype == kRRX ? rrx_c(reg(rm), c) : shift_c(reg(rm), stype, amt, c);

    static const unsigned kMap[16] = {kAND, kBIC, kORR, kORN, kEOR, 0, 0, 0,
                                      kADD, 0, kADC, kSBC, 0, kSUB, kRSB, 0};
    if (rd == 15 && setflags) {
        switch (op) {
            case 0: dp_op(kTST, 0, reg(rn), s.result, true, s.carry); return;
            case 4: dp_op(kTEQ, 0, reg(rn), s.result, true, s.carry); return;
            case 8: dp_op(kCMN, 0, reg(rn), s.result, true, s.carry); return;
            case 13: dp_op(kCMP, 0, reg(rn), s.result, true, s.carry); return;
            default: break;
        }
    }
    unsigned mapped = kMap[op];
    // MOV and the shifts are ORR with Rn == 15; MVN is ORN with Rn == 15. That is
    // how Thumb-2 spells `lsls r0, r1, #3` in 32 bits.
    if (op == 2 && rn == 15) mapped = kMOV;
    if (op == 3 && rn == 15) mapped = kMVN;
    dp_op(mapped, rd, reg(rn), s.result, setflags, s.carry);
}

// A6.3.12 Data-processing (register): the register-controlled shifts, the extends,
// and the miscellaneous group (CLZ, REV, SEL and the saturating adds).
void Cpu::t32_dp_register(uint32_t insn) {
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 8) & 0xF, rm = insn & 0xF;
    const unsigned op1 = (insn >> 20) & 0xF, op2 = (insn >> 4) & 0xF;

    if (op2 == 0 && (op1 & 8) == 0) {                     // LSL/LSR/ASR/ROR (register)
        const unsigned type = (op1 >> 1) & 3;
        const ShiftResult s = shift_c(reg(rn), type, reg(rm) & 0xFF, c);
        r[rd] = s.result;
        if (op1 & 1) { set_nz(s.result); c = s.carry; }
        return;
    }
    if ((op2 & 8) && (op1 & 8) == 0) {
        // The extends. `op1` here is bits 23..20 and selects the operation
        // *directly* -- 0 SXTH, 1 UXTH, 2 SXTB16, 3 UXTB16, 4 SXTB, 5 UXTB --
        // unlike the shift group just above, where the low bit is the S flag. An
        // earlier draft shifted op1 right by one for both, which turned `uxtb16`
        // into `uxth`: a mask of 0x0000FFFF where the guest asked for 0x00FF00FF,
        // and a number that is wrong only in the bytes that were supposed to
        // survive.
        const unsigned rot = ((insn >> 4) & 3) * 8;
        const uint32_t x = rot ? ((reg(rm) >> rot) | (reg(rm) << (32 - rot))) : reg(rm);
        const bool halves = op1 == 2 || op1 == 3;         // the ...16 forms
        uint32_t val;
        switch (op1) {
            case 0: val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(x))); break;
            case 1: val = x & 0xFFFF; break;
            case 2:                                       // SXTB16 / SXTAB16
                val = ((static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(x >> 16))) & 0xFFFF) << 16) |
                      (static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(x))) & 0xFFFF);
                break;
            case 3: val = (((x >> 16) & 0xFF) << 16) | (x & 0xFF); break;    // UXTB16 / UXTAB16
            case 4: val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(x))); break;
            case 5: val = x & 0xFF; break;
            default: undefined(insn); return;
        }
        if (rn != 15) {                                   // the accumulate forms
            if (halves)
                val = ((((val >> 16) + (reg(rn) >> 16)) & 0xFFFF) << 16) |
                      ((val + reg(rn)) & 0xFFFF);
            else val += reg(rn);
        }
        r[rd] = val;
        return;
    }
    if ((op1 & 0xC) == 0x8 && (op2 & 0xC) == 0x8) {       // the misc group
        const uint32_t x = reg(rm);
        switch (((op1 & 3) << 2) | (op2 & 3)) {
            case 0x4: r[rd] = (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24); return;  // REV
            case 0x5: r[rd] = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu); return;                  // REV16
            case 0x6: {                                                                                     // RBIT
                uint32_t res = 0;
                for (unsigned i = 0; i < 32; ++i) if (x & (1u << i)) res |= 1u << (31 - i);
                r[rd] = res;
                return;
            }
            case 0x7:                                                                                       // REVSH
                r[rd] = static_cast<uint32_t>(static_cast<int32_t>(
                    static_cast<int16_t>(((x & 0xFF) << 8) | ((x >> 8) & 0xFF))));
                return;
            case 0x8: {                                   // SEL
                const uint32_t a = reg(rn), b = x;
                uint32_t res = 0;
                for (unsigned i = 0; i < 4; ++i)
                    res |= ((ge >> i) & 1 ? a : b) & (0xFFu << (i * 8));
                r[rd] = res;
                return;
            }
            case 0xC: {                                   // CLZ
                unsigned k = 0;
                while (k < 32 && !(x & (0x80000000u >> k))) ++k;
                r[rd] = k;
                return;
            }
            default: break;
        }
    }
    undefined(insn);
}

// A6.3.16 Multiply, multiply accumulate.
void Cpu::t32_multiply(uint32_t insn) {
    const unsigned op1 = (insn >> 20) & 7, op2 = (insn >> 4) & 3;
    const unsigned rn = (insn >> 16) & 0xF, ra = (insn >> 12) & 0xF;
    const unsigned rd = (insn >> 8) & 0xF, rm = insn & 0xF;
    const uint32_t a = reg(rn), b = reg(rm);
    switch (op1) {
        case 0:
            if (op2 == 0) r[rd] = ra == 15 ? a * b : a * b + reg(ra);      // MUL / MLA
            else r[rd] = reg(ra) - a * b;                                   // MLS
            return;
        case 1: {                                                           // SMLABB family
            const bool nn = (insn >> 5) & 1, m = (insn >> 4) & 1;
            const int32_t x = nn ? static_cast<int16_t>(a >> 16) : static_cast<int16_t>(a);
            const int32_t y = m ? static_cast<int16_t>(b >> 16) : static_cast<int16_t>(b);
            r[rd] = static_cast<uint32_t>(static_cast<int64_t>(x) * y +
                                          (ra == 15 ? 0 : static_cast<int32_t>(reg(ra))));
            return;
        }
        case 5: {                                                           // SMMLA / SMMUL
            int64_t res = static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b);
            if (ra != 15) res += static_cast<int64_t>(static_cast<int32_t>(reg(ra))) << 32;
            if ((insn >> 4) & 1) res += 0x80000000ll;
            r[rd] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
            return;
        }
        default: undefined(insn); return;
    }
}

// A6.3.17 Long multiply, long multiply accumulate, and divide.
void Cpu::t32_long_multiply(uint32_t insn) {
    const unsigned op1 = (insn >> 20) & 7, op2 = (insn >> 4) & 0xF;
    const unsigned rn = (insn >> 16) & 0xF, rdlo = (insn >> 12) & 0xF;
    const unsigned rdhi = (insn >> 8) & 0xF, rm = insn & 0xF;
    const uint32_t a = reg(rn), b = reg(rm);
    switch (op1) {
        case 0: {                                          // SMULL
            const int64_t res = static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b);
            r[rdlo] = static_cast<uint32_t>(res);
            r[rdhi] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
            return;
        }
        case 1:                                            // SDIV
            if (b == 0) r[rdhi] = 0;
            else if (a == 0x80000000u && b == 0xFFFFFFFFu) r[rdhi] = 0x80000000u;
            else r[rdhi] = static_cast<uint32_t>(static_cast<int32_t>(a) / static_cast<int32_t>(b));
            return;
        case 2: {                                          // UMULL
            const uint64_t res = static_cast<uint64_t>(a) * b;
            r[rdlo] = static_cast<uint32_t>(res);
            r[rdhi] = static_cast<uint32_t>(res >> 32);
            return;
        }
        case 3: r[rdhi] = b == 0 ? 0 : a / b; return;      // UDIV
        case 4: {                                          // SMLAL and its halfword forms
            const int64_t acc = static_cast<int64_t>((static_cast<uint64_t>(reg(rdhi)) << 32) | reg(rdlo));
            int64_t res;
            if (op2 == 0) {
                res = acc + static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b);
            } else {
                const bool nn = (op2 >> 1) & 1, m = op2 & 1;
                const int32_t x = nn ? static_cast<int16_t>(a >> 16) : static_cast<int16_t>(a);
                const int32_t y = m ? static_cast<int16_t>(b >> 16) : static_cast<int16_t>(b);
                res = acc + static_cast<int64_t>(x) * y;
            }
            r[rdlo] = static_cast<uint32_t>(res);
            r[rdhi] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
            return;
        }
        case 6: {                                          // UMLAL / UMAAL
            if (op2 == 6) {                                // UMAAL
                const uint64_t res = static_cast<uint64_t>(a) * b + reg(rdlo) + reg(rdhi);
                r[rdlo] = static_cast<uint32_t>(res);
                r[rdhi] = static_cast<uint32_t>(res >> 32);
                return;
            }
            const uint64_t acc = (static_cast<uint64_t>(reg(rdhi)) << 32) | reg(rdlo);
            const uint64_t res = acc + static_cast<uint64_t>(a) * b;
            r[rdlo] = static_cast<uint32_t>(res);
            r[rdhi] = static_cast<uint32_t>(res >> 32);
            return;
        }
        default: undefined(insn); return;
    }
}

// A6.3.7-A6.3.10 Load and store, single data item.
//
// One shape covers all of them. Bits 24..20 say size, sign and direction; the
// second halfword says which of the four addressing forms this is. Splitting them
// per width produced four copies of the same addressing code in an earlier draft,
// and the copy that was wrong was the one used least.
namespace {
struct T32Addr {
    uint32_t addr, wb_addr;
    unsigned rn;
    bool writeback;
};
}  // namespace

void Cpu::t32_load_word(uint32_t insn)  { t32_store_single(insn); }
void Cpu::t32_load_byte(uint32_t insn)  { t32_store_single(insn); }
void Cpu::t32_load_half(uint32_t insn)  { t32_store_single(insn); }

void Cpu::t32_store_single(uint32_t insn) {
    const bool sign = (insn >> 24) & 1;                   // S: sign-extend the load
    const bool imm12_form = (insn >> 23) & 1;
    const unsigned size2 = (insn >> 21) & 3;              // 00 byte, 01 half, 10 word
    const bool load = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF, rt = (insn >> 12) & 0xF;
    const unsigned size = size2 == 0 ? 1 : (size2 == 1 ? 2 : 4);
    if (size2 == 3) { undefined(insn); return; }

    // PLD and PLI arrive here with Rt == 15 and a load bit set. They are hints and
    // have no architectural effect -- but they must not be executed as loads into
    // the PC, which is what happens if they are not recognised.
    if (load && rt == 15 && size2 != 2) return;

    if (rn == 15) {                                       // literal
        const uint32_t base = pc_val() & ~3u;
        const uint32_t imm = insn & 0xFFF;
        const uint32_t addr = ((insn >> 23) & 1) ? base + imm : base - imm;
        if (load) do_load(rt, addr, size, sign);
        else do_store(rt, addr, size);
        return;
    }

    uint32_t addr, offset_addr = 0;
    bool writeback = false;
    if (imm12_form) {
        addr = reg(rn) + (insn & 0xFFF);
    } else if ((insn >> 11) & 1) {                        // the 8-bit P/U/W form
        const bool p = (insn >> 10) & 1, u = (insn >> 9) & 1, w = (insn >> 8) & 1;
        const uint32_t imm = insn & 0xFF;
        const uint32_t base = reg(rn);
        offset_addr = u ? base + imm : base - imm;
        addr = p ? offset_addr : base;
        writeback = w;
    } else {                                              // register offset, LSL #imm2
        const unsigned rm = insn & 0xF;
        addr = reg(rn) + (reg(rm) << ((insn >> 4) & 3));
    }
    if (load) do_load(rt, addr, size, sign);
    else do_store(rt, addr, size);
    if (writeback && !(load && rt == rn)) r[rn] = offset_addr;
}

}  // namespace arm
