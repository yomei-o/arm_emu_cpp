// The A32 (ARM state) interpreter.
//
// Laid out in the order the ARM ARM's A5.1 encoding tables are: bits 27..25 pick
// the group, each group sub-decodes its own fields, and the function names are the
// section titles. An instruction is findable by the name the manual gives it.
//
// Two rules hold everywhere in this file and are the reason the R15 hazards do not
// spread:
//
//   - reading a register is `reg(i)`, never `r[i]`, because `reg(15)` is the only
//     place the +8 pipeline offset exists;
//   - writing a register is `set_reg(i, v)`, never `r[i] = v`, because writing R15
//     is an interworking branch and getting that wrong turns a `pop {pc}` into a
//     jump to an odd address in the wrong instruction set.
#include "cpu.h"
#include <cstdio>
#include <cstring>

namespace arm {

void Cpu::fail(const std::string& why, uint32_t insn) const {
    throw CpuError{why, cur_pc_, insn};
}

void Cpu::undefined(uint32_t insn) {
    // An instruction we do not implement is offered to the guest as SIGILL first:
    // a CPU without the instruction is exactly what the guest would be running on,
    // and probing code (OpenSSL's, glibc's ifunc resolvers) depends on getting the
    // signal rather than on the machine stopping. If nothing catches it, stopping
    // and printing the encoding is the only honest option -- a silent no-op leaves
    // a stale register and fails somewhere unrelated much later.
    if (on_undefined && on_undefined(insn, cur_pc_)) return;
    char buf[128];
    std::snprintf(buf, sizeof buf, "unimplemented %s instruction %08X",
                  thumb ? "Thumb" : "ARM", insn);
    fail(buf, insn);
}

bool Cpu::cond_holds(unsigned cond) const {
    switch (cond >> 1) {
        case 0: return (cond & 1) ? !z : z;                       // EQ / NE
        case 1: return (cond & 1) ? !c : c;                       // CS / CC
        case 2: return (cond & 1) ? !n : n;                       // MI / PL
        case 3: return (cond & 1) ? !v : v;                       // VS / VC
        case 4: return (cond & 1) ? (!c || z) : (c && !z);        // HI / LS
        case 5: return (cond & 1) ? (n != v) : (n == v);          // GE / LT
        case 6: return (cond & 1) ? (z || n != v) : (!z && n == v);  // GT / LE
        default: return true;                                     // AL, and 0b1111
    }
}

void Cpu::step() {
    if (max_insns && insns >= max_insns) fail("instruction limit reached", 0);
    ++insns;
    if (sample_every) {
        if (sample_left == 0) sample_left = sample_every;
        if (--sample_left == 0) {
            sample_left = sample_every;
            std::fprintf(stderr, "[pc] %08X %s\n", pc, thumb ? "T" : "A");
        }
    }
    if (pc_watch && pc == pc_watch && on_pc_watch) on_pc_watch();

    cur_pc_ = pc;
    if (thumb) { exec_thumb(); return; }
    const uint32_t insn = mem_.read<uint32_t>(pc);
    pc += 4;
    exec_a32(insn);
}

// ---------------------------------------------------------------------------
// Shared execution helpers. Both decoders call these; neither duplicates them.
// ---------------------------------------------------------------------------

void Cpu::dp_op(unsigned op, unsigned rd, uint32_t a, uint32_t b,
                bool setflags, bool shifter_carry) {
    uint32_t res = 0;
    bool write = true, cout = shifter_carry, vout = v;
    switch (op) {
        case kAND: res = a & b; break;
        case kEOR: res = a ^ b; break;
        // Every subtraction is AddWithCarry(x, NOT y, 1). Spelling it `a - b`
        // produces the right number and the wrong carry: on ARM, C after a
        // subtract means "no borrow".
        case kSUB: { const AddResult t = add_with_carry(a, ~b, true);  res = t.result; cout = t.carry; vout = t.overflow; break; }
        case kRSB: { const AddResult t = add_with_carry(~a, b, true);  res = t.result; cout = t.carry; vout = t.overflow; break; }
        case kADD: { const AddResult t = add_with_carry(a, b, false);  res = t.result; cout = t.carry; vout = t.overflow; break; }
        case kADC: { const AddResult t = add_with_carry(a, b, c);      res = t.result; cout = t.carry; vout = t.overflow; break; }
        case kSBC: { const AddResult t = add_with_carry(a, ~b, c);     res = t.result; cout = t.carry; vout = t.overflow; break; }
        case kRSC: { const AddResult t = add_with_carry(~a, b, c);     res = t.result; cout = t.carry; vout = t.overflow; break; }
        case kTST: res = a & b; write = false; break;
        case kTEQ: res = a ^ b; write = false; break;
        case kCMP: { const AddResult t = add_with_carry(a, ~b, true);  res = t.result; cout = t.carry; vout = t.overflow; write = false; break; }
        case kCMN: { const AddResult t = add_with_carry(a, b, false);  res = t.result; cout = t.carry; vout = t.overflow; write = false; break; }
        case kORR: res = a | b; break;
        case kMOV: res = b; break;
        case kBIC: res = a & ~b; break;
        case kMVN: res = ~b; break;
        default:   res = a | ~b; break;                    // ORN, Thumb-2 only
    }
    // `MOVS pc, lr` and friends are the exception-return form: in a user-mode
    // process they appear only in a signal restorer, where copying a saved SPSR is
    // not something this emulator models. Branch, and leave the flags alone --
    // updating them from the return address would corrupt the resumed context.
    if (setflags && rd != 15) { set_nz(res); c = cout; v = vout; }
    // ALUWritePC, not LoadWritePC: see the note on set_reg_alu(). In ARM state the
    // two are the same; in Thumb they are not.
    if (write) set_reg_alu(rd, res);
}

void Cpu::do_load(unsigned rt, uint32_t addr, unsigned size, bool sign, bool) {
    uint32_t val;
    switch (size) {
        case 1: {
            const uint8_t b = mem_.read<uint8_t>(addr);
            val = sign ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(b))) : b;
            break;
        }
        case 2: {
            const uint16_t h = mem_.read<uint16_t>(addr);
            val = sign ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(h))) : h;
            break;
        }
        default: val = mem_.read<uint32_t>(addr); break;
    }
    set_reg(rt, val);          // LoadWritePC is an interworking branch on ARMv7
}

void Cpu::do_store(unsigned rt, uint32_t addr, unsigned size) {
    const uint32_t val = reg(rt);              // storing R15 stores cur_pc + 8
    switch (size) {
        case 1: mem_.write<uint8_t>(addr, static_cast<uint8_t>(val)); break;
        case 2: mem_.write<uint16_t>(addr, static_cast<uint16_t>(val)); break;
        default: mem_.write<uint32_t>(addr, val); break;
    }
}

namespace {
inline unsigned popcount16(uint32_t v) {
    unsigned n = 0;
    for (unsigned i = 0; i < 16; ++i) if (v & (1u << i)) ++n;
    return n;
}
}  // namespace

void Cpu::do_stm(unsigned rn, uint32_t list, bool inc, bool before, bool wback) {
    const unsigned count = popcount16(list);
    const uint32_t base = reg(rn);
    // The four addressing modes are two bits, and all four reduce to "where does
    // the lowest-numbered register land". Decrementing modes write the same block
    // as their incrementing counterparts, just below the base.
    uint32_t addr = inc ? (before ? base + 4 : base)
                        : (before ? base - count * 4 : base - count * 4 + 4);
    for (unsigned i = 0; i < 16; ++i) {
        if (!(list & (1u << i))) continue;
        mem_.write<uint32_t>(addr, reg(i));
        addr += 4;
    }
    if (wback) r[rn] = inc ? base + count * 4 : base - count * 4;
}

void Cpu::do_ldm(unsigned rn, uint32_t list, bool inc, bool before, bool wback, bool) {
    const unsigned count = popcount16(list);
    const uint32_t base = reg(rn);
    uint32_t addr = inc ? (before ? base + 4 : base)
                        : (before ? base - count * 4 : base - count * 4 + 4);
    uint32_t pc_val_loaded = 0;
    bool loads_pc = false;
    for (unsigned i = 0; i < 16; ++i) {
        if (!(list & (1u << i))) continue;
        const uint32_t val = mem_.read<uint32_t>(addr);
        addr += 4;
        if (i == 15) { pc_val_loaded = val; loads_pc = true; }
        else r[i] = val;
    }
    // Writeback happens before the PC is written, and not at all when the base
    // register was itself in the list -- the loaded value wins, which is what the
    // architecture says and what a `pop {r4, sp, pc}` epilogue depends on.
    if (wback && !(list & (1u << rn))) r[rn] = inc ? base + count * 4 : base - count * 4;
    if (loads_pc) bx_write_pc(pc_val_loaded);
}

// ---------------------------------------------------------------------------
// A5.1 ARM instruction set encoding
// ---------------------------------------------------------------------------

void Cpu::exec_a32(uint32_t insn) {
    const unsigned cond = insn >> 28;
    if (cond == 0xF) { a32_unconditional(insn); return; }
    if (!cond_holds(cond)) return;

    switch ((insn >> 25) & 7) {
        case 0: a32_dp_reg(insn); break;
        case 1: a32_dp_imm(insn); break;
        case 2: a32_load_store(insn); break;
        case 3: if (insn & 0x10) a32_media(insn); else a32_load_store(insn); break;
        case 4: a32_block(insn); break;
        case 5: a32_branch(insn); break;
        default: a32_coproc(insn); break;                  // 110 and 111
    }
}

// A5.2 Data-processing and miscellaneous instructions, register forms.
void Cpu::a32_dp_reg(uint32_t insn) {
    const unsigned op1 = (insn >> 20) & 0x1F;
    const unsigned op2 = (insn >> 4) & 0xF;

    // The four "not really data-processing" holes punched through this group. They
    // are recognised first because their opcode fields overlap the DP ones exactly
    // -- `mul r0, r1, r2` and `and r0, r1, r2, lsl r0` differ only in bits 7..4.
    if (op2 == 0x9) {
        if ((op1 & 0x10) == 0) { a32_multiply(insn); return; }
        a32_sync(insn); return;
    }
    if ((op2 & 0x9) == 0x9) { a32_extra_load_store(insn); return; }   // 1011, 1101, 1111
    if ((op1 & 0x19) == 0x10) {                                       // 10xx0: misc
        if ((op2 & 0x8) == 0) { a32_misc(insn); return; }
        // 1xx0 with bit 3 set: the halfword multiplies, SMLABB and its family.
        const unsigned rd = (insn >> 16) & 0xF, ra = (insn >> 12) & 0xF;
        const unsigned rm = (insn >> 8) & 0xF, rn = insn & 0xF;
        const bool m = (insn >> 6) & 1, nn = (insn >> 5) & 1;
        const unsigned kind = (insn >> 21) & 3;
        const int32_t a16 = nn ? static_cast<int16_t>(reg(rn) >> 16) : static_cast<int16_t>(reg(rn));
        const int32_t b16 = m ? static_cast<int16_t>(reg(rm) >> 16) : static_cast<int16_t>(reg(rm));
        if (kind == 0) {                                   // SMLABB/BT/TB/TT
            const int64_t prod = static_cast<int64_t>(a16) * b16;
            const AddResult t = add_with_carry(static_cast<uint32_t>(prod), reg(ra), false);
            if (t.overflow) q = true;
            set_reg(rd, t.result);
        } else if (kind == 1) {                            // SMLAWB/WT or SMULWB/WT
            const int64_t prod = (static_cast<int64_t>(static_cast<int32_t>(reg(rn))) * b16) >> 16;
            if (nn) set_reg(rd, static_cast<uint32_t>(prod));                       // SMULW
            else set_reg(rd, static_cast<uint32_t>(prod) + reg(ra));                // SMLAW
        } else if (kind == 2) {                            // SMLALBB and family
            const unsigned dhi = (insn >> 16) & 0xF, dlo = (insn >> 12) & 0xF;
            const int64_t acc = (static_cast<int64_t>(reg(dhi)) << 32) | reg(dlo);
            const int64_t res = acc + static_cast<int64_t>(a16) * b16;
            r[dlo] = static_cast<uint32_t>(res);
            r[dhi] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
        } else {                                           // SMULBB/BT/TB/TT
            set_reg(rd, static_cast<uint32_t>(static_cast<int64_t>(a16) * b16));
        }
        return;
    }

    // Data-processing (register) and (register-shifted register).
    const unsigned op = (insn >> 21) & 0xF;
    const bool setflags = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 12) & 0xF, rm = insn & 0xF;
    uint32_t shifted;
    bool sc = c;
    if ((op2 & 1) == 0) {                                  // immediate shift
        unsigned stype, amt;
        decode_imm_shift((insn >> 5) & 3, (insn >> 7) & 0x1F, &stype, &amt);
        const ShiftResult s = stype == kRRX ? rrx_c(reg(rm), c)
                                            : shift_c(reg(rm), stype, amt, c);
        shifted = s.result; sc = s.carry;
    } else {                                               // register shift
        const unsigned rs = (insn >> 8) & 0xF;
        const unsigned amt = reg(rs) & 0xFF;
        const ShiftResult s = shift_c(reg(rm), (insn >> 5) & 3, amt, c);
        shifted = s.result; sc = s.carry;
    }
    dp_op(op, rd, reg(rn), shifted, setflags, sc);
}

// A5.2 with bit 25 set: immediate forms, plus the three encodings that stole the
// 10xx0 opcode holes (MOVW, MOVT, MSR immediate and the hint instructions).
void Cpu::a32_dp_imm(uint32_t insn) {
    const unsigned op1 = (insn >> 20) & 0x1F;
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 12) & 0xF;
    const uint32_t imm12 = insn & 0xFFF;

    if ((op1 & 0x1F) == 0x10) {                            // MOVW (16-bit immediate)
        set_reg(rd, (((insn >> 16) & 0xF) << 12) | imm12);
        return;
    }
    if ((op1 & 0x1F) == 0x14) {                            // MOVT: top half only
        set_reg(rd, (r[rd] & 0xFFFF) | ((((insn >> 16) & 0xF) << 12 | imm12) << 16));
        return;
    }
    if ((op1 & 0x1B) == 0x12) {                            // MSR (immediate) and hints
        if (rn == 0 && ((insn >> 22) & 1) == 0) {
            // Hints: NOP, YIELD, WFE, WFI, SEV, DBG. All of them are legitimately
            // nothing here -- there is one CPU and nothing to wait for.
            return;
        }
        const ShiftResult s = expand_imm_c(imm12, c);
        if ((insn >> 19) & 1) {                            // the flags byte
            n = (s.result >> 31) & 1; z = (s.result >> 30) & 1;
            c = (s.result >> 29) & 1; v = (s.result >> 28) & 1;
            q = (s.result >> 27) & 1;
        }
        return;
    }

    const unsigned op = (insn >> 21) & 0xF;
    const bool setflags = (insn >> 20) & 1;
    const ShiftResult s = expand_imm_c(imm12, c);
    dp_op(op, rd, reg(rn), s.result, setflags, s.carry);
}

// A5.2.12 Miscellaneous instructions.
void Cpu::a32_misc(uint32_t insn) {
    const unsigned op = (insn >> 21) & 3;
    const unsigned op2 = (insn >> 4) & 7;
    const unsigned rd = (insn >> 12) & 0xF, rm = insn & 0xF;
    switch (op2) {
        case 0:                                            // MRS / MSR (register)
            if (op & 1) {                                  // MSR
                const uint32_t val = reg(rm);
                if ((insn >> 19) & 1) {
                    n = (val >> 31) & 1; z = (val >> 30) & 1;
                    c = (val >> 29) & 1; v = (val >> 28) & 1; q = (val >> 27) & 1;
                }
                // The control byte holds the mode and interrupt bits; a user-mode
                // process cannot change them and the architecture ignores the write.
            } else {
                set_reg(rd, cpsr());
            }
            return;
        case 1:
            if (op == 1) { bx_write_pc(reg(rm)); return; }              // BX
            if (op == 3) {                                              // CLZ
                const uint32_t val = reg(rm);
                unsigned k = 0;
                while (k < 32 && !(val & (0x80000000u >> k))) ++k;
                set_reg(rd, k);
                return;
            }
            break;
        case 2: if (op == 1) { bx_write_pc(reg(rm)); return; } break;   // BXJ: no Jazelle
        case 3:
            if (op == 1) {                                              // BLX (register)
                const uint32_t target = reg(rm);
                set_lr(cur_pc_ + 4);
                bx_write_pc(target);
                return;
            }
            break;
        case 5: {                                          // QADD / QSUB / QDADD / QDSUB
            const unsigned rn = (insn >> 16) & 0xF;
            const int64_t a = static_cast<int32_t>(reg(rm));
            int64_t b = static_cast<int32_t>(reg(rn));
            auto sat32 = [this](int64_t x) -> uint32_t {
                if (x > 0x7FFFFFFFll) { q = true; return 0x7FFFFFFFu; }
                if (x < -0x80000000ll) { q = true; return 0x80000000u; }
                return static_cast<uint32_t>(static_cast<int32_t>(x));
            };
            if (op >= 2) b = static_cast<int32_t>(sat32(b * 2));        // QDADD / QDSUB
            set_reg(rd, sat32((op & 1) ? a - b : a + b));
            return;
        }
        case 7:                                            // BKPT: a debugger trap
            if (op == 1) fail("BKPT", insn);
            break;
        default: break;
    }
    undefined(insn);
}

// A5.2.5 Multiply and multiply-accumulate.
void Cpu::a32_multiply(uint32_t insn) {
    const unsigned op = (insn >> 21) & 7;
    const bool setflags = (insn >> 20) & 1;
    const unsigned rd = (insn >> 16) & 0xF, ra = (insn >> 12) & 0xF;
    const unsigned rm = (insn >> 8) & 0xF, rn = insn & 0xF;
    const uint32_t a = reg(rn), b = reg(rm);
    switch (op) {
        case 0: {                                          // MUL
            const uint32_t res = a * b;
            r[rd] = res;
            if (setflags) set_nz(res);
            return;
        }
        case 1: {                                          // MLA
            const uint32_t res = a * b + reg(ra);
            r[rd] = res;
            if (setflags) set_nz(res);
            return;
        }
        case 2: {                                          // UMAAL
            const uint64_t res = static_cast<uint64_t>(a) * b + reg(ra) + reg(rd);
            r[ra] = static_cast<uint32_t>(res);
            r[rd] = static_cast<uint32_t>(res >> 32);
            return;
        }
        case 3: r[rd] = reg(ra) - a * b; return;           // MLS
        case 4: {                                          // UMULL
            const uint64_t res = static_cast<uint64_t>(a) * b;
            r[ra] = static_cast<uint32_t>(res);
            r[rd] = static_cast<uint32_t>(res >> 32);
            if (setflags) { n = (res >> 63) & 1; z = res == 0; }
            return;
        }
        case 5: {                                          // UMLAL
            const uint64_t acc = (static_cast<uint64_t>(reg(rd)) << 32) | reg(ra);
            const uint64_t res = acc + static_cast<uint64_t>(a) * b;
            r[ra] = static_cast<uint32_t>(res);
            r[rd] = static_cast<uint32_t>(res >> 32);
            if (setflags) { n = (res >> 63) & 1; z = res == 0; }
            return;
        }
        case 6: {                                          // SMULL
            const int64_t res = static_cast<int64_t>(static_cast<int32_t>(a)) *
                                static_cast<int32_t>(b);
            r[ra] = static_cast<uint32_t>(res);
            r[rd] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
            if (setflags) { n = res < 0; z = res == 0; }
            return;
        }
        default: {                                         // SMLAL
            const int64_t acc = static_cast<int64_t>((static_cast<uint64_t>(reg(rd)) << 32) | reg(ra));
            const int64_t res = acc + static_cast<int64_t>(static_cast<int32_t>(a)) *
                                          static_cast<int32_t>(b);
            r[ra] = static_cast<uint32_t>(res);
            r[rd] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
            if (setflags) { n = res < 0; z = res == 0; }
            return;
        }
    }
}

// A5.2.10 Synchronization primitives: the exclusive family, and SWP.
//
// The monitor is real rather than "STREX always succeeds", and it has to be: the
// moment a second thread exists, an always-succeeding store-exclusive lets two
// threads both win the same compare-and-swap, and a counter incremented 100,000
// times comes out at a number that looks plausible and is simply wrong.
void Cpu::a32_sync(uint32_t insn) {
    const unsigned op = (insn >> 20) & 0xF;
    const unsigned rn = (insn >> 16) & 0xF, rd = (insn >> 12) & 0xF, rt = insn & 0xF;
    const uint32_t addr = reg(rn);
    switch (op) {
        case 0x0: case 0x4: {                              // SWP / SWPB
            const bool byte = op == 0x4;
            const uint32_t old = byte ? mem_.read<uint8_t>(addr) : mem_.read<uint32_t>(addr);
            if (byte) mem_.write<uint8_t>(addr, static_cast<uint8_t>(reg(rt)));
            else mem_.write<uint32_t>(addr, reg(rt));
            set_reg(rd, old);
            return;
        }
        case 0x8:                                          // STREX
            if (take_exclusive(addr)) { mem_.write<uint32_t>(addr, reg(rt)); set_reg(rd, 0); }
            else set_reg(rd, 1);
            return;
        case 0x9: set_exclusive(addr); set_reg(rd, mem_.read<uint32_t>(addr)); return;  // LDREX
        case 0xA:                                          // STREXD: a register pair
            if (take_exclusive(addr)) {
                mem_.write<uint32_t>(addr, reg(rt));
                mem_.write<uint32_t>(addr + 4, reg(rt + 1));
                set_reg(rd, 0);
            } else set_reg(rd, 1);
            return;
        case 0xB:                                          // LDREXD
            set_exclusive(addr);
            r[rd] = mem_.read<uint32_t>(addr);
            r[rd + 1] = mem_.read<uint32_t>(addr + 4);
            return;
        case 0xC:                                          // STREXB
            if (take_exclusive(addr)) {
                mem_.write<uint8_t>(addr, static_cast<uint8_t>(reg(rt)));
                set_reg(rd, 0);
            } else set_reg(rd, 1);
            return;
        case 0xD: set_exclusive(addr); set_reg(rd, mem_.read<uint8_t>(addr)); return;   // LDREXB
        case 0xE:                                          // STREXH
            if (take_exclusive(addr)) {
                mem_.write<uint16_t>(addr, static_cast<uint16_t>(reg(rt)));
                set_reg(rd, 0);
            } else set_reg(rd, 1);
            return;
        case 0xF: set_exclusive(addr); set_reg(rd, mem_.read<uint16_t>(addr)); return;  // LDREXH
        default: undefined(insn); return;
    }
}

// A5.2.8 Extra load/store: halfword, signed byte, and doubleword.
void Cpu::a32_extra_load_store(uint32_t insn) {
    const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
    const bool imm = (insn >> 22) & 1, w = (insn >> 21) & 1, load = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF, rt = (insn >> 12) & 0xF;
    const unsigned op2 = (insn >> 5) & 3;

    const uint32_t offset = imm ? ((((insn >> 8) & 0xF) << 4) | (insn & 0xF)) : reg(insn & 0xF);
    const uint32_t base = reg(rn);
    const uint32_t offset_addr = u ? base + offset : base - offset;
    const uint32_t addr = p ? offset_addr : base;

    switch (op2) {
        case 1:                                            // LDRH / STRH
            if (load) do_load(rt, addr, 2, false); else do_store(rt, addr, 2);
            break;
        case 2:
            if (load) do_load(rt, addr, 1, true);          // LDRSB
            else {                                         // LDRD
                r[rt] = mem_.read<uint32_t>(addr);
                r[rt + 1] = mem_.read<uint32_t>(addr + 4);
            }
            break;
        default:
            if (load) do_load(rt, addr, 2, true);          // LDRSH
            else {                                         // STRD
                mem_.write<uint32_t>(addr, reg(rt));
                mem_.write<uint32_t>(addr + 4, reg(rt + 1));
            }
            break;
    }
    // Writeback last, and skipped when the load already wrote the base -- the
    // loaded value wins.
    if ((!p || w) && rn != 15) {
        const bool loaded_base = load && (rt == rn || (op2 == 2 && !load));
        if (!loaded_base) r[rn] = offset_addr;
    }
}

// A5.3 Load/store word and unsigned byte.
void Cpu::a32_load_store(uint32_t insn) {
    const bool reg_offset = (insn >> 25) & 1;
    const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
    const bool byte = (insn >> 22) & 1, w = (insn >> 21) & 1, load = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF, rt = (insn >> 12) & 0xF;

    uint32_t offset;
    if (!reg_offset) {
        offset = insn & 0xFFF;
    } else {
        unsigned stype, amt;
        decode_imm_shift((insn >> 5) & 3, (insn >> 7) & 0x1F, &stype, &amt);
        const unsigned rm = insn & 0xF;
        offset = stype == kRRX ? rrx_c(reg(rm), c).result : shift_c(reg(rm), stype, amt, c).result;
    }
    // A literal load's base is the aligned PC. In ARM state pc_val() is already
    // word-aligned, but Thumb's is not, and this helper is shared -- so align it
    // here rather than relying on the caller's state.
    const uint32_t base = rn == 15 ? (pc_val() & ~3u) : reg(rn);
    const uint32_t offset_addr = u ? base + offset : base - offset;
    const uint32_t addr = p ? offset_addr : base;

    if (load) do_load(rt, addr, byte ? 1 : 4, false);
    else do_store(rt, addr, byte ? 1 : 4);
    if ((!p || w) && rn != 15 && !(load && rt == rn)) r[rn] = offset_addr;
}

// A5.4 Media instructions.
void Cpu::a32_media(uint32_t insn) {
    const unsigned op1 = (insn >> 20) & 0x1F;
    const unsigned op2 = (insn >> 5) & 7;
    const unsigned rd = (insn >> 12) & 0xF, rn = (insn >> 16) & 0xF, rm = insn & 0xF;

    // Bitfield instructions first: they are the ones a compiler emits constantly.
    //
    // These are keyed on bits **24..21**, not on the usual op1 (24..20), because bit
    // 20 belongs to the width/msb field. Masking op1 instead makes SBFX and UBFX
    // differ only in a bit that is really part of an operand: an earlier draft
    // matched `(op1 & 0x1E) == 0x1A`, which caught SBFX (1101x) and missed UBFX
    // (1111x) entirely -- so `(uint8_t)(h >> 8)` fell through to the extend group
    // and produced a byte from the wrong place.
    const unsigned bfop = (insn >> 21) & 0xF;
    if ((bfop == 13 || bfop == 15) && (op2 & 3) == 2) {    // SBFX / UBFX
        const unsigned lsb = (insn >> 7) & 0x1F, widthm1 = (insn >> 16) & 0x1F;
        const unsigned width = widthm1 + 1;
        uint32_t val = reg(rm);
        val = width == 32 ? (val >> lsb) : ((val >> lsb) & ((1u << width) - 1));
        if (bfop == 13 && width < 32 && (val & (1u << widthm1)))
            val |= ~((1u << width) - 1);                   // SBFX sign-extends
        set_reg(rd, val);
        return;
    }
    if (bfop == 14 && (op2 & 3) == 0) {                    // BFC / BFI
        const unsigned lsb = (insn >> 7) & 0x1F, msb = (insn >> 16) & 0x1F;
        if (msb < lsb) { undefined(insn); return; }
        const uint32_t mask = msb == 31 ? (0xFFFFFFFFu << lsb)
                                        : (((1u << (msb - lsb + 1)) - 1) << lsb);
        const uint32_t src = rm == 15 ? 0 : reg(rm);       // Rn == 15 is BFC
        set_reg(rd, (r[rd] & ~mask) | ((src << lsb) & mask));
        return;
    }
    if (op1 == 0x1F && op2 == 7) fail("permanently undefined instruction (UDF)", insn);

    if ((op1 & 0x18) == 0x10) {                            // signed multiplies, divides
        const unsigned ra = (insn >> 12) & 0xF;
        const unsigned rdhi = (insn >> 16) & 0xF;
        const uint32_t a = reg(insn & 0xF), b = reg((insn >> 8) & 0xF);
        switch (op1 & 7) {
            case 0: {                                      // SMLAD / SMUAD / SMLSD / SMUSD
                const bool swap = (op2 >> 1) & 1 ? false : ((insn >> 5) & 1) != 0;
                (void)swap;
                const bool m = (insn >> 5) & 1;
                const int32_t bl = static_cast<int16_t>(m ? (b >> 16) : b);
                const int32_t bh = static_cast<int16_t>(m ? b : (b >> 16));
                const int32_t al = static_cast<int16_t>(a);
                const int32_t ah = static_cast<int16_t>(a >> 16);
                const bool sub = (op2 & 2) != 0;
                int64_t res = sub ? (static_cast<int64_t>(al) * bl - static_cast<int64_t>(ah) * bh)
                                  : (static_cast<int64_t>(al) * bl + static_cast<int64_t>(ah) * bh);
                if (ra != 15) res += static_cast<int32_t>(reg(ra));
                set_reg(rdhi, static_cast<uint32_t>(res));
                return;
            }
            case 1:                                        // SDIV
                if (b == 0) set_reg(rdhi, 0);
                else if (a == 0x80000000u && b == 0xFFFFFFFFu) set_reg(rdhi, 0x80000000u);
                else set_reg(rdhi, static_cast<uint32_t>(static_cast<int32_t>(a) / static_cast<int32_t>(b)));
                return;
            case 3:                                        // UDIV
                set_reg(rdhi, b == 0 ? 0 : a / b);
                return;
            case 4: {                                      // SMLALD / SMLSLD
                const unsigned dlo = (insn >> 12) & 0xF;
                const bool m = (insn >> 5) & 1;
                const int32_t bl = static_cast<int16_t>(m ? (b >> 16) : b);
                const int32_t bh = static_cast<int16_t>(m ? b : (b >> 16));
                const int64_t acc = static_cast<int64_t>((static_cast<uint64_t>(reg(rdhi)) << 32) | reg(dlo));
                const bool sub = (op2 & 2) != 0;
                const int64_t prod = sub
                    ? static_cast<int64_t>(static_cast<int16_t>(a)) * bl -
                          static_cast<int64_t>(static_cast<int16_t>(a >> 16)) * bh
                    : static_cast<int64_t>(static_cast<int16_t>(a)) * bl +
                          static_cast<int64_t>(static_cast<int16_t>(a >> 16)) * bh;
                const int64_t res = acc + prod;
                r[dlo] = static_cast<uint32_t>(res);
                r[rdhi] = static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32);
                return;
            }
            case 5: {                                      // SMMLA / SMMUL / SMMLS
                int64_t res = static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b);
                if (ra != 15)
                    res = (op2 & 2) ? (static_cast<int64_t>(static_cast<int32_t>(reg(ra))) << 32) - res
                                    : (static_cast<int64_t>(static_cast<int32_t>(reg(ra))) << 32) + res;
                if ((insn >> 5) & 1) res += 0x80000000ll;   // the R bit rounds
                set_reg(rdhi, static_cast<uint32_t>(static_cast<uint64_t>(res) >> 32));
                return;
            }
            default: break;
        }
        undefined(insn);
        return;
    }

    if ((op1 & 0x18) == 0x18 && op2 == 0) {                // USAD8 / USADA8
        const unsigned ra = (insn >> 12) & 0xF;
        const uint32_t a = reg(insn & 0xF), b = reg((insn >> 8) & 0xF);
        uint32_t sum = ra == 15 ? 0 : reg(ra);
        for (unsigned i = 0; i < 4; ++i) {
            const int d = static_cast<int>((a >> (i * 8)) & 0xFF) -
                          static_cast<int>((b >> (i * 8)) & 0xFF);
            sum += static_cast<uint32_t>(d < 0 ? -d : d);
        }
        set_reg(rn, sum);
        return;
    }

    if ((op1 & 0x1C) == 0) {                               // parallel add/sub, signed
        // Not yet needed by any guest here. Refused rather than approximated.
        undefined(insn);
        return;
    }
    if ((op1 & 0x1C) == 4) {                               // parallel add/sub, unsigned
        undefined(insn);
        return;
    }

    // A5.4.3 Packing, unpacking, saturation and reversal.
    //
    // Keyed exactly on the manual's table: op1 is bits 22..20 and op2 is bits 7..5.
    // The extends are the trap here -- SXTB is op1 = 010 and SXTH is op1 = 011, so
    // the *width* is the low bit of op1 and the signedness is bit 2. An earlier
    // draft read the signedness from bit 22 and the width from nowhere, which made
    // `(uint8_t)(h >> 8)` come out as a halfword extract: a number that looks like
    // a plausible byte until the input has anything in bits 15..8.
    auto ror = [](uint32_t x, unsigned amt) {
        return amt ? ((x >> amt) | (x << (32 - amt))) : x;
    };
    const unsigned rot = ((insn >> 10) & 3) * 8;
    const unsigned sub = (insn >> 20) & 7;

    if (op2 == 3) {                                        // the extend family
        const uint32_t x = ror(reg(rm), rot);
        uint32_t val;
        switch (sub) {
            case 0:                                        // SXTB16 / SXTAB16
            case 4: {                                      // UXTB16 / UXTAB16
                const bool sign = sub == 0;
                uint32_t lo = sign ? static_cast<uint32_t>(static_cast<int8_t>(x)) & 0xFFFF
                                   : (x & 0xFF);
                uint32_t hi = sign ? static_cast<uint32_t>(static_cast<int8_t>(x >> 16)) & 0xFFFF
                                   : ((x >> 16) & 0xFF);
                if (rn != 15) { lo = (lo + reg(rn)) & 0xFFFF; hi = (hi + (reg(rn) >> 16)) & 0xFFFF; }
                set_reg(rd, (hi << 16) | lo);
                return;
            }
            case 2: val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(x))); break;
            case 3: val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(x))); break;
            case 6: val = x & 0xFF; break;
            case 7: val = x & 0xFFFF; break;
            default: undefined(insn); return;
        }
        if (rn != 15) val += reg(rn);                      // the "A" (accumulate) forms
        set_reg(rd, val);
        return;
    }

    if (sub == 0) {
        if ((op2 & 1) == 0) {                              // PKHBT / PKHTB
            const unsigned imm5 = (insn >> 7) & 0x1F;
            const uint32_t rnv = reg(rn);
            if (op2 & 2) {                                 // PKHTB: bottom from shifted Rm
                const uint32_t sh = shift_c(reg(rm), kASR, imm5 ? imm5 : 32, c).result;
                set_reg(rd, (rnv & 0xFFFF0000u) | (sh & 0xFFFF));
            } else {
                const uint32_t sh = shift_c(reg(rm), kLSL, imm5, c).result;
                set_reg(rd, (rnv & 0xFFFF) | (sh & 0xFFFF0000u));
            }
            return;
        }
        if (op2 == 5) {                                    // SEL, on the GE bits
            const uint32_t a = reg(rn), b = reg(rm);
            uint32_t res = 0;
            for (unsigned i = 0; i < 4; ++i)
                res |= ((ge >> i) & 1 ? a : b) & (0xFFu << (i * 8));
            set_reg(rd, res);
            return;
        }
    }

    // SSAT and USAT: op1 = 01x and 11x, with op2's low bit clear.
    if ((op2 & 1) == 0 && (sub == 2 || sub == 3 || sub == 6 || sub == 7)) {
        const bool unsig = sub >= 6;
        const unsigned sat_imm = (insn >> 16) & 0x1F;
        unsigned stype, amt;
        decode_imm_shift(((insn >> 6) & 1) ? kASR : kLSL, (insn >> 7) & 0x1F, &stype, &amt);
        const int64_t val = static_cast<int32_t>(
            stype == kRRX ? rrx_c(reg(rm), c).result : shift_c(reg(rm), stype, amt, c).result);
        if (unsig) {
            const int64_t hi = (1ll << sat_imm) - 1;
            if (val > hi) { q = true; set_reg(rd, static_cast<uint32_t>(hi)); }
            else if (val < 0) { q = true; set_reg(rd, 0); }
            else set_reg(rd, static_cast<uint32_t>(val));
        } else {
            // SSAT's immediate is "saturate to N bits" with N = sat_imm + 1, unlike
            // USAT's, which is N = sat_imm. The two fields look identical and mean
            // different things.
            const int64_t hi = (1ll << sat_imm) - 1, lo = -(1ll << sat_imm);
            if (val > hi) { q = true; set_reg(rd, static_cast<uint32_t>(hi)); }
            else if (val < lo) { q = true; set_reg(rd, static_cast<uint32_t>(lo)); }
            else set_reg(rd, static_cast<uint32_t>(val));
        }
        return;
    }

    // REV / REV16 (op1 = 011) and RBIT / REVSH (op1 = 111).
    if (sub == 3 || sub == 7) {
        const uint32_t x = reg(rm);
        const bool high = sub == 7;
        if (op2 == 1) {
            if (high) {                                    // RBIT
                uint32_t res = 0;
                for (unsigned i = 0; i < 32; ++i) if (x & (1u << i)) res |= 1u << (31 - i);
                set_reg(rd, res);
            } else {                                       // REV
                set_reg(rd, (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24));
            }
            return;
        }
        if (op2 == 5) {
            if (high)                                      // REVSH
                set_reg(rd, static_cast<uint32_t>(static_cast<int32_t>(
                    static_cast<int16_t>(((x & 0xFF) << 8) | ((x >> 8) & 0xFF)))));
            else                                           // REV16
                set_reg(rd, ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu));
            return;
        }
    }
    undefined(insn);
}

// A5.5 Block data transfer.
void Cpu::a32_block(uint32_t insn) {
    const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
    const bool s = (insn >> 22) & 1, w = (insn >> 21) & 1, load = (insn >> 20) & 1;
    const unsigned rn = (insn >> 16) & 0xF;
    const uint32_t list = insn & 0xFFFF;
    if (load) do_ldm(rn, list, u, p, w, s);
    else do_stm(rn, list, u, p, w);
}

// A5.5 B / BL.
void Cpu::a32_branch(uint32_t insn) {
    const int32_t imm = (static_cast<int32_t>(insn << 8) >> 8) * 4;   // sign-extend imm24
    if ((insn >> 24) & 1) set_lr(cur_pc_ + 4);                        // BL
    branch_write_pc(pc_val() + imm);
}

// A5.6 Coprocessor instructions and supervisor call.
void Cpu::a32_coproc(uint32_t insn) {
    if (((insn >> 24) & 0xF) == 0xF) {                     // SVC
        if (on_svc && on_svc(insn & 0x00FFFFFF)) return;
        halted = true;
        return;
    }
    const unsigned coproc = (insn >> 8) & 0xF;
    if (coproc == 10 || coproc == 11) {
        if (exec_vfp(insn)) return;
    } else if (coproc == 15) {
        if (exec_cp15(insn)) return;
    } else if (coproc == 14) {
        // The debug coprocessor. A guest that reads DBGDIDR is probing for a
        // debugger; answering zero and moving on is what a CPU without one does.
        if (((insn >> 4) & 1) && ((insn >> 20) & 1)) { set_reg((insn >> 12) & 0xF, 0); return; }
        if (((insn >> 4) & 1)) return;
    }
    undefined(insn);
}

// A5.7 Unconditional instructions. The `cond == 0b1111` space, which is where
// ARMv6 and v7 put everything that could not be conditional: NEON, the barriers,
// the hints, and BLX to a Thumb target.
void Cpu::a32_unconditional(uint32_t insn) {
    if ((insn & 0xFE000000u) == 0xF2000000u) {             // Advanced SIMD data-processing
        if (exec_neon(insn)) return;
        undefined(insn);
        return;
    }
    if ((insn & 0xFF100000u) == 0xF4000000u) {             // Advanced SIMD load/store
        if (exec_neon_ldst(insn)) return;
        undefined(insn);
        return;
    }
    if ((insn & 0xFE000000u) == 0xFA000000u) {             // BLX (immediate)
        const int32_t imm = ((static_cast<int32_t>(insn << 8) >> 8) * 4) |
                            (((insn >> 24) & 1) << 1);
        // The target is computed from the PC **as it reads in the current state**,
        // which is ARM here -- so the base must be taken before `thumb` changes.
        // Switching first makes pc_val() answer +4 instead of +8, and the branch
        // lands four bytes short: inside the callee, at a real instruction, which
        // runs and returns.
        const uint32_t target = pc_val() + imm;
        set_lr(cur_pc_ + 4);
        thumb = true;
        pc = target & ~1u;
        return;
    }
    const unsigned op1 = (insn >> 20) & 0xFF;
    if ((op1 & 0xF0) == 0x50 || (op1 & 0xF0) == 0x40) {
        // PLD / PLI: prefetch hints. Nothing to do, and that is not a stub -- a
        // prefetch has no architectural effect.
        if ((insn & 0xF0000000u) == 0xF0000000u) return;
    }
    if ((insn & 0xFFF00000u) == 0xF5700000u) {             // CLREX / DSB / DMB / ISB
        const unsigned op = (insn >> 4) & 0xF;
        if (op == 1) clear_exclusive();                    // CLREX
        return;                                            // the barriers: one CPU
    }
    if ((insn & 0xFFF10020u) == 0xF1000000u) return;       // CPS: no privileged state
    if ((insn & 0xFFFF00F0u) == 0xF1010000u) return;       // SETEND: little-endian only
    // The coprocessor "2" forms -- LDC2/STC2/MCR2/MRC2 -- reach NEON and VFP the
    // same way their conditional siblings do.
    const unsigned coproc = (insn >> 8) & 0xF;
    if ((coproc == 10 || coproc == 11) && (insn & 0x0C000000u) != 0) {
        if (exec_vfp(insn)) return;
    }
    undefined(insn);
}

// The system coprocessor. A userland guest reaches exactly three registers through
// it, and every one of them matters:
//
//   c13,c0,2 (TPIDRURW) and c13,c0,3 (TPIDRURO) -- the thread pointer. ARMv7
//   glibc and musl read TPIDRURO directly for __thread; older binaries call the
//   kernel helper page instead, which this emulator implements as real ARM code
//   that ends up right back here.
//
//   c0,c0,1 (CTR) -- the cache type register. glibc's memcpy reads the cache line
//   size out of it, and a zero there makes the loop that divides by it fail in a
//   way that looks nothing like a cache question.
bool Cpu::exec_cp15(uint32_t insn) {
    if (((insn >> 24) & 0xF) != 0xE || !((insn >> 4) & 1)) return false;   // MCR/MRC only
    const bool load = (insn >> 20) & 1;
    const unsigned opc1 = (insn >> 21) & 7, crn = (insn >> 16) & 0xF;
    const unsigned rt = (insn >> 12) & 0xF, crm = insn & 0xF, opc2 = (insn >> 5) & 7;

    if (crn == 13 && crm == 0 && opc1 == 0) {
        if (opc2 == 2 || opc2 == 3) {                      // TPIDRURW / TPIDRURO
            if (load) set_reg(rt, tpidruro); else tpidruro = reg(rt);
            return true;
        }
    }
    if (load && crn == 0 && crm == 0 && opc1 == 0) {
        switch (opc2) {
            // CTR: format v7, 4-word (16-byte) I and D lines, 64-byte DminLine
            // exponent 4 -> 16 words. The number itself only has to be sane and
            // consistent; glibc divides by it.
            case 1: set_reg(rt, 0x8444C004u); return true;
            case 0: set_reg(rt, 0x410FC073u); return true;   // MIDR: a Cortex-A7
            case 5: set_reg(rt, 0); return true;             // MPIDR: CPU 0
            default: set_reg(rt, 0); return true;
        }
    }
    // Cache maintenance (c7) is write-only and has no architectural effect here:
    // there are no caches to clean and the instruction stream is re-read every time.
    if (crn == 7 && !load) return true;
    if (load) { set_reg(rt, 0); return true; }
    return true;
}

}  // namespace arm
