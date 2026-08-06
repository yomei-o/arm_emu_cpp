// Advanced SIMD (NEON).
//
// Grown strictly by demand, and loud about the gaps: an encoding this file does
// not recognise returns false, which the caller turns into SIGILL or a stop with
// the encoding printed. It never silently does nothing. A guest that keeps running
// on a register a NEON instruction should have written fails somewhere unrelated
// and hours later -- that failure mode is the reason this project refuses the
// unknown rather than approximating it.
//
// The register file is `Cpu::d`, and a quad register is simply the pair
// (d[n], d[n+1]). Everything here is written as a loop over d-registers and then
// over lanes within one, so a D-form and a Q-form instruction share their body and
// differ only in the loop bound -- which is what the architecture actually says
// and half the size of writing them separately.
#include "cpu.h"
#include <cmath>
#include <cstring>

namespace arm {

namespace {

inline uint64_t lane_mask(unsigned esize) {
    return esize == 8 ? ~0ull : ((1ull << (esize * 8)) - 1);
}
inline int64_t sign_extend(uint64_t v, unsigned esize) {
    const unsigned bits = esize * 8;
    if (bits == 64) return static_cast<int64_t>(v);
    const uint64_t m = 1ull << (bits - 1);
    return static_cast<int64_t>((v ^ m) - m);
}
inline float as_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t as_u32(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// AdvSIMDExpandImm: the one-register modified immediate. Twelve `cmode` cases,
// and they are not a pattern -- a byte shifted to one of four positions, a
// halfword to one of two, a shifting-ones form, a replicated byte, and a
// bit-per-byte float. The aarch64 sibling got this group wrong by treating
// ORR/BIC (immediate) as MOVI, which replaced a strlen NUL mask with a constant
// and made every string in glibc two bytes short. It is written out case by case
// here for that reason.
uint64_t adv_simd_expand_imm(unsigned op, unsigned cmode, uint32_t imm8) {
    const uint64_t b = imm8;
    auto rep8 = [](uint64_t x) { return x | (x << 32); };
    switch (cmode >> 1) {
        case 0: return rep8(b);                                    // 0000: 000000ab
        case 1: return rep8(b << 8);                               // 0010
        case 2: return rep8(b << 16);                              // 0100
        case 3: return rep8(b << 24);                              // 0110
        case 4: return rep8(b | (b << 16));                        // 1000: 16-bit
        case 5: return rep8((b << 8) | (b << 24));                 // 1010
        case 6:                                                    // 1100/1101: shifting ones
            return rep8((cmode & 1) ? ((b << 16) | 0xFFFFu) : ((b << 8) | 0xFFu));
        default:
            if (cmode == 0xE && op == 0) {                         // replicate each bit to a byte
                uint64_t out = 0;
                for (unsigned i = 0; i < 8; ++i)
                    if (b & (1u << i)) out |= 0xFFull << (i * 8);
                return out;
            }
            if (cmode == 0xF && op == 0) {                         // a float immediate
                const uint32_t f = ((b & 0x80) << 24) | (((b & 0x40) ? 0 : 1) << 30) |
                                   (((b & 0x40) ? 0x1Fu : 0u) << 25) | ((b & 0x3F) << 19);
                return rep8(f);
            }
            if (cmode == 0xE && op == 1) {                         // a 64-bit byte mask
                uint64_t out = 0;
                for (unsigned i = 0; i < 8; ++i)
                    if (b & (1u << i)) out |= 0xFFull << (i * 8);
                return out;
            }
            return rep8(b);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// A7.7 Advanced SIMD element and structure load/store
// ---------------------------------------------------------------------------
//
// These are the instructions a memcpy is made of, so they are implemented in
// full rather than by demand. The addressing is shared by every form: a base
// register, an optional alignment hint that has no effect here, and a writeback
// that is either the number of bytes transferred or a register.
bool Cpu::exec_neon_ldst(uint32_t insn) {
    const unsigned rn = (insn >> 16) & 0xF, rm = insn & 0xF;
    const unsigned vd = (((insn >> 22) & 1) << 4) | ((insn >> 12) & 0xF);
    const bool load = (insn >> 21) & 1;
    const bool single = (insn >> 23) & 1;                // one element, or a whole structure
    const unsigned type = (insn >> 8) & 0xF;
    uint32_t addr = reg(rn);
    unsigned bytes = 0;

    if (!single) {                                       // VLDn / VSTn (multiple structures)
        const unsigned size = (insn >> 6) & 3;
        const unsigned esize = 1u << size;
        // The `type` field says how many registers and with what interleave. Only
        // the un-interleaved forms (VLD1/VST1) and the two-way interleave are used
        // by the guests here; the rest are refused rather than guessed at.
        unsigned regs, step, n;
        switch (type) {
            case 0x7: regs = 1; step = 1; n = 1; break;   // VLD1, one register
            case 0xA: regs = 2; step = 1; n = 1; break;   // VLD1, two registers
            case 0x6: regs = 3; step = 1; n = 1; break;   // VLD1, three registers
            case 0x2: regs = 4; step = 1; n = 1; break;   // VLD1, four registers
            case 0x8: regs = 2; step = 1; n = 2; break;   // VLD2
            case 0x9: regs = 2; step = 2; n = 2; break;   // VLD2, stride 2
            case 0x3: regs = 4; step = 2; n = 2; break;   // VLD2, four registers
            case 0x4: regs = 3; step = 1; n = 3; break;   // VLD3
            case 0x5: regs = 3; step = 2; n = 3; break;
            case 0x0: regs = 4; step = 1; n = 4; break;   // VLD4
            case 0x1: regs = 4; step = 2; n = 4; break;
            default: return false;
        }
        if (n == 1) {
            for (unsigned i = 0; i < regs; ++i) {
                const unsigned rd = (vd + i) & 31;
                if (load) {
                    uint64_t val = 0;
                    for (unsigned k = 0; k < 8; ++k)
                        val |= static_cast<uint64_t>(mem_.read<uint8_t>(addr + k)) << (k * 8);
                    d[rd] = val;
                } else {
                    for (unsigned k = 0; k < 8; ++k)
                        mem_.write<uint8_t>(addr + k, static_cast<uint8_t>(d[rd] >> (k * 8)));
                }
                addr += 8;
            }
            bytes = regs * 8;
        } else {
            // Interleaved: element j of structure i goes to register j, lane i.
            const unsigned lanes = 8 / esize;
            for (unsigned i = 0; i < lanes; ++i) {
                for (unsigned j = 0; j < n; ++j) {
                    const unsigned rd = (vd + j * step) & 31;
                    if (load) {
                        uint64_t val = 0;
                        for (unsigned k = 0; k < esize; ++k)
                            val |= static_cast<uint64_t>(mem_.read<uint8_t>(addr + k)) << (k * 8);
                        set_lane(&d[rd], esize, i, val);
                    } else {
                        const uint64_t val = get_lane(d[rd], esize, i);
                        for (unsigned k = 0; k < esize; ++k)
                            mem_.write<uint8_t>(addr + k, static_cast<uint8_t>(val >> (k * 8)));
                    }
                    addr += esize;
                }
            }
            bytes = lanes * n * esize;
            (void)regs;
        }
    } else {
        const unsigned size = (insn >> 10) & 3;
        if (size == 3) {                                 // VLD1/2/3/4 to all lanes
            const unsigned esz = 1u << ((insn >> 6) & 3);
            const unsigned n = ((insn >> 8) & 3) + 1;
            const bool two = (insn >> 5) & 1;
            for (unsigned j = 0; j < n; ++j) {
                uint64_t val = 0;
                for (unsigned k = 0; k < esz; ++k)
                    val |= static_cast<uint64_t>(mem_.read<uint8_t>(addr + k)) << (k * 8);
                uint64_t rep = 0;
                for (unsigned i = 0; i < 8 / esz; ++i) rep |= (val & lane_mask(esz)) << (i * esz * 8);
                d[(vd + j) & 31] = rep;
                if (two) d[(vd + j + 1) & 31] = rep;
                addr += esz;
            }
            bytes = n * esz;
        } else {                                         // one lane
            const unsigned esize = 1u << size;
            const unsigned n = ((insn >> 8) & 3) + 1;
            unsigned index, step = 1;
            if (size == 0) index = (insn >> 5) & 7;
            else if (size == 1) { index = (insn >> 6) & 3; step = ((insn >> 5) & 1) ? 2 : 1; }
            else { index = (insn >> 7) & 1; step = ((insn >> 6) & 1) ? 2 : 1; }
            for (unsigned j = 0; j < n; ++j) {
                const unsigned rd = (vd + j * step) & 31;
                if (load) {
                    uint64_t val = 0;
                    for (unsigned k = 0; k < esize; ++k)
                        val |= static_cast<uint64_t>(mem_.read<uint8_t>(addr + k)) << (k * 8);
                    set_lane(&d[rd], esize, index, val);
                } else {
                    const uint64_t val = get_lane(d[rd], esize, index);
                    for (unsigned k = 0; k < esize; ++k)
                        mem_.write<uint8_t>(addr + k, static_cast<uint8_t>(val >> (k * 8)));
                }
                addr += esize;
            }
            bytes = n * esize;
        }
    }

    // Writeback: Rm == 15 means "no writeback", Rm == 13 means "advance by the
    // transfer size", anything else is a register increment. That three-way
    // meaning of one field is the encoding's, not an approximation.
    if (rm != 15) r[rn] = rm == 13 ? reg(rn) + bytes : reg(rn) + reg(rm);
    return true;
}

// ---------------------------------------------------------------------------
// A7.4 Advanced SIMD data-processing
// ---------------------------------------------------------------------------

bool Cpu::exec_neon(uint32_t insn) {
    const unsigned u = (insn >> 24) & 1;
    const unsigned c = (insn >> 8) & 0xF;
    const unsigned bit4 = (insn >> 4) & 1;
    const bool quad = (insn >> 6) & 1;
    const unsigned regs = quad ? 2 : 1;

    const unsigned vd = (((insn >> 22) & 1) << 4) | ((insn >> 12) & 0xF);
    const unsigned vn = (((insn >> 7) & 1) << 4) | ((insn >> 16) & 0xF);
    const unsigned vm = (((insn >> 5) & 1) << 4) | (insn & 0xF);

    // ---- one register and a modified immediate --------------------------------
    // 1111 001 i 1 D 000 imm3 Vd cmode 0 Q op 1 imm4 -- note bit 24 is `i`, part of
    // the immediate, so it must not appear in the mask.
    if ((insn & 0xFEB80090u) == 0xF2800010u) {
        const unsigned cmode = (insn >> 8) & 0xF;
        const unsigned op = (insn >> 5) & 1;
        const uint32_t imm8 = (((insn >> 24) & 1) << 7) | (((insn >> 16) & 7) << 4) | (insn & 0xF);
        const uint64_t imm = adv_simd_expand_imm(op, cmode, imm8);
        for (unsigned i = 0; i < regs; ++i) {
            uint64_t& dst = d[(vd + i) & 31];
            // cmode 0xxx1 and 10xx1 with op are the ORR/BIC immediate forms, not
            // MOVI. Treating them as MOVI is the bug that cost the sibling project
            // a night: it replaces a mask the guest built with a constant.
            if ((cmode & 1) && cmode < 0xC) dst = op ? (dst & ~imm) : (dst | imm);
            else dst = op ? ~imm : imm;
        }
        return true;
    }

    // ---- three registers of the same length -----------------------------------
    // Selected by bit 23 alone. Bit 24 is **U**, the signedness -- so a mask that
    // includes it matches only half the group, and the *unsigned* half of NEON
    // (VSUB, VEOR/VBSL, the unsigned compares, max/min and VABD) is refused while
    // the signed half works. gcc's own vectoriser walked into that on the first
    // `vsub.i8` it emitted.
    if ((insn & 0xFE800000u) == 0xF2000000u) {
        const unsigned size = (insn >> 20) & 3;
        const unsigned esize = 1u << size;
        const unsigned lanes = 8 / esize;

        // The bitwise group: one opcode, six operations picked by `size` and U.
        if (c == 1 && bit4) {
            for (unsigned i = 0; i < regs; ++i) {
                const uint64_t x = d[(vn + i) & 31], y = d[(vm + i) & 31];
                uint64_t& dst = d[(vd + i) & 31];
                if (!u) {
                    switch (size) {
                        case 0: dst = x & y; break;                       // VAND
                        case 1: dst = x & ~y; break;                      // VBIC
                        case 2: dst = x | y; break;                       // VORR (and VMOV)
                        default: dst = x | ~y; break;                     // VORN
                    }
                } else {
                    // The three insert forms differ only in *which operand is the
                    // mask*, and that is the whole of what there is to get wrong.
                    // VBSL's mask is the **destination**: `vbsl q8, q10, q11` reads
                    // q8 as the selector. Reading Vn as the mask instead inverts the
                    // selection, which is not a crash -- it is a `a > b ? a - b : 0`
                    // that returns the difference exactly where it should have
                    // returned zero.
                    switch (size) {
                        case 0: dst = x ^ y; break;                       // VEOR
                        case 1: dst = (dst & x) | (~dst & y); break;      // VBSL
                        case 2: dst = (x & y) | (dst & ~y); break;        // VBIT
                        default: dst = (x & ~y) | (dst & y); break;       // VBIF
                    }
                }
            }
            return true;
        }

        // ---- the pairwise operations ------------------------------------------
        //
        // These do not fit the lane-for-lane loop below: they combine *adjacent
        // lanes within each source*, and the result's lower half comes from Vn while
        // its upper half comes from Vm. They are how a compiler finishes a reduction,
        // which is why a vectorised `sum += a[i]` reaches them and nothing else does.
        // Only the 64-bit form exists.
        const bool pairwise_int = (c == 0xA) || (c == 0xB && bit4);
        const bool pairwise_f32 = (c == 0xD && !bit4 && u && !(size & 1)) ||
                                  (c == 0xF && !bit4 && u);
        if ((pairwise_int || pairwise_f32) && !quad) {
            const unsigned es = pairwise_f32 ? 4 : esize;
            const unsigned n = 8 / es, half = n / 2;
            const uint64_t src[2] = {d[vn & 31], d[vm & 31]};
            uint64_t out = 0;
            for (unsigned s2 = 0; s2 < 2; ++s2) {
                for (unsigned k = 0; k < half; ++k) {
                    const uint64_t p = get_lane(src[s2], es, k * 2);
                    const uint64_t q = get_lane(src[s2], es, k * 2 + 1);
                    uint64_t v;
                    if (pairwise_f32) {
                        const float pf = as_f32(static_cast<uint32_t>(p));
                        const float qf = as_f32(static_cast<uint32_t>(q));
                        // VPADD.F32 is opcode 1101; VPMAX/VPMIN.F32 is 1111 with the
                        // size bit choosing between them.
                        v = c == 0xD ? as_u32(pf + qf)
                                     : as_u32((size & 1) ? (pf < qf ? pf : qf) : (pf > qf ? pf : qf));
                    } else if (c == 0xB) {
                        v = p + q;                                    // VPADD
                    } else {                                          // VPMAX / VPMIN
                        const int64_t ps = sign_extend(p, es), qs = sign_extend(q, es);
                        const bool want_max = !bit4;
                        v = u ? (want_max ? (p > q ? p : q) : (p < q ? p : q))
                              : static_cast<uint64_t>(want_max ? (ps > qs ? ps : qs)
                                                               : (ps < qs ? ps : qs));
                    }
                    set_lane(&out, es, s2 * half + k, v);
                }
            }
            d[vd & 31] = out;
            return true;
        }

        // Single-precision float, which shares this group behind opcodes 0xC-0xF.
        // `U` is part of the opcode here rather than a signedness bit, so every row
        // has to test it: 1101/op=0 is VADD/VSUB when U is clear and VPADD/VABD when
        // it is set, and 1101/op=1 is VMLA/VMLS against VMUL. Ignoring U turns a
        // multiply-accumulate into a multiply, which loses the accumulator.
        if (c >= 0xC && !(size & 2)) {
            for (unsigned i = 0; i < regs; ++i) {
                uint64_t out = 0;
                for (unsigned k = 0; k < 2; ++k) {
                    const float x = as_f32(static_cast<uint32_t>(get_lane(d[(vn + i) & 31], 4, k)));
                    const float y = as_f32(static_cast<uint32_t>(get_lane(d[(vm + i) & 31], 4, k)));
                    const float acc = as_f32(static_cast<uint32_t>(get_lane(d[(vd + i) & 31], 4, k)));
                    const bool sz = (size & 1) != 0;
                    uint32_t res;
                    if (c == 0xC && bit4 && !u) res = as_u32(sz ? acc - x * y : acc + x * y);  // VFMA/VFMS
                    else if (c == 0xD && !bit4 && !u) res = as_u32(sz ? x - y : x + y);        // VADD/VSUB
                    else if (c == 0xD && !bit4 && u) res = as_u32(x > y ? x - y : y - x);      // VABD
                    else if (c == 0xD && bit4 && !u) res = as_u32(sz ? acc - x * y : acc + x * y);  // VMLA/VMLS
                    else if (c == 0xD && bit4 && u) res = as_u32(x * y);                       // VMUL
                    else if (c == 0xE && !bit4) {                                              // VCEQ/VCGE/VCGT
                        const bool t = !u ? (x == y) : (sz ? (x > y) : (x >= y));
                        res = t ? 0xFFFFFFFFu : 0;
                    } else if (c == 0xF && !bit4 && !u)
                        res = as_u32(sz ? (x < y ? x : y) : (x > y ? x : y));                  // VMAX/VMIN
                    else return false;
                    out |= static_cast<uint64_t>(res) << (k * 32);
                }
                d[(vd + i) & 31] = out;
            }
            return true;
        }

        for (unsigned i = 0; i < regs; ++i) {
            const uint64_t xr = d[(vn + i) & 31], yr = d[(vm + i) & 31];
            uint64_t out = d[(vd + i) & 31];
            for (unsigned k = 0; k < lanes; ++k) {
                const uint64_t xu = get_lane(xr, esize, k), yu = get_lane(yr, esize, k);
                const int64_t xs = sign_extend(xu, esize), ys = sign_extend(yu, esize);
                const uint64_t acc = get_lane(out, esize, k);
                // The true, un-truncated result of the lane's add or subtract, in a
                // type wide enough to hold it. The halving and saturating forms are
                // *defined* on this value rather than on a wrapped one, and computing
                // them from the wrapped result is wrong in exactly the cases they
                // exist for: `vqsub.u8` is how a compiler writes
                // `a > b ? a - b : 0`, and a wrapping subtract turns the clamp into
                // 255-and-counting.
                const int64_t sum   = u ? static_cast<int64_t>(xu) + static_cast<int64_t>(yu)
                                        : xs + ys;
                const int64_t diff  = u ? static_cast<int64_t>(xu) - static_cast<int64_t>(yu)
                                        : xs - ys;
                // 64-bit lanes are refused for the saturating and halving forms
                // rather than approximated: their range does not fit the int64_t the
                // arithmetic above is done in, and a clamp computed against the wrong
                // bound is exactly the silent wrong answer this file exists to avoid.
                if (esize == 8 && (c == 0 || c == 1 || c == 2)) return false;
                const int64_t hi    = u ? static_cast<int64_t>((1ull << (esize * 8)) - 1)
                                        : ((1ll << (esize * 8 - 1)) - 1);
                const int64_t lo    = u ? 0 : -(1ll << (esize * 8 - 1));
                auto sat = [&](int64_t v) -> uint64_t {
                    return static_cast<uint64_t>(v > hi ? hi : (v < lo ? lo : v)) & lane_mask(esize);
                };

                uint64_t res;
                switch (c) {
                    case 0x0:                                                     // VQADD / VHADD
                        res = bit4 ? sat(sum) : (static_cast<uint64_t>(sum >> 1) & lane_mask(esize));
                        break;
                    case 0x1:                                                     // VRHADD
                        if (bit4) return false;                                   // the bitwise group
                        res = static_cast<uint64_t>((sum + 1) >> 1) & lane_mask(esize);
                        break;
                    case 0x2:                                                     // VQSUB / VHSUB
                        res = bit4 ? sat(diff) : (static_cast<uint64_t>(diff >> 1) & lane_mask(esize));
                        break;
                    case 0x3: res = (bit4 ? (u ? xu >= yu : xs >= ys)
                                          : (u ? xu > yu : xs > ys)) ? lane_mask(esize) : 0; break;
                    case 0x6: res = bit4 ? (u ? (xu < yu ? xu : yu) : static_cast<uint64_t>(xs < ys ? xs : ys))
                                         : (u ? (xu > yu ? xu : yu) : static_cast<uint64_t>(xs > ys ? xs : ys)); break;
                    case 0x7: {                                                    // VABD / VABA
                        const uint64_t diff = u ? (xu > yu ? xu - yu : yu - xu)
                                                : static_cast<uint64_t>(xs > ys ? xs - ys : ys - xs);
                        res = bit4 ? acc + diff : diff;
                        break;
                    }
                    case 0x8:
                        if (!bit4) res = u ? (xu - yu) : (xu + yu);                // VSUB / VADD
                        else res = (u ? (xu == yu) : ((xu & yu) != 0)) ? lane_mask(esize) : 0;  // VCEQ/VTST
                        break;
                    case 0x9:
                        // U selects the *polynomial* multiply here, not an unsigned
                        // one -- a different operation over GF(2), and refusing it is
                        // the only honest answer until something needs it.
                        if (bit4) { if (u) return false; res = xu * yu; }          // VMUL
                        else res = u ? (acc - xu * yu) : (acc + xu * yu);          // VMLA / VMLS
                        break;
                    case 0x4:                                                      // VSHL (register)
                        if (bit4) return false;                                    // VQSHL: not yet
                        {
                            const int8_t sh = static_cast<int8_t>(yu & 0xFF);
                            if (sh >= 0) res = sh >= 64 ? 0 : (xu << sh);
                            else if (-sh >= static_cast<int>(esize * 8)) res = u ? 0 : (xs < 0 ? ~0ull : 0);
                            else res = u ? (xu >> -sh) : static_cast<uint64_t>(xs >> -sh);
                        }
                        break;
                    default: return false;
                }
                set_lane(&out, esize, k, res);
            }
            d[(vd + i) & 31] = out;
        }
        return true;
    }

    // ---- two registers and a shift amount -------------------------------------
    // Same shape as the modified immediate above and told apart by imm6<5:3> being
    // non-zero, which is why that case has to come first.
    if ((insn & 0xFE800010u) == 0xF2800010u) {
        // The immediate's width tells you the element size: L:imm6 with the highest
        // set bit selecting 8, 16, 32 or 64.
        const unsigned imm6 = (insn >> 16) & 0x3F;
        const unsigned l = (insn >> 7) & 1;
        unsigned esize;
        unsigned shift;
        if (l) { esize = 8; shift = 64 - imm6; }
        else if (imm6 & 0x20) { esize = 4; shift = 64 - imm6; }
        else if (imm6 & 0x10) { esize = 2; shift = 32 - imm6; }
        else { esize = 1; shift = 16 - imm6; }
        const unsigned lanes = 8 / esize;
        const unsigned op = (insn >> 8) & 0xF;

        if (op == 0xA) {                                  // VSHLL / VMOVL: widening
            // `imm6` already names the **source** element -- that is what the
            // decode above derives `esize` from -- so halving it again turns
            // `vmovl.u16` into a byte widen: the same registers, half the elements,
            // and a result that is wrong in a way that looks like sign extension.
            const unsigned src_es = esize;
            const unsigned src_lanes = 8 / src_es;
            const unsigned wsize = src_es * 2, wper = 8 / wsize;
            const unsigned sh = src_es * 8 - shift;       // = imm6 - esize*8
            uint64_t out[2] = {0, 0};
            const uint64_t src = d[vm & 31];
            for (unsigned k = 0; k < src_lanes; ++k) {
                const uint64_t xu = get_lane(src, src_es, k);
                const uint64_t val = (u ? xu : static_cast<uint64_t>(sign_extend(xu, src_es))) << sh;
                set_lane(&out[k / wper], wsize, k % wper, val);
            }
            d[vd & 31] = out[0];
            d[(vd + 1) & 31] = out[1];
            return true;
        }

        for (unsigned i = 0; i < regs; ++i) {
            const uint64_t src = d[(vm + i) & 31];
            uint64_t out = d[(vd + i) & 31];
            for (unsigned k = 0; k < lanes; ++k) {
                const uint64_t xu = get_lane(src, esize, k);
                const int64_t xs = sign_extend(xu, esize);
                const uint64_t acc = get_lane(out, esize, k);
                uint64_t res;
                switch (op) {
                    case 0x0: res = u ? (xu >> shift) : static_cast<uint64_t>(xs >> shift); break;  // VSHR
                    case 0x1: res = acc + (u ? (xu >> shift) : static_cast<uint64_t>(xs >> shift)); break;  // VSRA
                    case 0x2: {                                                      // VRSHR
                        const uint64_t rnd = shift ? (1ull << (shift - 1)) : 0;
                        res = u ? ((xu + rnd) >> shift)
                                : static_cast<uint64_t>((xs + static_cast<int64_t>(rnd)) >> shift);
                        break;
                    }
                    case 0x5: res = xu << (esize * 8 - shift); break;                // VSHL (immediate)
                    default: return false;
                }
                set_lane(&out, esize, k, res);
            }
            d[(vd + i) & 31] = out;
        }
        return true;
    }

    // ---- two registers, miscellaneous (A7.4.7) --------------------------------
    //
    // The (A, B) table below was read out of the assembler rather than remembered:
    // every row was produced by assembling the mnemonic and reading the bits back.
    // That is worth saying because this group is exactly the shape that invites a
    // plausible wrong table -- a dozen operations picked by two small fields with no
    // pattern between them -- and the sibling project lost an evening to a
    // zero-compare table whose opcodes were each off by one. `vneg.s16` decoding as
    // `vcle #0` does not fault; it inverts a sign test.
    //
    //   A=00  0000 VREV64  0001 VREV32  0010 VREV16  010x VPADDL
    //         1000 VCLS    1001 VCLZ    1010 VCNT    1011 VMVN
    //         110x VPADAL  1110 VQABS   1111 VQNEG
    //   A=01  0000 VCGT#0  0001 VCGE#0  0010 VCEQ#0  0011 VCLE#0  0100 VCLT#0
    //         0110 VABS    0111 VNEG    (B<3> is F: the same rows in single precision)
    //   A=10  0000 VSWP    0001 VTRN    0010 VUZP    0011 VZIP
    //         0100 VMOVN (bit6=0) / VQMOVUN (bit6=1)   0101 VQMOVN   0110 VSHLL
    //   A=11  1010 VRECPE  1011 VRSQRTE  11xx VCVT float<->integer
    if ((insn & 0xFFB00810u) == 0xF3B00000u) {
        const unsigned size = (insn >> 18) & 3;
        const unsigned esize = 1u << size;
        const unsigned lanes = 8 / esize;
        const unsigned grp = (insn >> 16) & 3;
        const unsigned b = (insn >> 7) & 0xF;
        const bool flt = (b & 8) != 0;                 // the F bit, for the A=01 rows
        const bool bit6 = ((insn >> 6) & 1) != 0;

        if (grp == 0) {
            for (unsigned i = 0; i < regs; ++i) {
                const uint64_t src = d[(vm + i) & 31];
                const uint64_t acc_reg = d[(vd + i) & 31];
                if (b <= 2) {                          // VREV64 / VREV32 / VREV16
                    const unsigned region = 8u >> b;   // bytes in each reversed region
                    const unsigned per = region / esize;
                    uint64_t res = 0;
                    for (unsigned g = 0; g < 8 / region; ++g)
                        for (unsigned k = 0; k < per; ++k)
                            set_lane(&res, esize, g * per + (per - 1 - k),
                                     get_lane(src, esize, g * per + k));
                    d[(vd + i) & 31] = res;
                    continue;
                }
                if ((b & 0xE) == 0x4 || (b & 0xE) == 0xC) {   // VPADDL (010x) / VPADAL (110x)
                    const bool accumulate = (b & 0xE) == 0xC;
                    const bool unsig = (b & 1) != 0;
                    uint64_t res = 0;
                    for (unsigned k = 0; k < lanes / 2; ++k) {
                        const uint64_t p = get_lane(src, esize, k * 2);
                        const uint64_t q = get_lane(src, esize, k * 2 + 1);
                        int64_t sum = unsig ? static_cast<int64_t>(p) + static_cast<int64_t>(q)
                                            : sign_extend(p, esize) + sign_extend(q, esize);
                        if (accumulate) sum += static_cast<int64_t>(get_lane(acc_reg, esize * 2, k));
                        set_lane(&res, esize * 2, k, static_cast<uint64_t>(sum));
                    }
                    d[(vd + i) & 31] = res;
                    continue;
                }
                uint64_t res = 0;
                for (unsigned k = 0; k < lanes; ++k) {
                    const uint64_t x = get_lane(src, esize, k);
                    const int64_t xs = sign_extend(x, esize);
                    const unsigned bits = esize * 8;
                    uint64_t v;
                    switch (b) {
                        case 0x8: {                        // VCLS: leading sign bits
                            unsigned n = 0;
                            const uint64_t sign = (x >> (bits - 1)) & 1;
                            for (unsigned t = bits - 1; t-- > 0;) {
                                if (((x >> t) & 1) != sign) break;
                                ++n;
                            }
                            v = n;
                            break;
                        }
                        case 0x9: {                        // VCLZ
                            unsigned n = 0;
                            for (unsigned t = bits; t-- > 0;) {
                                if (x & (1ull << t)) break;
                                ++n;
                            }
                            v = n;
                            break;
                        }
                        case 0xA: {                        // VCNT
                            unsigned n = 0;
                            for (unsigned t = 0; t < bits; ++t) if (x & (1ull << t)) ++n;
                            v = n;
                            break;
                        }
                        case 0xB: v = ~x; break;           // VMVN
                        case 0xE: {                        // VQABS
                            const int64_t hi = (1ll << (bits - 1)) - 1;
                            const int64_t m = xs < 0 ? -xs : xs;
                            v = static_cast<uint64_t>(m > hi ? hi : m);
                            break;
                        }
                        case 0xF: {                        // VQNEG
                            const int64_t hi = (1ll << (bits - 1)) - 1;
                            const int64_t m = -xs;
                            v = static_cast<uint64_t>(m > hi ? hi : m);
                            break;
                        }
                        default: return false;
                    }
                    set_lane(&res, esize, k, v);
                }
                d[(vd + i) & 31] = res;
            }
            return true;
        }

        if (grp == 1) {                                    // compare with zero, VABS, VNEG
            for (unsigned i = 0; i < regs; ++i) {
                const uint64_t src = d[(vm + i) & 31];
                uint64_t res = 0;
                for (unsigned k = 0; k < lanes; ++k) {
                    const uint64_t x = get_lane(src, esize, k);
                    uint64_t v;
                    if (flt) {
                        if (esize != 4) return false;
                        const float f = as_f32(static_cast<uint32_t>(x));
                        switch (b & 7) {
                            case 0: v = f >  0 ? 0xFFFFFFFFu : 0; break;   // VCGT
                            case 1: v = f >= 0 ? 0xFFFFFFFFu : 0; break;   // VCGE
                            case 2: v = f == 0 ? 0xFFFFFFFFu : 0; break;   // VCEQ
                            case 3: v = f <= 0 ? 0xFFFFFFFFu : 0; break;   // VCLE
                            case 4: v = f <  0 ? 0xFFFFFFFFu : 0; break;   // VCLT
                            case 6: v = as_u32(f < 0 ? -f : f); break;     // VABS
                            case 7: v = as_u32(-f); break;                 // VNEG
                            default: return false;
                        }
                    } else {
                        const int64_t xs = sign_extend(x, esize);
                        switch (b & 7) {
                            case 0: v = xs >  0 ? lane_mask(esize) : 0; break;
                            case 1: v = xs >= 0 ? lane_mask(esize) : 0; break;
                            case 2: v = xs == 0 ? lane_mask(esize) : 0; break;
                            case 3: v = xs <= 0 ? lane_mask(esize) : 0; break;
                            case 4: v = xs <  0 ? lane_mask(esize) : 0; break;
                            case 6: v = static_cast<uint64_t>(xs < 0 ? -xs : xs); break;
                            case 7: v = static_cast<uint64_t>(-xs); break;
                            default: return false;
                        }
                    }
                    set_lane(&res, esize, k, v);
                }
                d[(vd + i) & 31] = res;
            }
            return true;
        }

        if (grp == 2) {
            if (b == 0) {                                  // VSWP
                for (unsigned i = 0; i < regs; ++i) {
                    const uint64_t t = d[(vd + i) & 31];
                    d[(vd + i) & 31] = d[(vm + i) & 31];
                    d[(vm + i) & 31] = t;
                }
                return true;
            }
            if (b == 1) {                                  // VTRN
                for (unsigned i = 0; i < regs; ++i) {
                    uint64_t x = d[(vd + i) & 31], y = d[(vm + i) & 31];
                    for (unsigned k = 0; k < lanes; k += 2) {
                        const uint64_t a1 = get_lane(x, esize, k + 1);
                        const uint64_t b0 = get_lane(y, esize, k);
                        set_lane(&x, esize, k + 1, b0);
                        set_lane(&y, esize, k, a1);
                    }
                    d[(vd + i) & 31] = x;
                    d[(vm + i) & 31] = y;
                }
                return true;
            }
            if (b == 2 || b == 3) {                        // VUZP / VZIP
                // Vd and Vm together are one 2N-element vector: VUZP takes it apart
                // into evens and odds, VZIP interleaves the two halves back. Both
                // write *both* registers, which is why neither fits the usual shape.
                const unsigned total = lanes * 2 * regs;
                uint64_t in[4] = {0, 0, 0, 0}, out[4] = {0, 0, 0, 0};
                for (unsigned i = 0; i < regs; ++i) {
                    in[i] = d[(vd + i) & 31];
                    in[regs + i] = d[(vm + i) & 31];
                }
                for (unsigned k = 0; k < total; ++k) {
                    const uint64_t v = get_lane(in[k / lanes], esize, k % lanes);
                    const unsigned dst = (b == 2)
                        ? ((k & 1) ? (total / 2 + k / 2) : (k / 2))
                        : ((k < total / 2) ? (k * 2) : ((k - total / 2) * 2 + 1));
                    set_lane(&out[dst / lanes], esize, dst % lanes, v);
                }
                for (unsigned i = 0; i < regs; ++i) {
                    d[(vd + i) & 31] = out[i];
                    d[(vm + i) & 31] = out[regs + i];
                }
                return true;
            }
            if (b == 4 || b == 5) {                        // VMOVN / VQMOVUN / VQMOVN
                const unsigned wsize = esize * 2;
                const unsigned wper = 8 / wsize;
                uint64_t res = 0;
                for (unsigned k = 0; k < lanes; ++k) {
                    const uint64_t w = get_lane(d[(vm + k / wper) & 31], wsize, k % wper);
                    uint64_t v;
                    if (b == 4 && !bit6) {
                        v = w;                             // VMOVN: plain truncation
                    } else {
                        const bool unsig_src = (b == 5) && bit6;      // VQMOVN.U
                        const bool unsig_dst = unsig_src || (b == 4); // VQMOVUN
                        const int64_t x = unsig_src ? static_cast<int64_t>(w)
                                                    : sign_extend(w, wsize);
                        const int64_t hi = unsig_dst ? static_cast<int64_t>(lane_mask(esize))
                                                     : ((1ll << (esize * 8 - 1)) - 1);
                        const int64_t lo = unsig_dst ? 0 : -(1ll << (esize * 8 - 1));
                        v = static_cast<uint64_t>(x > hi ? hi : (x < lo ? lo : x));
                    }
                    set_lane(&res, esize, k, v);
                }
                d[vd & 31] = res;
                return true;
            }
            if (b == 6) {                                  // VSHLL by the element size
                const unsigned wsize = esize * 2;
                const unsigned wper = 8 / wsize;
                uint64_t out[2] = {0, 0};
                for (unsigned k = 0; k < lanes; ++k) {
                    const uint64_t x = get_lane(d[vm & 31], esize, k);
                    set_lane(&out[k / wper], wsize, k % wper, x << (esize * 8));
                }
                d[vd & 31] = out[0];
                d[(vd + 1) & 31] = out[1];
                return true;
            }
            return false;
        }

        // grp == 3: the reciprocal estimates and the float/integer converts.
        if ((b & 0xC) == 0xC) {                            // VCVT float <-> integer
            if (esize != 4) return false;
            const bool to_int = (b & 2) != 0;
            const bool unsig = (b & 1) != 0;
            for (unsigned i = 0; i < regs; ++i) {
                uint64_t res = 0;
                for (unsigned k = 0; k < 2; ++k) {
                    const uint32_t x = static_cast<uint32_t>(get_lane(d[(vm + i) & 31], 4, k));
                    uint32_t v;
                    if (to_int) {
                        // Round toward zero and *saturate*: NEON's convert clamps at
                        // the ends of the range rather than wrapping, so a float that
                        // does not fit becomes the largest integer and not a random
                        // one.
                        double f = std::trunc(static_cast<double>(as_f32(x)));
                        if (unsig) {
                            if (!(f >= 0)) f = 0;
                            if (!(f <= 4294967295.0)) f = 4294967295.0;
                            v = static_cast<uint32_t>(f);
                        } else {
                            if (!(f >= -2147483648.0)) f = -2147483648.0;
                            if (!(f <= 2147483647.0)) f = 2147483647.0;
                            v = static_cast<uint32_t>(static_cast<int32_t>(f));
                        }
                    } else {
                        v = as_u32(unsig ? static_cast<float>(x)
                                         : static_cast<float>(static_cast<int32_t>(x)));
                    }
                    set_lane(&res, 4, k, v);
                }
                d[(vd + i) & 31] = res;
            }
            return true;
        }
        return false;
    }

    // ---- VDUP (scalar) and VTBL/VTBX -------------------------------------------
    // Both sit just outside the miscellaneous group's mask, because bit 11 is set in
    // each -- which is also why neither can be folded into the table above.
    if ((insn & 0xFFB00F90u) == 0xF3B00C00u) {             // VDUP (scalar)
        const unsigned imm4 = (insn >> 16) & 0xF;
        unsigned es, index;
        if (imm4 & 1) { es = 1; index = imm4 >> 1; }
        else if (imm4 & 2) { es = 2; index = imm4 >> 2; }
        else if (imm4 & 4) { es = 4; index = imm4 >> 3; }
        else return false;
        const uint64_t val = get_lane(d[vm & 31], es, index);
        uint64_t rep = 0;
        for (unsigned k = 0; k < 8 / es; ++k) set_lane(&rep, es, k, val);
        for (unsigned i = 0; i < regs; ++i) d[(vd + i) & 31] = rep;
        return true;
    }
    if ((insn & 0xFFB00C10u) == 0xF3B00800u) {             // VTBL / VTBX
        const unsigned len = ((insn >> 8) & 3) + 1;
        const bool extend = ((insn >> 6) & 1) != 0;        // VTBX keeps the old byte
        const uint64_t idx = d[vm & 31];
        uint64_t out = d[vd & 31];
        for (unsigned k = 0; k < 8; ++k) {
            const unsigned i = static_cast<unsigned>((idx >> (k * 8)) & 0xFF);
            if (i >= len * 8) {
                if (!extend) out &= ~(0xFFull << (k * 8));
                continue;
            }
            const uint64_t byte = (d[(vn + i / 8) & 31] >> ((i % 8) * 8)) & 0xFF;
            out = (out & ~(0xFFull << (k * 8))) | (byte << (k * 8));
        }
        d[vd & 31] = out;
        return true;
    }


    // ---- VEXT ------------------------------------------------------------------
    if ((insn & 0xFFB00010u) == 0xF2B00000u) {            // VEXT
        const unsigned imm = (insn >> 8) & 0xF;
        uint8_t buf[32];
        for (unsigned i = 0; i < regs; ++i) {
            for (unsigned k = 0; k < 8; ++k) buf[i * 8 + k] = static_cast<uint8_t>(d[(vn + i) & 31] >> (k * 8));
        }
        for (unsigned i = 0; i < regs; ++i) {
            for (unsigned k = 0; k < 8; ++k) buf[regs * 8 + i * 8 + k] = static_cast<uint8_t>(d[(vm + i) & 31] >> (k * 8));
        }
        for (unsigned i = 0; i < regs; ++i) {
            uint64_t out = 0;
            for (unsigned k = 0; k < 8; ++k)
                out |= static_cast<uint64_t>(buf[imm + i * 8 + k]) << (k * 8);
            d[(vd + i) & 31] = out;
        }
        return true;
    }

    // ---- three registers of different lengths (A7.4.4) -------------------------
    //
    // Three shapes share this encoding, and the difference is which operands are
    // 128-bit:
    //
    //   long   (L)  both operands narrow, result wide     VADDL, VMULL, VMLAL, ...
    //   wide   (W)  Vn already wide, Vm narrow            VADDW, VSUBW
    //   narrow (HN) both operands wide, result narrow     VADDHN, VSUBHN
    //
    // `size` always names the **narrow** element, so a "wide" element is twice that.
    // Reading it as the result's size instead halves or doubles every index, which
    // does not fault -- it reads the neighbouring lane.
    if ((insn & 0xFE800050u) == 0xF2800000u && ((insn >> 20) & 3) != 3) {
        const unsigned size = (insn >> 20) & 3;
        const unsigned esize = 1u << size;            // narrow element, bytes
        const unsigned wsize = esize * 2;
        const unsigned lanes = 8 / esize;             // narrow elements per d-register
        const unsigned wper = 8 / wsize;              // wide elements per d-register
        const unsigned op = (insn >> 8) & 0xF;

        auto ext = [&](uint64_t v) -> int64_t {
            return u ? static_cast<int64_t>(v) : sign_extend(v, esize);
        };
        auto extw = [&](uint64_t v) -> int64_t {
            return u ? static_cast<int64_t>(v) : sign_extend(v, wsize);
        };
        auto narrow_of = [&](unsigned base, unsigned k) { return get_lane(d[base & 31], esize, k); };
        auto wide_of = [&](unsigned base, unsigned k) {
            return get_lane(d[(base + k / wper) & 31], wsize, k % wper);
        };

        // The narrowing forms write one d-register; everything else writes a pair.
        const bool narrowing = op == 0x4 || op == 0x6;
        uint64_t out[2] = {0, 0};
        if (narrowing) out[0] = 0;
        else { out[0] = d[vd & 31]; out[1] = d[(vd + 1) & 31]; }

        for (unsigned k = 0; k < lanes; ++k) {
            uint64_t res;
            if (narrowing) {
                // VADDHN / VSUBHN: add or subtract the wide lanes and keep the *high*
                // half of each result. U selects the rounding variants.
                const int64_t x = extw(wide_of(vn, k)), y = extw(wide_of(vm, k));
                const int64_t full = op == 0x4 ? x + y : x - y;
                const unsigned shift = esize * 8;
                const int64_t rounded = u ? full + (1ll << (shift - 1)) : full;
                set_lane(&out[0], esize, k, static_cast<uint64_t>(rounded >> shift));
                continue;
            }
            const int64_t y = ext(narrow_of(vm, k));
            const uint64_t acc = wide_of(vd, k);
            switch (op) {
                case 0x0: res = static_cast<uint64_t>(ext(narrow_of(vn, k)) + y); break;   // VADDL
                case 0x1: res = static_cast<uint64_t>(extw(wide_of(vn, k)) + y); break;    // VADDW
                case 0x2: res = static_cast<uint64_t>(ext(narrow_of(vn, k)) - y); break;   // VSUBL
                case 0x3: res = static_cast<uint64_t>(extw(wide_of(vn, k)) - y); break;    // VSUBW
                case 0x5: case 0x7: {                                                      // VABAL / VABDL
                    const int64_t x = ext(narrow_of(vn, k));
                    const int64_t diff = x > y ? x - y : y - x;
                    res = static_cast<uint64_t>(op == 0x5 ? static_cast<int64_t>(acc) + diff : diff);
                    break;
                }
                case 0x8: res = acc + static_cast<uint64_t>(ext(narrow_of(vn, k)) * y); break;   // VMLAL
                case 0xA: res = acc - static_cast<uint64_t>(ext(narrow_of(vn, k)) * y); break;   // VMLSL
                case 0xC: res = static_cast<uint64_t>(ext(narrow_of(vn, k)) * y); break;         // VMULL
                // VQDMLAL/VQDMLSL/VQDMULL (the saturating doubling forms) and the
                // polynomial VMULL are refused rather than approximated.
                default: return false;
            }
            set_lane(&out[k / wper], wsize, k % wper, res);
        }
        d[vd & 31] = out[0];
        if (!narrowing) d[(vd + 1) & 31] = out[1];
        return true;
    }

    return false;
}

}  // namespace arm
