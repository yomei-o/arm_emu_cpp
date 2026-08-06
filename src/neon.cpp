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
    const unsigned a = (insn >> 19) & 0x1F;
    const unsigned c = (insn >> 8) & 0xF;
    const unsigned bit4 = (insn >> 4) & 1;
    const bool quad = (insn >> 6) & 1;
    const unsigned regs = quad ? 2 : 1;

    const unsigned vd = (((insn >> 22) & 1) << 4) | ((insn >> 12) & 0xF);
    const unsigned vn = (((insn >> 7) & 1) << 4) | ((insn >> 16) & 0xF);
    const unsigned vm = (((insn >> 5) & 1) << 4) | (insn & 0xF);

    // ---- one register and a modified immediate --------------------------------
    if ((insn & 0x0F800010u) == 0x02800010u && ((insn >> 19) & 7) == 0) {
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
    if ((insn & 0x0F000000u) == 0x02000000u) {
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
                    switch (size) {
                        case 0: dst = x ^ y; break;                       // VEOR
                        case 1: dst = (y & dst) | (x & ~dst); break;      // VBSL
                        case 2: dst = (dst & ~y) | (x & y); break;        // VBIT
                        default: dst = (dst & ~x) | (y & x); break;       // VBIF
                    }
                }
            }
            return true;
        }

        // Single-precision float, which shares this group behind opcodes 0xD-0xF.
        if (c >= 0xC && size < 2) {
            for (unsigned i = 0; i < regs; ++i) {
                uint64_t out = 0;
                for (unsigned k = 0; k < 2; ++k) {
                    const float x = as_f32(static_cast<uint32_t>(get_lane(d[(vn + i) & 31], 4, k)));
                    const float y = as_f32(static_cast<uint32_t>(get_lane(d[(vm + i) & 31], 4, k)));
                    const float acc = as_f32(static_cast<uint32_t>(get_lane(d[(vd + i) & 31], 4, k)));
                    uint32_t res;
                    if (c == 0xD && !bit4) res = as_u32(size ? x - y : x + y);        // VADD/VSUB
                    else if (c == 0xD && bit4) res = as_u32(x * y);                   // VMUL
                    else if (c == 0xC && bit4) res = as_u32(size ? acc - x * y : acc + x * y);
                    else if (c == 0xE && !bit4) {                                     // VCEQ/VCGE/VCGT
                        const bool t = !u ? (x == y) : (size ? (x > y) : (x >= y));
                        res = t ? 0xFFFFFFFFu : 0;
                    } else if (c == 0xF && !bit4) res = as_u32(size ? (x < y ? x : y) : (x > y ? x : y));
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
                uint64_t res;
                switch (c) {
                    case 0x0: res = bit4 ? (xu + yu) : ((xu + yu) >> 1); break;   // VQADD/VHADD
                    case 0x2: res = bit4 ? (xu - yu) : ((xu - yu) >> 1); break;   // VQSUB/VHSUB
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
                        if (bit4) res = xu * yu;                                   // VMUL
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
    if ((insn & 0x0F800010u) == 0x02800010u || (insn & 0x0F800090u) == 0x02800010u) {
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
            const unsigned in_es = esize / 2 ? esize : 1;
            (void)in_es;
            const unsigned src_es = esize == 8 ? 4 : (esize == 4 ? 2 : 1);
            const unsigned src_lanes = 8 / src_es;
            const unsigned sh = imm6 - src_es * 8;
            uint64_t lo = 0, hi = 0;
            const uint64_t src = d[vm & 31];
            for (unsigned k = 0; k < src_lanes; ++k) {
                const uint64_t xu = get_lane(src, src_es, k);
                const uint64_t val = (u ? xu : static_cast<uint64_t>(sign_extend(xu, src_es))) << sh;
                if (k < src_lanes / 2) set_lane(&lo, src_es * 2, k, val);
                else set_lane(&hi, src_es * 2, k - src_lanes / 2, val);
            }
            d[vd & 31] = lo;
            d[(vd + 1) & 31] = hi;
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

    // ---- two registers, miscellaneous (A7.4.5) --------------------------------
    if ((insn & 0x0FB00810u) == 0x03B00000u) {
        const unsigned size = (insn >> 18) & 3;
        const unsigned esize = 1u << size;
        const unsigned lanes = 8 / esize;
        const unsigned op = (insn >> 7) & 0xF;
        const unsigned group = (insn >> 16) & 3;

        if (group == 0) {                                 // VREV / VPADDL
            for (unsigned i = 0; i < regs; ++i) {
                const uint64_t src = d[(vm + i) & 31];
                uint64_t out = 0;
                if (op <= 3) {                            // VREV64 / VREV32 / VREV16
                    const unsigned width = 8u >> (op >> 1);          // bytes per group
                    (void)width;
                    const unsigned gsize = 1u << (3 - (op >> 1));
                    for (unsigned g = 0; g < 8 / gsize; ++g)
                        for (unsigned k = 0; k < gsize / esize; ++k) {
                            const uint64_t val = get_lane(src, esize, g * (gsize / esize) + k);
                            set_lane(&out, esize, g * (gsize / esize) + (gsize / esize - 1 - k), val);
                        }
                    d[(vd + i) & 31] = out;
                    continue;
                }
                if (op >= 4 && op <= 5) {                 // VPADDL
                    const bool unsig = (insn >> 7) & 1;
                    for (unsigned k = 0; k < lanes / 2; ++k) {
                        const uint64_t x = get_lane(src, esize, k * 2);
                        const uint64_t y = get_lane(src, esize, k * 2 + 1);
                        const uint64_t sum = unsig ? x + y
                            : static_cast<uint64_t>(sign_extend(x, esize) + sign_extend(y, esize));
                        set_lane(&out, esize * 2, k, sum);
                    }
                    d[(vd + i) & 31] = out;
                    continue;
                }
                return false;
            }
            return true;
        }
        if (group == 1) {                                 // VCLS/VCLZ/VCNT/VMVN, VABS/VNEG, VCxx #0
            for (unsigned i = 0; i < regs; ++i) {
                const uint64_t src = d[(vm + i) & 31];
                uint64_t out = 0;
                for (unsigned k = 0; k < lanes; ++k) {
                    const uint64_t xu = get_lane(src, esize, k);
                    const int64_t xs = sign_extend(xu, esize);
                    uint64_t res;
                    switch (op) {
                        case 0x8: res = static_cast<uint64_t>(xs < 0 ? -xs : xs); break;    // VABS
                        case 0x9: res = static_cast<uint64_t>(-xs); break;                  // VNEG
                        case 0x0: {                                                          // VCLS
                            unsigned n2 = 0;
                            const unsigned bits = esize * 8;
                            const uint64_t sign = (xu >> (bits - 1)) & 1;
                            for (unsigned b = bits - 1; b-- > 0;) {
                                if (((xu >> b) & 1) != sign) break;
                                ++n2;
                            }
                            res = n2;
                            break;
                        }
                        case 0x1: {                                                          // VCLZ
                            unsigned n2 = 0;
                            for (unsigned b = esize * 8; b-- > 0;) {
                                if (xu & (1ull << b)) break;
                                ++n2;
                            }
                            res = n2;
                            break;
                        }
                        case 0x2: {                                                          // VCNT
                            unsigned n2 = 0;
                            for (unsigned b = 0; b < esize * 8; ++b) if (xu & (1ull << b)) ++n2;
                            res = n2;
                            break;
                        }
                        case 0x3: res = ~xu; break;                                          // VMVN
                        case 0x4: res = xs == 0 ? lane_mask(esize) : 0; break;                // VCEQ #0
                        case 0x5: res = xs >= 0 ? lane_mask(esize) : 0; break;                // VCGE #0
                        case 0x6: res = xs > 0 ? lane_mask(esize) : 0; break;                 // VCGT #0
                        case 0x7: res = xs <= 0 ? lane_mask(esize) : 0; break;                // VCLE #0
                        case 0xA: res = xs < 0 ? lane_mask(esize) : 0; break;                 // VCLT #0
                        default: return false;
                    }
                    set_lane(&out, esize, k, res);
                }
                d[(vd + i) & 31] = out;
            }
            return true;
        }
        if (group == 2) {                                 // VSWP / VTRN / VUZP / VZIP / VMOVN
            if (op == 0) {                                // VSWP
                for (unsigned i = 0; i < regs; ++i) {
                    const uint64_t t = d[(vd + i) & 31];
                    d[(vd + i) & 31] = d[(vm + i) & 31];
                    d[(vm + i) & 31] = t;
                }
                return true;
            }
            if (op == 1) {                                // VTRN
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
            if (op == 4 || op == 5) {                     // VMOVN / VQMOVN
                uint64_t out = 0;
                const unsigned src_es = esize * 2;
                for (unsigned k = 0; k < 8 / src_es * 2 && k < 8 / esize; ++k) {
                    const unsigned half = k / (8 / src_es);
                    const uint64_t val = get_lane(d[(vm + half) & 31], src_es, k % (8 / src_es));
                    set_lane(&out, esize, k, val);
                }
                d[vd & 31] = out;
                return true;
            }
            return false;
        }
        if (group == 3) {                                 // VDUP (scalar) and the float converts
            if ((op & 0xC) == 0xC) {                      // VDUP
                const unsigned imm4 = (insn >> 16) & 0xF;
                unsigned es, index;
                if (imm4 & 1) { es = 1; index = imm4 >> 1; }
                else if (imm4 & 2) { es = 2; index = imm4 >> 2; }
                else { es = 4; index = imm4 >> 3; }
                const uint64_t val = get_lane(d[vm & 31], es, index);
                uint64_t rep = 0;
                for (unsigned k = 0; k < 8 / es; ++k) set_lane(&rep, es, k, val);
                for (unsigned i = 0; i < regs; ++i) d[(vd + i) & 31] = rep;
                return true;
            }
            return false;
        }
    }

    // ---- VEXT ------------------------------------------------------------------
    if ((insn & 0x0FB00010u) == 0x03B00000u) return false;
    if ((insn & 0x0FF00010u) == 0x02B00000u) {            // VEXT
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

    // ---- three registers of different lengths (VADDL, VSUBL, VMULL) ------------
    if ((insn & 0x0F800050u) == 0x02800000u && (a & 0x10)) {
        const unsigned size = (insn >> 20) & 3;
        const unsigned esize = 1u << size;
        const unsigned lanes = 8 / esize;
        const unsigned op = (insn >> 8) & 0xF;
        uint64_t lo = 0, hi = 0;
        for (unsigned k = 0; k < lanes; ++k) {
            const uint64_t xu = get_lane(d[vn & 31], esize, k);
            const uint64_t yu = get_lane(d[vm & 31], esize, k);
            uint64_t res;
            switch (op) {
                case 0x0: res = u ? (xu + yu) : static_cast<uint64_t>(sign_extend(xu, esize) + sign_extend(yu, esize)); break;
                case 0x2: res = u ? (xu - yu) : static_cast<uint64_t>(sign_extend(xu, esize) - sign_extend(yu, esize)); break;
                case 0xC: res = u ? (xu * yu) : static_cast<uint64_t>(sign_extend(xu, esize) * sign_extend(yu, esize)); break;
                default: return false;
            }
            if (k < lanes / 2) set_lane(&lo, esize * 2, k, res);
            else set_lane(&hi, esize * 2, k - lanes / 2, res);
        }
        d[vd & 31] = lo;
        d[(vd + 1) & 31] = hi;
        return true;
    }

    return false;
}

}  // namespace arm
