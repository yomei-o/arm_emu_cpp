// VFPv3/VFPv4: the scalar floating-point unit.
//
// This is not optional for the target. Raspberry Pi OS is `armhf` -- hard-float --
// which means every `double` crosses a function boundary in a VFP register and a
// libm that cannot do arithmetic is a libm that returns whatever was left in d0.
// The ABI is the reason: on `armel` a missing VFP would show up as a crash, and on
// `armhf` it shows up as a wrong number.
//
// The register file is one array of 32 doubles (`Cpu::d`), viewed three ways --
// s-registers are halves of it, NEON's q-registers are pairs of it. Compilers
// depend on the aliasing: a float moved through `s0` and then read as the low lane
// of `d0` is ordinary compiler output, not a trick.
//
// Register *numbering* is the part that bites. A single-precision register number
// is the four-bit field with the D bit appended at the **bottom**; a
// double-precision one is the same field with D at the **top**. Two different
// splices of the same bits, and mixing them up addresses a register that exists,
// so nothing faults -- the arithmetic is simply done on the wrong values.
#include "cpu.h"
#include <cmath>
#include <cstring>

namespace arm {

namespace {

// The two register-number splices, named so the call sites read as the manual does.
inline unsigned sreg(unsigned four, unsigned bit) { return (four << 1) | bit; }
inline unsigned dreg(unsigned four, unsigned bit) { return (bit << 4) | four; }

// VFPExpandImm: the 8-bit immediate VMOV can load, in the manual's own bit
// recipe -- sign, NOT(bit 6), bit 6 replicated eight times, then the low six bits
// of the immediate. It exists so a compiler can materialise 1.0, 0.5 or 2.0
// without a literal pool, and those are exactly the constants it produces.
//
// The double expansion is used for both precisions: every value it can encode is
// exactly representable as a float, so converting afterwards loses nothing.
inline double vfp_expand_imm(uint32_t imm8) {
    const uint64_t sign = (imm8 >> 7) & 1;
    const uint64_t b6 = (imm8 >> 6) & 1;
    const uint64_t bits = (sign << 63) | ((b6 ^ 1) << 62) |
                          ((b6 ? 0xFFull : 0ull) << 54) |
                          (static_cast<uint64_t>(imm8 & 0x3F) << 48);
    double out;
    std::memcpy(&out, &bits, 8);
    return out;
}

}  // namespace

bool Cpu::exec_vfp(uint32_t insn) {
    const unsigned cp = (insn >> 8) & 0xF;
    if (cp != 10 && cp != 11) return false;
    const bool dp = cp == 11;                    // the coprocessor number *is* the size

    // ---- VMOV between two core registers and a 64-bit VFP register (or two
    // single ones). This is how a `double` is passed on a soft-float call and how
    // a `long long` reaches NEON.
    if ((insn & 0x0FE00000u) == 0x0C400000u) {
        const bool to_arm = (insn >> 20) & 1;
        const unsigned rt = (insn >> 12) & 0xF, rt2 = (insn >> 16) & 0xF;
        if (dp) {
            const unsigned vm = dreg(insn & 0xF, (insn >> 5) & 1);
            if (to_arm) {
                r[rt] = static_cast<uint32_t>(d[vm]);
                r[rt2] = static_cast<uint32_t>(d[vm] >> 32);
            } else {
                d[vm] = (static_cast<uint64_t>(reg(rt2)) << 32) | reg(rt);
            }
        } else {
            const unsigned vm = sreg(insn & 0xF, (insn >> 5) & 1);
            if (to_arm) { r[rt] = s_get(vm); r[rt2] = s_get(vm + 1); }
            else { s_set(vm, reg(rt)); s_set(vm + 1, reg(rt2)); }
        }
        return true;
    }

    const unsigned top = (insn >> 24) & 0xF;

    // ---- VLDR / VSTR / VLDM / VSTM / VPUSH / VPOP -----------------------------
    if (top == 0xC || top == 0xD) {
        const bool p = (insn >> 24) & 1, u = (insn >> 23) & 1;
        const bool w = (insn >> 21) & 1, load = (insn >> 20) & 1;
        const unsigned rn = (insn >> 16) & 0xF;
        const unsigned dbit = (insn >> 22) & 1;
        const unsigned vd = dp ? dreg((insn >> 12) & 0xF, dbit) : sreg((insn >> 12) & 0xF, dbit);
        const uint32_t imm8 = insn & 0xFF;

        if (p && !w) {                            // VLDR / VSTR
            const uint32_t base = rn == 15 ? (pc_val() & ~3u) : reg(rn);
            const uint32_t addr = u ? base + imm8 * 4 : base - imm8 * 4;
            if (dp) {
                if (load) d[vd] = (static_cast<uint64_t>(mem_.read<uint32_t>(addr + 4)) << 32) |
                                  mem_.read<uint32_t>(addr);
                else {
                    mem_.write<uint32_t>(addr, static_cast<uint32_t>(d[vd]));
                    mem_.write<uint32_t>(addr + 4, static_cast<uint32_t>(d[vd] >> 32));
                }
            } else {
                if (load) s_set(vd, mem_.read<uint32_t>(addr));
                else mem_.write<uint32_t>(addr, s_get(vd));
            }
            return true;
        }
        // VLDM / VSTM. The register count is the immediate; for doubles it counts
        // words, so it is twice the number of registers.
        const unsigned count = dp ? imm8 / 2 : imm8;
        const uint32_t base = reg(rn);
        uint32_t addr = u ? base : base - count * (dp ? 8 : 4);
        const uint32_t end = u ? base + count * (dp ? 8 : 4) : base;
        for (unsigned i = 0; i < count; ++i) {
            if (dp) {
                const unsigned rd = (vd + i) & 31;
                if (load) d[rd] = (static_cast<uint64_t>(mem_.read<uint32_t>(addr + 4)) << 32) |
                                  mem_.read<uint32_t>(addr);
                else {
                    mem_.write<uint32_t>(addr, static_cast<uint32_t>(d[rd]));
                    mem_.write<uint32_t>(addr + 4, static_cast<uint32_t>(d[rd] >> 32));
                }
                addr += 8;
            } else {
                const unsigned rs = (vd + i) & 31;
                if (load) s_set(rs, mem_.read<uint32_t>(addr));
                else mem_.write<uint32_t>(addr, s_get(rs));
                addr += 4;
            }
        }
        if (w) r[rn] = u ? end : base - count * (dp ? 8 : 4);
        return true;
    }

    if (top != 0xE) return false;

    // ---- the MCR/MRC-shaped transfers (bit 4 set) ------------------------------
    if ((insn >> 4) & 1) {
        const unsigned opc1 = (insn >> 21) & 7;
        const unsigned rt = (insn >> 12) & 0xF;
        // VMSR / VMRS: the only way a floating-point comparison reaches a
        // conditional branch. `vmrs APSR_nzcv, fpscr` -- Rt == 15 -- copies the FP
        // flags into the integer ones, and without it every float comparison in
        // the guest branches on stale flags.
        if (((insn >> 20) & 0xF) == 0xF && ((insn >> 16) & 0xF) == 1 && cp == 10) {
            if (rt == 15) {
                n = (fpscr >> 31) & 1; z = (fpscr >> 30) & 1;
                c = (fpscr >> 29) & 1; v = (fpscr >> 28) & 1;
            } else r[rt] = fpscr;
            return true;
        }
        if (((insn >> 20) & 0xF) == 0xE && ((insn >> 16) & 0xF) == 1 && cp == 10) {
            fpscr = reg(rt);
            return true;
        }
        // VMOV between a core register and a single-precision register.
        if (cp == 10 && (opc1 & 6) == 0 && ((insn >> 8) & 0xF) == 10 &&
            ((insn >> 5) & 3) == 0 && (insn & 0xF) == 0) {
            const unsigned vn = sreg((insn >> 16) & 0xF, (insn >> 7) & 1);
            if ((insn >> 20) & 1) r[rt] = s_get(vn);
            else s_set(vn, reg(rt));
            return true;
        }
        // VMOV between a core register and a NEON scalar (an element of a
        // d-register), and VDUP from a core register. Both are coprocessor 11.
        if (cp == 11) {
            const bool to_arm = (insn >> 20) & 1;
            const unsigned opc1b = (insn >> 21) & 3;
            const unsigned opc2 = (insn >> 5) & 3;
            const unsigned vn = dreg((insn >> 16) & 0xF, (insn >> 7) & 1);
            if ((insn >> 23) & 1) {                        // VDUP (core register)
                const unsigned bq = (((insn >> 22) & 1) << 1) | ((insn >> 5) & 1);
                const unsigned esize = bq == 0 ? 4 : (bq == 1 ? 2 : 1);
                const bool quad = (insn >> 21) & 1;
                const uint32_t val = reg(rt);
                uint64_t lane = 0;
                for (unsigned i = 0; i < 8 / esize; ++i)
                    lane |= (static_cast<uint64_t>(val) & ((esize == 4) ? 0xFFFFFFFFull
                                                          : ((1ull << (esize * 8)) - 1)))
                            << (i * esize * 8);
                d[vn] = lane;
                if (quad) d[vn | 1] = lane;
                return true;
            }
            unsigned esize, index;
            if (opc1b & 2) { esize = 1; index = ((insn >> 21) & 1) * 4 + opc2; }
            else if (opc2 & 1) { esize = 2; index = ((insn >> 21) & 1) * 2 + (opc2 >> 1); }
            else { esize = 4; index = (insn >> 21) & 1; }
            if (to_arm) {
                const uint64_t lane = get_lane(d[vn], esize, index);
                const bool unsig = (insn >> 23) & 1;
                uint32_t val = static_cast<uint32_t>(lane);
                if (!unsig && esize == 1) val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(val)));
                if (!unsig && esize == 2) val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(val)));
                r[rt] = val;
            } else {
                set_lane(&d[vn], esize, index, reg(rt));
            }
            return true;
        }
        return false;
    }

    // ---- floating-point data-processing (the CDP shape) ------------------------
    const unsigned opc1 = (insn >> 20) & 0xF;
    const unsigned dbit = (insn >> 22) & 1;
    const unsigned vd = dp ? dreg((insn >> 12) & 0xF, dbit) : sreg((insn >> 12) & 0xF, dbit);
    const unsigned vn = dp ? dreg((insn >> 16) & 0xF, (insn >> 7) & 1)
                           : sreg((insn >> 16) & 0xF, (insn >> 7) & 1);
    const unsigned vm = dp ? dreg(insn & 0xF, (insn >> 5) & 1)
                           : sreg(insn & 0xF, (insn >> 5) & 1);
    const bool op = (insn >> 6) & 1;

    auto getn = [&]() { return dp ? d_get(vn) : static_cast<double>(s_getf(vn)); };
    auto getm = [&]() { return dp ? d_get(vm) : static_cast<double>(s_getf(vm)); };
    auto getd = [&]() { return dp ? d_get(vd) : static_cast<double>(s_getf(vd)); };
    // Single-precision results are rounded to float on the way in, not merely
    // stored: computing in double and truncating at the end gives a different
    // answer from a real VFP for anything that rounds twice.
    auto put = [&](double x) { if (dp) d_set(vd, x); else s_setf(vd, static_cast<float>(x)); };

    switch (opc1 & 0xB) {
        case 0x0: put(op ? getd() - getn() * getm() : getd() + getn() * getm()); return true;  // VMLA/VMLS
        case 0x1: put(op ? -getd() - getn() * getm() : -getd() + getn() * getm()); return true; // VNMLA/VNMLS
        case 0x2: put(op ? -(getn() * getm()) : getn() * getm()); return true;                  // VMUL/VNMUL
        case 0x3: put(op ? getn() - getm() : getn() + getm()); return true;                     // VADD/VSUB
        case 0x8: if (!op) { put(getn() / getm()); return true; } break;                        // VDIV
        case 0x9: put(op ? -(getd()) - getn() * getm() : -(getd()) + getn() * getm()); return true;  // VFNMS/VFNMA
        case 0xA: put(op ? getd() - getn() * getm() : getd() + getn() * getm()); return true;   // VFMA/VFMS
        case 0xB: break;
        default: break;
    }
    if ((opc1 & 0xB) != 0xB) return false;

    if (!op) {                                    // VMOV (immediate)
        const uint32_t imm8 = (((insn >> 16) & 0xF) << 4) | (insn & 0xF);
        put(vfp_expand_imm(imm8));
        return true;
    }

    const unsigned opc2 = (insn >> 16) & 0xF;
    const bool e = (insn >> 7) & 1;
    switch (opc2) {
        case 0x0:
            if (e) put(std::fabs(getm())); else put(getm());          // VABS / VMOV (register)
            return true;
        case 0x1:
            if (e) put(std::sqrt(getm())); else put(-getm());         // VSQRT / VNEG
            return true;
        case 0x4: case 0x5: {                                        // VCMP / VCMPE
            const double a = getd();
            const double b = (opc2 & 1) ? 0.0 : getm();
            uint32_t flags;
            if (std::isnan(a) || std::isnan(b)) flags = 0x3u << 28;   // C and V: unordered
            else if (a == b) flags = 0x6u << 28;                      // Z and C
            else if (a < b) flags = 0x8u << 28;                       // N
            else flags = 0x2u << 28;                                  // C
            fpscr = (fpscr & 0x0FFFFFFFu) | flags;
            return true;
        }
        case 0x7:                                                     // VCVT double <-> single
            if (dp) s_setf(sreg((insn >> 12) & 0xF, dbit), static_cast<float>(d_get(vm)));
            else d_set(dreg((insn >> 12) & 0xF, dbit), static_cast<double>(s_getf(vm)));
            return true;
        case 0x8: {                                                   // VCVT float from integer
            const unsigned sm = sreg(insn & 0xF, (insn >> 5) & 1);
            const uint32_t bits = s_get(sm);
            const double val = e ? static_cast<double>(static_cast<int32_t>(bits))
                                 : static_cast<double>(bits);
            put(val);
            return true;
        }
        case 0xC: case 0xD: {                                         // VCVT integer from float
            const double val = getm();
            const unsigned sd = sreg((insn >> 12) & 0xF, dbit);
            // The `R` bit picks round-to-nearest over round-to-zero; C converts
            // toward zero, which is what the common case (VCVT, not VCVTR) wants.
            const double rounded = e ? std::trunc(val) : std::nearbyint(val);
            if (opc2 & 1) {                                           // signed
                double clamped = rounded;
                if (!(clamped >= -2147483648.0)) clamped = -2147483648.0;
                if (!(clamped <= 2147483647.0)) clamped = 2147483647.0;
                s_set(sd, static_cast<uint32_t>(static_cast<int32_t>(clamped)));
            } else {                                                  // unsigned
                double clamped = rounded;
                if (!(clamped >= 0.0)) clamped = 0.0;
                if (!(clamped <= 4294967295.0)) clamped = 4294967295.0;
                s_set(sd, static_cast<uint32_t>(clamped));
            }
            return true;
        }
        case 0x2: case 0x3:                                           // VCVTB/VCVTT: half precision
        case 0xA: case 0xB: case 0xE: case 0xF:                       // VCVT fixed point
        default:
            return false;
    }
}

}  // namespace arm
