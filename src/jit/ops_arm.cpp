// The ARM guest's ops for the threaded backend: data-processing instructions that
// read or set the flags, and the branches that depend on them.  The arithmetic is
// Cpu::dp_op's, line for line, so that the two cannot disagree.
//
// N, Z, C and V are kept exact in the CPU's own bools after every flag-setting op -
// there is no deferred form - so the interpreter, hooks and the verifier always see
// the real thing.
#include "threaded.h"

namespace jit {

namespace {

using arm::Cpu;

// Cpu::cond_holds
inline bool cond_holds(const Cpu* cpu, unsigned cond) {
    bool r;
    switch (cond >> 1) {
        case 0: r = cpu->z; break;
        case 1: r = cpu->c; break;
        case 2: r = cpu->n; break;
        case 3: r = cpu->v; break;
        case 4: r = cpu->c && !cpu->z; break;
        case 5: r = cpu->n == cpu->v; break;
        case 6: r = (cpu->n == cpu->v) && !cpu->z; break;
        default: return true;
    }
    return (cond & 1) ? !r : r;
}

// One data-processing instruction.  `carry` is the shifter carry for the logical
// operations: negative leaves C alone.
template <bool FL>
inline uint32_t dp(Cpu* cpu, unsigned op, uint32_t a, uint32_t b, int64_t carry) {
    uint32_t res = 0;
    bool cout = carry < 0 ? cpu->c : carry != 0, vout = cpu->v;
    switch (op) {
        case Cpu::kAND: case Cpu::kTST: res = a & b; break;
        case Cpu::kEOR: case Cpu::kTEQ: res = a ^ b; break;
        case Cpu::kSUB: case Cpu::kCMP: { const arm::AddResult t = arm::add_with_carry(a, ~b, true);  res = t.result; cout = t.carry; vout = t.overflow; break; }
        case Cpu::kRSB: { const arm::AddResult t = arm::add_with_carry(~a, b, true);  res = t.result; cout = t.carry; vout = t.overflow; break; }
        case Cpu::kADD: case Cpu::kCMN: { const arm::AddResult t = arm::add_with_carry(a, b, false);  res = t.result; cout = t.carry; vout = t.overflow; break; }
        case Cpu::kADC: { const arm::AddResult t = arm::add_with_carry(a, b, cpu->c);      res = t.result; cout = t.carry; vout = t.overflow; break; }
        case Cpu::kSBC: { const arm::AddResult t = arm::add_with_carry(a, ~b, cpu->c);     res = t.result; cout = t.carry; vout = t.overflow; break; }
        case Cpu::kRSC: { const arm::AddResult t = arm::add_with_carry(~a, b, cpu->c);     res = t.result; cout = t.carry; vout = t.overflow; break; }
        case Cpu::kORR: res = a | b; break;
        case Cpu::kMOV: res = b; break;
        case Cpu::kBIC: res = a & ~b; break;
        case Cpu::kMVN: res = ~b; break;
        default:        res = a | ~b; break;
    }
    if (FL) { cpu->set_nz(res); cpu->c = cout; cpu->v = vout; }
    return res;
}

template <bool FL>
const Uop* h_dp(Ctx* c, const Uop* u) {
    S(u->d) = dp<FL>(c->cpu, u->cc, static_cast<uint32_t>(S(u->a)), static_cast<uint32_t>(S(u->b)), u->imm2);
    NEXT();
}
template <bool FL>
const Uop* h_dpi(Ctx* c, const Uop* u) {
    S(u->d) = dp<FL>(c->cpu, u->cc, static_cast<uint32_t>(S(u->a)), static_cast<uint32_t>(u->imm), u->imm2);
    NEXT();
}

const Uop* h_jcc(Ctx* c, const Uop* u) {
    c->cpu->pc = static_cast<uint32_t>(cond_holds(c->cpu, u->cc) ? u->imm : u->imm2);
    return nullptr;
}

// The instructions after a failed condition are skipped: leave, at the next one.
const Uop* h_exitif(Ctx* c, const Uop* u) {
    if (cond_holds(c->cpu, u->cc)) NEXT();
    c->cpu->pc = static_cast<uint32_t>(u->imm);
    c->partial = u->ni;
    return nullptr;
}

// A shift by a register: only the low byte of the amount counts, and an amount of
// 32 or more is not the same as the shift modulo 32.
const Uop* h_shift(Ctx* c, const Uop* u) {
    S(u->d) = arm::shift_c(static_cast<uint32_t>(S(u->a)), u->cc, static_cast<unsigned>(S(u->b) & 0xFF),
                           c->cpu->c).result;
    NEXT();
}

// LDM / STM as one op: the whole register list in a loop.
const Uop* h_ldm(Ctx* c, const Uop* u) {
    c->cur = u;
    uint32_t addr = static_cast<uint32_t>(S(u->a)) + static_cast<uint32_t>(u->c);
    const uint32_t list = static_cast<uint32_t>(u->imm);
    for (unsigned i = 0; i < 15; ++i) {
        if (!(list & (1u << i))) continue;
        S(i * 8) = Glue::rd32(*c->mem, addr);
        addr += 4;
    }
    if (list & 0x8000) S(u->d) = Glue::rd32(*c->mem, addr);
    NEXT();
}
const Uop* h_stm(Ctx* c, const Uop* u) {
    c->cur = u;
    uint32_t addr = static_cast<uint32_t>(S(u->a)) + static_cast<uint32_t>(u->c);
    const uint32_t list = static_cast<uint32_t>(u->imm);
    for (unsigned i = 0; i < 15; ++i) {
        if (!(list & (1u << i))) continue;
        Glue::wr32(*c->mem, addr, static_cast<uint32_t>(S(i * 8)));
        addr += 4;
    }
    CHK_FLUSH();
    NEXT();
}

const Uop* h_bx(Ctx* c, const Uop* u) {
    c->cpu->bx_write_pc(static_cast<uint32_t>(S(u->a)));
    return nullptr;
}

Fn select_arm(const Op& o) {
    const bool fl = (o.flags & kFlagEager) != 0;
    switch (o.kind) {
        case OpKind::A32Dp: return fl ? &h_dp<true> : &h_dp<false>;
        case OpKind::A32DpI: return fl ? &h_dpi<true> : &h_dpi<false>;
        case OpKind::A32Jcc: return h_jcc;
        case OpKind::A32ExitIf: return h_exitif;
        case OpKind::A32Bx: return h_bx;
        case OpKind::A32Shift: return h_shift;
        case OpKind::A32Ldm: return h_ldm;
        case OpKind::A32Stm: return h_stm;
        default: return nullptr;
    }
}

}  // namespace

Fn guest_select(const Op& o) { return select_arm(o); }

}  // namespace jit
