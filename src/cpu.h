// An ARMv7-A (A32 + T32) user-mode interpreter.
//
// This is the sibling of aarch64_emu_cpp pointed one architecture generation
// back, and the two are not the same job. A64 is a clean field layout: every
// instruction four bytes, one encoding space, no condition field, and a program
// counter that is not a register. A32 is none of those things, and the three
// differences below are where every silent bug in a 32-bit ARM interpreter comes
// from:
//
//  - **R15 is the program counter, and reading it is not reading a register.**
//    In ARM state R15 reads as the address of the current instruction **+ 8**; in
//    Thumb state, **+ 4** -- regardless of whether the Thumb instruction is two
//    bytes or four. That offset is the pipeline the architecture froze into its
//    ABI, and every PC-relative address in a real binary depends on it. It is
//    computed in exactly one place here (`pc_val()`), and no handler is allowed to
//    touch `r[15]` directly.
//
//  - **Writing R15 branches, and it may change instruction set.** On ARMv7 a
//    write to the PC from a data-processing instruction or a load is an
//    *interworking* branch: bit 0 selects Thumb. So `pop {pc}` returning into a
//    Thumb function is not a special case anywhere -- it falls out of
//    `bx_write_pc()`, which is the only way any handler writes the PC.
//
//  - **Almost every A32 instruction is conditional**, and in Thumb the same
//    conditionality is carried out of band, in ITSTATE, by the IT instruction.
//    A condition that is evaluated in the wrong place does not crash; it silently
//    executes or skips one instruction, and the program computes a wrong number.
//
// The decoder is laid out in the order the ARM ARM (DDI 0406C) organises the
// encoding space, so an instruction can be found by the section name the manual
// gives it.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include "memory.h"

namespace jit {
class Engine;
}

namespace arm {

struct CpuError {
    std::string what;
    uint32_t pc;
    uint32_t insn;
};

// The result of AddWithCarry, which is the primitive behind ADD, SUB, CMP, CMN,
// ADC, SBC and RSC -- all of them, including the subtractions, which are
// AddWithCarry(x, NOT(y), 1). Writing subtraction that way rather than as `x - y`
// is what makes the carry flag come out right: on ARM, carry after a subtract
// means "no borrow", which is the opposite of what a naive implementation
// produces and is invisible until a 64-bit subtraction built from SUBS/SBC
// disagrees with the host in the low bit.
struct AddResult {
    uint32_t result;
    bool carry, overflow;
};

inline AddResult add_with_carry(uint32_t x, uint32_t y, bool carry_in) {
    const uint64_t usum = static_cast<uint64_t>(x) + y + (carry_in ? 1 : 0);
    const int64_t ssum = static_cast<int64_t>(static_cast<int32_t>(x)) +
                         static_cast<int32_t>(y) + (carry_in ? 1 : 0);
    const uint32_t r = static_cast<uint32_t>(usum);
    return {r, usum != r, static_cast<int64_t>(static_cast<int32_t>(r)) != ssum};
}

// Shift types, in the encoding's own order: the two-bit `type` field of every
// shifted-register operand means exactly this.
enum ShiftType { kLSL = 0, kLSR = 1, kASR = 2, kROR = 3 };

struct ShiftResult {
    uint32_t result;
    bool carry;
};

// Shift_C: the barrel shifter with its carry-out. The special cases are the
// point -- LSR #0 means LSR #32, ASR #0 means ASR #32, ROR #0 means RRX -- and
// they are decoded by DecodeImmShift before this is called, so `amount` here is
// already the real amount and 0 means "no shift" for every type except RRX,
// which is spelled with its own type value.
inline ShiftResult shift_c(uint32_t value, unsigned type, unsigned amount, bool carry_in) {
    if (amount == 0) return {value, carry_in};
    switch (type) {
        case kLSL:
            if (amount > 32) return {0, false};
            if (amount == 32) return {0, (value & 1) != 0};
            return {value << amount, ((value >> (32 - amount)) & 1) != 0};
        case kLSR:
            if (amount > 32) return {0, false};
            if (amount == 32) return {0, (value >> 31) != 0};
            return {value >> amount, ((value >> (amount - 1)) & 1) != 0};
        case kASR: {
            const int32_t sv = static_cast<int32_t>(value);
            if (amount >= 32) return {static_cast<uint32_t>(sv >> 31), (value >> 31) != 0};
            return {static_cast<uint32_t>(sv >> amount), ((value >> (amount - 1)) & 1) != 0};
        }
        default: {                                   // ROR
            const unsigned a = amount & 31;
            if (a == 0) return {value, (value >> 31) != 0};
            const uint32_t r = (value >> a) | (value << (32 - a));
            return {r, (r >> 31) != 0};
        }
    }
}

// RRX: rotate right one place through the carry flag. Its own function because it
// is the one shift whose input includes a flag, so it cannot share the signature.
inline ShiftResult rrx_c(uint32_t value, bool carry_in) {
    return {(value >> 1) | (carry_in ? 0x80000000u : 0), (value & 1) != 0};
}

// DecodeImmShift: turn the encoded (type, imm5) pair into the amount the shifter
// actually applies. Returns kRRX (4) as the type when type==ROR and imm5==0.
enum { kRRX = 4 };
inline void decode_imm_shift(unsigned type, unsigned imm5, unsigned* out_type, unsigned* out_amt) {
    switch (type) {
        case kLSL: *out_type = kLSL; *out_amt = imm5; break;
        case kLSR: *out_type = kLSR; *out_amt = imm5 ? imm5 : 32; break;
        case kASR: *out_type = kASR; *out_amt = imm5 ? imm5 : 32; break;
        default:
            if (imm5) { *out_type = kROR; *out_amt = imm5; }
            else { *out_type = kRRX; *out_amt = 1; }
            break;
    }
}

// ARMExpandImm_C: the 12-bit modified immediate. An 8-bit value rotated right by
// twice a 4-bit field -- which is why an ARM immediate can be 0xFF000000 but not
// 0x00FFFF00, and why a compiler splits large constants into two instructions.
inline ShiftResult expand_imm_c(uint32_t imm12, bool carry_in) {
    const uint32_t imm8 = imm12 & 0xFF;
    const unsigned rot = ((imm12 >> 8) & 0xF) * 2;
    if (rot == 0) return {imm8, carry_in};
    const uint32_t r = (imm8 >> rot) | (imm8 << (32 - rot));
    return {r, (r >> 31) != 0};
}

// ThumbExpandImm_C: the same idea with a different, denser encoding -- Thumb-2
// spends the top bits on the four repeating-byte patterns (0x00XY00XY and
// friends) that turn up in masks, and only then falls back to a rotated 8-bit
// value with an implicit leading 1.
inline ShiftResult thumb_expand_imm_c(uint32_t imm12, bool carry_in) {
    if ((imm12 & 0xC00) == 0) {
        const uint32_t b = imm12 & 0xFF;
        switch ((imm12 >> 8) & 3) {
            case 0: return {b, carry_in};
            case 1: return {(b << 16) | b, carry_in};
            case 2: return {(b << 24) | (b << 8), carry_in};
            default: return {(b << 24) | (b << 16) | (b << 8) | b, carry_in};
        }
    }
    const uint32_t val = 0x80u | (imm12 & 0x7F);
    const unsigned rot = (imm12 >> 7) & 0x1F;
    const uint32_t r = (val >> rot) | (val << (32 - rot));
    return {r, (r >> 31) != 0};
}

// A 32-bit register held in a 64-bit cell.  The translator (src/jit) addresses guest
// registers as 8-byte slots, so the register file is laid out that way; everything
// else sees an ordinary uint32_t - reads convert to one, writes narrow to one - which
// keeps the interpreter's arithmetic exactly what it was.  The upper half is always
// zero.
struct R32 {
    uint64_t v = 0;
    operator uint32_t() const { return static_cast<uint32_t>(v); }
    R32& operator=(uint32_t x) { v = x; return *this; }
    R32& operator+=(uint32_t x) { v = static_cast<uint32_t>(v + x); return *this; }
};

class Cpu {
public:
    explicit Cpu(Memory& mem) : mem_(mem) {}
    ~Cpu();
    Cpu(const Cpu&) = delete;
    Cpu& operator=(const Cpu&) = delete;

    // r[15] exists as storage but is never the truth: the PC lives in `pc`, and
    // every read of R15 goes through pc_val(). Keeping the slot means LDM/STM
    // register lists can be walked without a special case at index 15.
    R32 r[16];
    R32 pc;                          // address of the *next* instruction to fetch
    bool n = false, z = false, c = false, v = false, q = false;
    uint32_t ge = 0;                 // CPSR GE[3:0], written by the parallel adds
    bool thumb = false;              // CPSR.T
    // ITSTATE, in the CPSR's own packed form: bits 7..5 are the base condition,
    // bits 4..0 the mask. Kept packed rather than as a queue because that is what
    // a signal frame stores and what MSR/CPSR reads have to show.
    uint32_t itstate = 0;

    // VFP/NEON: one register file, three views. d[0..31] is the storage; s[i] is
    // half of d[i/2]; q[i] is the pair d[2i],d[2i+1]. Representing it as the d
    // array rather than as separate banks is what makes the aliasing automatic --
    // and the aliasing is load-bearing, because compilers move a float through
    // s-registers and then reinterpret the pair as a NEON vector.
    uint64_t d[32] = {0};
    uint32_t fpscr = 0;

    // The user-mode read-only thread pointer, CP15 c13,c0,3 (TPIDRURO). glibc and
    // musl on ARMv7 read it with `mrc p15, 0, rX, c13, c0, 3`; older binaries call
    // the kernel helper at 0xFFFF0FE0 instead, and both end up here.
    uint32_t tpidruro = 0;

    bool halted = false;
    int exit_code = 0;
    uint64_t insns = 0;
    uint64_t max_insns = 0;          // 0 = unlimited; a runaway guard for tests
    // Print the PC every N instructions. A guest that stops making progress looks
    // exactly like one doing a lot of work; sampling tells them apart in seconds.
    uint64_t sample_every = 0, sample_left = 0;
    uint32_t pc_watch = 0;
    std::function<void()> on_pc_watch;

    // ---- register access -------------------------------------------------------
    //
    // The whole R15 problem, in four functions. Nothing else in the interpreter
    // reads or writes r[15].

    // The value R15 reads as: the current instruction's address plus the pipeline
    // offset the architecture defines. Not `pc`, which has already advanced.
    uint32_t pc_val() const { return cur_pc_ + (thumb ? 4 : 8); }
    uint32_t reg(unsigned i) const { return i == 15 ? pc_val() : r[i]; }
    // A plain branch: no interworking, the instruction set does not change.
    void branch_write_pc(uint32_t val) { pc = thumb ? (val & ~1u) : (val & ~3u); }
    // An interworking branch. Bit 0 selects Thumb, and this is what a write to R15
    // from ALU, LDR, LDM and BX all do on ARMv7 -- so `pop {pc}` into a Thumb
    // function needs no special case anywhere.
    void bx_write_pc(uint32_t val) {
        if (val & 1) { thumb = true; pc = val & ~1u; }
        else { thumb = false; pc = val & ~3u; }
    }
    void set_reg(unsigned i, uint32_t val) {
        if (i == 15) bx_write_pc(val); else r[i] = val;
    }
    // ALUWritePC: what a *data-processing* write to R15 does, and it is not the same
    // as a load's. On ARMv7 it interworks **only in ARM state**; in Thumb state it is
    // a plain branch that stays in Thumb.
    //
    // The difference is load-bearing. A compiler builds a jump table out of
    // `adr r2, table; add r3, r2, n, lsl #4; mov pc, r3` -- and every entry in that
    // table is at an even address, because Thumb instructions are two-byte aligned
    // and nothing marks them. Interworking there reads bit 0 as "ARM", and the
    // machine starts decoding a table of Thumb instructions four bytes at a time.
    // musl's `__aeabi_uidiv` is exactly this shape.
    void alu_write_pc(uint32_t val) {
        if (thumb) branch_write_pc(val); else bx_write_pc(val);
    }
    void set_reg_alu(unsigned i, uint32_t val) {
        if (i == 15) alu_write_pc(val); else r[i] = val;
    }
    // For the handful of encodings whose PC write is defined as a plain branch
    // (B, BL to the same state, and Thumb's ADD PC in some forms).
    void set_reg_branch(unsigned i, uint32_t val) {
        if (i == 15) branch_write_pc(val); else r[i] = val;
    }

    uint32_t sp() const { return r[13]; }
    void set_sp(uint32_t v) { r[13] = v; }
    uint32_t lr() const { return r[14]; }
    void set_lr(uint32_t v) { r[14] = v; }

    // ---- VFP/NEON register views ----------------------------------------------
    uint32_t s_get(unsigned i) const {
        return static_cast<uint32_t>(d[i >> 1] >> ((i & 1) * 32));
    }
    void s_set(unsigned i, uint32_t v) {
        const unsigned sh = (i & 1) * 32;
        d[i >> 1] = (d[i >> 1] & ~(0xFFFFFFFFull << sh)) | (static_cast<uint64_t>(v) << sh);
    }
    double d_get(unsigned i) const { double x; std::memcpy(&x, &d[i], 8); return x; }
    void d_set(unsigned i, double x) { std::memcpy(&d[i], &x, 8); }
    float s_getf(unsigned i) const { const uint32_t b = s_get(i); float x; std::memcpy(&x, &b, 4); return x; }
    void s_setf(unsigned i, float x) { uint32_t b; std::memcpy(&b, &x, 4); s_set(i, b); }

    // A lane of a d-register, by element size in bytes. Shared by the NEON file
    // and the element-transfer instructions.
    static uint64_t get_lane(uint64_t reg, unsigned esize, unsigned idx) {
        const unsigned sh = idx * esize * 8;
        const uint64_t m = esize == 8 ? ~0ull : ((1ull << (esize * 8)) - 1);
        return (reg >> sh) & m;
    }
    static void set_lane(uint64_t* reg, unsigned esize, unsigned idx, uint64_t val) {
        const unsigned sh = idx * esize * 8;
        const uint64_t m = esize == 8 ? ~0ull : ((1ull << (esize * 8)) - 1);
        *reg = (*reg & ~(m << sh)) | ((val & m) << sh);
    }

    // ---- the OS interface ------------------------------------------------------
    // SVC handler: returns false to stop the machine.
    std::function<bool(uint32_t imm)> on_svc;
    // An instruction the decoder does not implement. Returning true means the host
    // dealt with it -- in practice by delivering SIGILL, which is the
    // architecturally correct answer for a CPU that does not have the instruction.
    // Returning false lets the emulator stop and print the encoding, which is the
    // default and is deliberately loud: an instruction that silently does nothing
    // leaves a stale register and fails somewhere unrelated, hours later.
    std::function<bool(uint32_t insn, uint32_t pc)> on_undefined;

    // ---- flags -----------------------------------------------------------------
    bool cond_holds(unsigned cond) const;
    void set_nz(uint32_t result) { n = (result >> 31) != 0; z = result == 0; }
    uint32_t cpsr() const {
        return (uint32_t(n) << 31) | (uint32_t(z) << 30) | (uint32_t(c) << 29) |
               (uint32_t(v) << 28) | (uint32_t(q) << 27) |
               ((itstate & 0x3) << 25) | (uint32_t(thumb) << 5) |
               (ge << 16) | (((itstate >> 2) & 0x3F) << 10) | 0x10;
    }
    void set_cpsr(uint32_t val) {
        n = (val >> 31) & 1; z = (val >> 30) & 1; c = (val >> 29) & 1;
        v = (val >> 28) & 1; q = (val >> 27) & 1;
        ge = (val >> 16) & 0xF;
        thumb = (val >> 5) & 1;
        itstate = ((val >> 25) & 0x3) | (((val >> 10) & 0x3F) << 2);
    }

    // ---- context ---------------------------------------------------------------
    // What a signal frame stores: Linux's `struct sigcontext` for ARM, in its own
    // order, is built from exactly this.
    struct Regs { uint32_t r[16], cpsr; };
    Regs save_regs() const {
        Regs s{};
        for (int i = 0; i < 15; ++i) s.r[i] = r[i];
        s.r[15] = cur_pc_;
        s.cpsr = cpsr();
        return s;
    }

    // Everything a thread owns. Wider than `Regs`: a context switch has to preserve
    // the VFP bank and the thread pointer too, and a switch that drops them corrupts
    // whichever thread happens to be holding a value there.
    struct Context {
        uint32_t r[16] = {0};
        uint32_t pc = 0, cpsr = 0, tpidruro = 0, fpscr = 0;
        uint64_t d[32] = {0};
    };
    Context save_context() const {
        Context ctx;
        for (int i = 0; i < 16; ++i) ctx.r[i] = r[i];
        for (int i = 0; i < 32; ++i) ctx.d[i] = d[i];
        ctx.pc = pc; ctx.cpsr = cpsr(); ctx.tpidruro = tpidruro; ctx.fpscr = fpscr;
        return ctx;
    }
    void load_context(const Context& ctx) {
        for (int i = 0; i < 16; ++i) r[i] = ctx.r[i];
        for (int i = 0; i < 32; ++i) d[i] = ctx.d[i];
        pc = ctx.pc; set_cpsr(ctx.cpsr); tpidruro = ctx.tpidruro; fpscr = ctx.fpscr;
        // The monitor is not part of the context -- it belongs to the CPU, and a
        // switch is precisely the event a real one loses it to. Keeping it across a
        // switch would let a thread's STREX succeed against a LDREX another thread
        // made, which is the one thing the instruction exists to prevent.
        clear_exclusive();
    }
    void clear_exclusive() { excl_valid_ = false; }
    // LDREX arms the monitor; STREX consumes it. Public because the kernel helper
    // page's __kuser_cmpxchg is the same operation reached a different way.
    void set_exclusive(uint32_t addr) { excl_valid_ = true; excl_addr_ = addr; }
    bool take_exclusive(uint32_t addr) {
        const bool ok = excl_valid_ && excl_addr_ == addr;
        excl_valid_ = false;
        return ok;
    }

    // ---- the run loop ----------------------------------------------------------
    // Preemption, for a multi-threaded guest. Zero -- the default -- costs one
    // predictable branch per instruction and never fires; the scheduler turns it on
    // only once a second thread exists. Without it a guest that spins on a lock
    // instead of blocking on a futex would never give the CPU back.
    uint64_t preempt_every = 0, preempt_left = 0;
    std::function<void()> on_preempt;
    // Set by a vfork child reaching execve or _exit: the run loop stops so the host
    // can restore the parent.
    bool stop_requested = false;
    std::function<void()> on_stop_requested;

    // ---- dynamic translation (src/jit) -------------------------------------------
    //
    // Temporaries for translated blocks.  They sit in the CPU so that a block addresses
    // them the way it addresses registers: as an offset from r[0] (r[] is 8-byte cells
    // for exactly that reason).  Slot 0 is never written and always reads zero.
    uint64_t jit_tmp[48] = {};
    // Starts / stops the translator.  Off, nothing changes.
    void enable_jit(bool on);
    bool jit_on() const { return jit_ != nullptr; }
    // Whether translated execution may be used right now: not in the middle of an IT
    // block, and not while sampling, watching a pc or an address, or in strict mode.
    bool jit_usable() const;
    // Runs translated blocks until `deadline` instructions have executed, pc reaches
    // `stop_pc`, the guest halts, `*yield` becomes true or pc is somewhere the
    // translator will not go.  False: nothing ran, step().
    bool jit_run(uint64_t deadline, uint64_t stop_pc, const bool* yield);

    void run() { run_until(0, false); }
    void run_until(uint32_t stop_pc, bool bounded = true) {
        if (!jit_checked_) maybe_enable_jit();
        for (;;) {
            while (!halted && !stop_requested && !(bounded && pc == stop_pc)) {
                if (jit_ && jit_usable()) {
                    const uint64_t before = insns;
                    // Blocks run many instructions between looks at the counters, so the
                    // preemption count is spent in bulk afterwards.
                    const uint64_t deadline = preempt_left ? insns + preempt_left : ~0ull;
                    if (jit_run(deadline, bounded ? stop_pc : ~0ull, &stop_requested)) {
                        if (preempt_left) {
                            const uint64_t ran = insns - before;
                            if (ran >= preempt_left) {
                                preempt_left = preempt_every;
                                if (on_preempt) on_preempt();
                            } else {
                                preempt_left -= ran;
                            }
                        }
                        continue;
                    }
                }
                step();
                if (preempt_left && --preempt_left == 0) {
                    preempt_left = preempt_every;
                    if (on_preempt) on_preempt();
                }
            }
            if (stop_requested && on_stop_requested) {
                stop_requested = false;
                on_stop_requested();
                continue;
            }
            return;
        }
    }
    void step();

    Memory& mem() { return mem_; }
    uint32_t cur_pc() const { return cur_pc_; }

private:
    void maybe_enable_jit();
    jit::Engine* jit_ = nullptr;
    bool jit_checked_ = false;
    friend class jit::Engine;
public:

    [[noreturn]] void fail(const std::string& why, uint32_t insn) const;
    // An instruction the decoder does not implement: offer it to the host as SIGILL
    // and, if nothing takes it, stop and print the encoding.
    void undefined(uint32_t insn);

    // ---- shared execution helpers ---------------------------------------------
    // A32 and T32 encode the same operations differently but execute them
    // identically, so the *execution* lives here once and each decoder calls it.
    // Splitting them the other way -- a shared decoder with two encodings -- was
    // tried in the x86 sibling and produces a function with a flag argument for
    // every difference.

    // The data-processing opcodes, in the A32 encoding's own numbering.
    enum DpOp {
        kAND = 0, kEOR = 1, kSUB = 2, kRSB = 3, kADD = 4, kADC = 5, kSBC = 6,
        kRSC = 7, kTST = 8, kTEQ = 9, kCMP = 10, kCMN = 11, kORR = 12, kMOV = 13,
        kBIC = 14, kMVN = 15,
        // Thumb-2 adds two that A32 spells as BIC/MVN with a swap.
        kORN = 16
    };
    // One data-processing instruction. `shifter_carry` is the carry the operand's
    // shift produced, which the logical operations write to C and the arithmetic
    // ones ignore -- getting that backwards is silent and shows up as a wrong
    // carry into an unrelated ADC four instructions later.
    void dp_op(unsigned op, unsigned rd, uint32_t rn_val, uint32_t op2,
               bool setflags, bool shifter_carry);

    // Load/store, once, for every addressing mode both instruction sets produce.
    // `size` is 1/2/4/8 bytes; `sign` sign-extends a 1- or 2-byte load.
    void do_load(unsigned rt, uint32_t addr, unsigned size, bool sign, bool is_signed_ldrd = false);
    void do_store(unsigned rt, uint32_t addr, unsigned size);

    // The parallel (SIMD-within-a-register) add and subtract group, shared by A32's
    // A5.4.1/A5.4.2 and Thumb's A6.3.13. Six operations, three variants each, both
    // signednesses -- and only the plain variant writes the GE flags, which `SEL`
    // then reads. That pairing is the whole point of the group: glibc's ARMv6
    // `strlen` finds a zero byte with `uqsub8` and picks the lane with `sel`.
    enum ParOp { kPAdd16, kPAsx, kPSax, kPSub16, kPAdd8, kPSub8 };
    enum ParKind { kPPlain, kPSat, kPHalve };
    void parallel_add_sub(unsigned op, unsigned kind, bool unsig,
                          unsigned rd, uint32_t a, uint32_t b);

    // LDM/STM over a register list, shared by A32's four addressing modes, Thumb's
    // PUSH/POP and Thumb-2's LDMIA/STMDB.
    void do_ldm(unsigned rn, uint32_t list, bool inc, bool before, bool wback, bool user_regs);
    void do_stm(unsigned rn, uint32_t list, bool inc, bool before, bool wback);

    // Decoder entry points, one per instruction set. Both are in their own file.
    void exec_a32(uint32_t insn);
    void exec_thumb();
    // VFP and Advanced SIMD, reached from both. Returns false when the encoding is
    // not one this file handles, so the caller can report it rather than assume.
    bool exec_vfp(uint32_t insn);
    bool exec_neon(uint32_t insn);
    bool exec_neon_ldst(uint32_t insn);
    // The coprocessor register transfers a userland guest actually makes: CP15's
    // thread-pointer registers and the cache-line size, and CP10/11's FPSCR.
    bool exec_cp15(uint32_t insn);

    // ITSTATE advance, called after every Thumb instruction. Public because the
    // A32 decoder must clear it on a branch into Thumb state.
    void it_advance() {
        if ((itstate & 7) == 0) itstate = 0;
        else itstate = (itstate & 0xE0) | ((itstate << 1) & 0x1F);
    }
    bool in_it_block() const { return (itstate & 0xF) != 0; }
    unsigned it_cond() const { return (itstate >> 4) & 0xF; }
    // Whether the instruction *currently executing* is inside an IT block.
    //
    // Not the same question as `in_it_block()`, and the difference is a real bug:
    // ITSTATE is advanced before the instruction runs, so by the time a handler
    // asks, the last instruction of a block already reads as outside one. Sixteen-bit
    // Thumb decides whether to write the flags from exactly this -- `movs` and `mov`
    // are the same encoding, separated only by "am I in an IT block" -- so the last
    // instruction of every IT block was setting the flags. `ite ls / movls / movhi`
    // then left Z from the MOV, and the `bhi` that follows read it: a scan loop in
    // musl's strtod that never terminates.
    bool in_it = false;

private:
    Memory& mem_;
    // The address of the instruction being executed. `pc` has already advanced by
    // the time a handler runs, and every PC-relative form needs the original.
    uint32_t cur_pc_ = 0;

    bool excl_valid_ = false;
    uint32_t excl_addr_ = 0;

    // A32 instruction groups, in the ARM ARM's order (A5.1: bits 27..25).
    void a32_dp_reg(uint32_t insn);
    void a32_dp_imm(uint32_t insn);
    void a32_load_store(uint32_t insn);
    void a32_media(uint32_t insn);
    void a32_block(uint32_t insn);
    void a32_branch(uint32_t insn);
    void a32_coproc(uint32_t insn);
    void a32_misc(uint32_t insn);
    void a32_multiply(uint32_t insn);
    void a32_extra_load_store(uint32_t insn);
    void a32_sync(uint32_t insn);
    void a32_unconditional(uint32_t insn);

    // T32, split by size. `exec_thumb` fetches and dispatches.
    void t16(uint16_t insn);
    void t32(uint32_t insn);
    void t32_dp_modified_imm(uint32_t insn);
    void t32_dp_plain_imm(uint32_t insn);
    void t32_branch_misc(uint32_t insn);
    void t32_load_store_multiple(uint32_t insn);
    void t32_load_store_dual(uint32_t insn);
    void t32_dp_shifted_reg(uint32_t insn);
    void t32_dp_register(uint32_t insn);
    void t32_load_word(uint32_t insn);
    void t32_load_byte(uint32_t insn);
    void t32_load_half(uint32_t insn);
    void t32_store_single(uint32_t insn);
    void t32_multiply(uint32_t insn);
    void t32_long_multiply(uint32_t insn);
};

// The kernel helper page ARM Linux maps at the top of every process. Old binaries
// -- and Raspberry Pi's own toolchain output for ARMv6 -- call into it for atomics
// and the thread pointer rather than using the instructions, because the
// instructions are not on every ARM the kernel supports.
constexpr uint32_t kKuserBase = 0xFFFF0000u;
constexpr uint32_t kKuserMemoryBarrier = 0xFFFF0FA0u;
constexpr uint32_t kKuserCmpxchg = 0xFFFF0FC0u;
constexpr uint32_t kKuserGetTls = 0xFFFF0FE0u;
constexpr uint32_t kKuserCmpxchg64 = 0xFFFF0F60u;

}  // namespace arm
