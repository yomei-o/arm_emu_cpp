// The one header that says which guest this copy of the translator serves.
//
// Everything else under jit/ is written against the names declared here, so the
// translation cache, the threaded and WebAssembly backends and the verifier are
// shared between emulators for different CPUs; see the AArch64 and x86 repositories.
//
// This is the 32-bit ARM one.  Guest registers are 32-bit values stored in 64-bit
// cells (arm::R32), the upper half always zero, so every translated operation is a
// 4-byte one.  The program counter's low bit stands for the Thumb state: a block is
// identified by `pc | thumb`.
#pragma once

#include <cstdint>
#include <string>

#include "../cpu.h"

namespace jit {

using GCpu = arm::Cpu;
using GMemory = arm::Memory;
using GObserver = arm::CodeObserver;
using GFault = arm::CpuError;

// Names of the environment variables that control the translator.
#define JIT_ENV(n) "ARMEMU_" n

constexpr int kCodeBits = 12;
// The page cache that the wasm and native tiers probe inline: 2^kTlbBits direct mapped
// entries of {page number, host address}, indexed by (addr >> kTlbPageBits) & mask.
constexpr int kTlbPageBits = 16;
constexpr int kTlbBits = 10;

// Byte offsets, from the start of the register file (r0), of the cells translated
// code uses.
struct GuestSlots {
    int32_t zero = 0;      // a cell that always reads zero
    int32_t rip = 0;       // the program counter
    int32_t tmp0 = 0;      // the first temporary
    int32_t ntmp = 0;      // how many temporaries
    int32_t fl_n = 0, fl_z = 0, fl_c = 0, fl_v = 0;   // the NZCV flags, a byte each
    int32_t thumb = 0;     // CPSR.T, a byte
};

// A guest's machine state as the verifier compares it.
struct GuestState {
    uint32_t r[16];
    uint32_t pc, ge, itstate, fpscr, tpidruro;
    bool thumb, n, z, c, v, q;
    uint64_t d[32];
    uint64_t icount;
    bool halted;
};

struct Glue {
    static uint8_t* env(GCpu& c) { return reinterpret_cast<uint8_t*>(c.r); }
    static void fill_slots(GuestSlots& s, GCpu& c);

    static uint64_t pc(const GCpu& c) { return static_cast<uint64_t>(static_cast<uint32_t>(c.pc)) | (c.thumb ? 1 : 0); }
    // Writes the address only: the instruction set is whatever it was.  (Switching it is
    // the interworking branch's job, A32Bx.)
    static void set_pc(GCpu& c, uint64_t v) { c.pc = static_cast<uint32_t>(v & ~1ull); }
    // A bounded run stops at an address whatever instruction set it is reached in.
    static bool is_stop(uint64_t pc, uint64_t stop) { return (pc & ~1ull) == stop; }
    static uint64_t icount(const GCpu& c) { return c.insns; }
    static void add_icount(GCpu& c, uint64_t n) { c.insns += n; }
    static bool halted(const GCpu& c) { return c.halted || c.stop_requested; }
    static bool in_hook(const GCpu&, uint64_t) { return false; }
    static void step(GCpu& c) { c.step(); }
    // Flags are always exact in this guest, so there is nothing to settle.
    static void settle(GCpu&) {}

    static uint8_t rd8(GMemory& m, uint64_t a) { return m.read<uint8_t>(static_cast<uint32_t>(a)); }
    static uint16_t rd16(GMemory& m, uint64_t a) { return m.read<uint16_t>(static_cast<uint32_t>(a)); }
    static uint32_t rd32(GMemory& m, uint64_t a) { return m.read<uint32_t>(static_cast<uint32_t>(a)); }
    static uint64_t rd64(GMemory& m, uint64_t a) { return m.read<uint64_t>(static_cast<uint32_t>(a)); }
    static void wr8(GMemory& m, uint64_t a, uint8_t v) { m.write<uint8_t>(static_cast<uint32_t>(a), v); }
    static void wr16(GMemory& m, uint64_t a, uint16_t v) { m.write<uint16_t>(static_cast<uint32_t>(a), v); }
    static void wr32(GMemory& m, uint64_t a, uint32_t v) { m.write<uint32_t>(static_cast<uint32_t>(a), v); }
    static void wr64(GMemory& m, uint64_t a, uint64_t v) { m.write<uint64_t>(static_cast<uint32_t>(a), v); }

    static const void* tlb_read_base(const GMemory& m) { return m.tlb_read_base(); }
    static const void* tlb_write_base(const GMemory& m) { return m.tlb_write_base(); }

    static void take(const GCpu& c, GuestState& s);
    static void put(GCpu& c, const GuestState& s);
    static std::string diff(const GuestState& a, const GuestState& b);
};

}  // namespace jit
