// The ARM guest's side of the translator: how the engine finds the registers, and
// the Cpu entry points that hand control to it.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "engine.h"

namespace jit {

void Glue::fill_slots(GuestSlots& s, GCpu& cpu) {
    auto off = [&](const void* p) {
        return static_cast<int32_t>(reinterpret_cast<const uint8_t*>(p) -
                                    reinterpret_cast<const uint8_t*>(cpu.r));
    };
    s.zero = off(&cpu.jit_tmp[0]);
    s.tmp0 = off(&cpu.jit_tmp[1]);
    s.ntmp = static_cast<int32_t>(sizeof(cpu.jit_tmp) / sizeof(cpu.jit_tmp[0])) - 1;
    s.rip = off(&cpu.pc);
}

void Glue::take(const GCpu& c, GuestState& s) {
    for (int i = 0; i < 16; ++i) s.r[i] = c.r[i];
    s.pc = c.pc;
    s.ge = c.ge;
    s.itstate = c.itstate;
    s.fpscr = c.fpscr;
    s.tpidruro = c.tpidruro;
    s.thumb = c.thumb;
    s.n = c.n; s.z = c.z; s.c = c.c; s.v = c.v; s.q = c.q;
    std::memcpy(s.d, c.d, sizeof s.d);
    s.icount = c.insns;
    s.halted = c.halted;
}

void Glue::put(GCpu& c, const GuestState& s) {
    for (int i = 0; i < 16; ++i) c.r[i] = s.r[i];
    c.pc = s.pc;
    c.ge = s.ge;
    c.itstate = s.itstate;
    c.fpscr = s.fpscr;
    c.tpidruro = s.tpidruro;
    c.thumb = s.thumb;
    c.n = s.n; c.z = s.z; c.c = s.c; c.v = s.v; c.q = s.q;
    std::memcpy(c.d, s.d, sizeof c.d);
    c.insns = s.icount;
    c.halted = s.halted;
}

std::string Glue::diff(const GuestState& s1, const GuestState& s2) {
    std::string diff;
    char buf[256];
    auto note = [&](const char* what, uint64_t a, uint64_t c) {
        std::snprintf(buf, sizeof buf, "  %s: jit %08llX  interp %08llX\n", what,
                      (unsigned long long)a, (unsigned long long)c);
        diff += buf;
    };
    for (int i = 0; i < 16; ++i) {
        if (s1.r[i] != s2.r[i]) {
            char name[8];
            std::snprintf(name, sizeof name, "r%d", i);
            note(name, s1.r[i], s2.r[i]);
        }
    }
    if (s1.pc != s2.pc) note("pc", s1.pc, s2.pc);
    if (s1.thumb != s2.thumb) note("thumb", s1.thumb, s2.thumb);
    if (s1.itstate != s2.itstate) note("itstate", s1.itstate, s2.itstate);
    if (s1.ge != s2.ge) note("ge", s1.ge, s2.ge);
    if (s1.tpidruro != s2.tpidruro) note("tpidruro", s1.tpidruro, s2.tpidruro);
    if (s1.n != s2.n || s1.z != s2.z || s1.c != s2.c || s1.v != s2.v || s1.q != s2.q)
        note("nzcvq", (s1.n << 4) | (s1.z << 3) | (s1.c << 2) | (s1.v << 1) | s1.q,
             (s2.n << 4) | (s2.z << 3) | (s2.c << 2) | (s2.v << 1) | s2.q);
    if (s1.icount != s2.icount) note("insns", s1.icount, s2.icount);
    if (s1.halted != s2.halted) note("halted", s1.halted, s2.halted);
    if (std::memcmp(s1.d, s2.d, sizeof s1.d) != 0) diff += "  vfp/neon registers differ\n";
    if (s1.fpscr != s2.fpscr) note("fpscr", s1.fpscr, s2.fpscr);
    return diff;
}

}  // namespace jit

namespace arm {

Cpu::~Cpu() { delete jit_; }

void Cpu::enable_jit(bool on) {
    if (on == (jit_ != nullptr)) return;
    if (on) {
        jit_ = new jit::Engine(*this, mem_);
    } else {
        delete jit_;
        jit_ = nullptr;
    }
}

// On unless ARMEMU_JIT=0.  Decided on the first run, so tools that build a Cpu and
// never run it do not pay for an engine.
void Cpu::maybe_enable_jit() {
    jit_checked_ = true;
    const char* v = std::getenv("ARMEMU_JIT");
    if (v && v[0] == '0') return;
    enable_jit(true);
}

// Thumb code is not translated yet; it runs in the interpreter.
bool Cpu::jit_usable() const {
    return jit_ && !thumb && !sample_every && !pc_watch && !mem_.strict && !mem_.watch_hi;
}

bool Cpu::jit_run(uint64_t deadline, uint64_t stop_pc, const bool* yield) {
    // The instruction limit is a runaway guard that step() enforces; stop the blocks
    // at it so that step() still gets to.
    if (max_insns && deadline > max_insns) deadline = max_insns;
    return jit_->run(deadline, stop_pc, yield);
}

}  // namespace arm
