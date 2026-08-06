// ELF32 loader for ARM Linux executables, and the initial process image.
//
// The auxiliary vector is the part that has to be exactly right. A libc reads all
// of it before `main`, and on ARM two entries decide things that are invisible
// until much later:
//
//   AT_HWCAP tells glibc which memcpy, strlen and memset to install. It picks its
//   IFUNC resolvers from this word, so a HWCAP that claims less than the emulator
//   implements costs speed, and one that claims *more* sends the guest into
//   instructions that are not there -- as a SIGILL inside strlen, hundreds of
//   thousands of instructions after start-up, with nothing pointing at the auxv.
//
//   AT_PLATFORM is the string the loader appends to library search paths
//   ("/lib/arm-linux-gnueabihf/v7l/"). It is "v7l" for ARMv7, which is also what
//   `uname -m` reports, and the two must agree or a guest finds no libraries in one
//   place and reports the other.
#include "loader.h"
#include <cstring>
#include <cstdio>
#include "cpu.h"          // the kuser helper addresses live with the CPU that reaches them

namespace arm {

namespace {

struct Ehdr {
    uint8_t ident[16];
    uint16_t type, machine;
    uint32_t version, entry, phoff, shoff, flags;
    uint16_t ehsize, phentsize, phnum, shentsize, shnum, shstrndx;
};
struct Phdr {
    uint32_t type, offset, vaddr, paddr, filesz, memsz, flags, align;
};

constexpr uint32_t PT_LOAD = 1, PT_INTERP = 3;

constexpr uint32_t AT_NULL = 0, AT_PHDR = 3, AT_PHENT = 4, AT_PHNUM = 5,
                   AT_PAGESZ = 6, AT_BASE = 7, AT_FLAGS = 8, AT_ENTRY = 9,
                   AT_UID = 11, AT_EUID = 12, AT_GID = 13, AT_EGID = 14,
                   AT_PLATFORM = 15, AT_HWCAP = 16, AT_CLKTCK = 17,
                   AT_SECURE = 23, AT_RANDOM = 25, AT_HWCAP2 = 26, AT_EXECFN = 31;

// What a Raspberry Pi 2/3 running the 32-bit OS reports, minus the two bits that
// describe the kernel rather than the instruction set (LPAE, EVTSTRM). Every
// feature named here is implemented: NEON and both VFP generations in vfp.cpp and
// neon.cpp, TLS as CP15 c13, and the divide instructions in the media group.
//
// The temptation is to advertise everything and let the guest sort it out. That is
// backwards -- HWCAP is a *promise*, and glibc keeps the emulator to it by
// selecting code paths from it.
constexpr uint32_t kHwcap =
    (1u << 1) |    // HALF        halfword loads and stores
    (1u << 2) |    // THUMB
    (1u << 4) |    // FAST_MULT
    (1u << 6) |    // VFP
    (1u << 7) |    // EDSP        the saturating and packing instructions
    (1u << 12) |   // NEON
    (1u << 13) |   // VFPv3
    (1u << 15) |   // TLS         CP15 c13,c0,{2,3}
    (1u << 16) |   // VFPv4       the fused multiply-adds
    (1u << 17) |   // IDIVA       SDIV/UDIV in ARM state
    (1u << 18) |   // IDIVT       SDIV/UDIV in Thumb state
    (1u << 19);    // VFPD32      32 double registers, not 16

}  // namespace

bool load_elf(const std::vector<uint8_t>& f, Memory& mem, uint32_t base,
              LoadedImage* out, std::string* err) {
    if (f.size() < sizeof(Ehdr) || std::memcmp(f.data(), "\x7f" "ELF", 4) != 0) {
        *err = "not an ELF file"; return false;
    }
    Ehdr eh{};
    std::memcpy(&eh, f.data(), sizeof eh);
    if (eh.ident[4] != 1) { *err = "not ELF32 (a 64-bit ARM binary needs aarch64_emu_cpp)"; return false; }
    if (eh.ident[5] != 1) { *err = "not little-endian"; return false; }
    if (eh.machine != 40) { *err = "not an ARM image (EM_ARM = 40)"; return false; }
    if (eh.type != 2 && eh.type != 3) { *err = "not an executable"; return false; }

    // ET_EXEC says where it wants to live; ET_DYN -- a PIE, or the dynamic loader
    // itself -- is placed wherever we choose and every address in it shifts by that.
    if (eh.type == 2) base = 0;
    out->base = base;
    out->entry = (eh.entry & ~1u) + base;
    out->thumb_entry = (eh.entry & 1) != 0;
    out->phdr_addr = 0;
    out->phent = eh.phentsize;
    out->phnum = eh.phnum;
    out->interp.clear();
    uint32_t brk = 0;

    for (unsigned i = 0; i < eh.phnum; ++i) {
        const size_t po = static_cast<size_t>(eh.phoff) + i * eh.phentsize;
        if (po + sizeof(Phdr) > f.size()) { *err = "program header out of range"; return false; }
        Phdr ph{};
        std::memcpy(&ph, f.data() + po, sizeof ph);

        if (ph.type == PT_INTERP && ph.filesz) {
            out->interp.assign(reinterpret_cast<const char*>(f.data() + ph.offset),
                               static_cast<size_t>(ph.filesz - 1));
        }
        if (ph.type != PT_LOAD) continue;
        if (static_cast<uint64_t>(ph.offset) + ph.filesz > f.size()) {
            *err = "PT_LOAD extends past the file"; return false;
        }
        mem.write_bytes(base + ph.vaddr, f.data() + ph.offset, ph.filesz);
        if (ph.memsz > ph.filesz) mem.set(base + ph.vaddr + ph.filesz, 0, ph.memsz - ph.filesz);
        if (base + ph.vaddr + ph.memsz > brk) brk = base + ph.vaddr + ph.memsz;

        // The program headers are usually inside the first PT_LOAD; a libc finds
        // them through AT_PHDR and walks them to locate its own PT_TLS and
        // PT_DYNAMIC.
        if (!out->phdr_addr && ph.offset <= eh.phoff &&
            static_cast<uint64_t>(eh.phoff) + static_cast<uint64_t>(eh.phnum) * eh.phentsize
                <= static_cast<uint64_t>(ph.offset) + ph.filesz) {
            out->phdr_addr = base + ph.vaddr + (eh.phoff - ph.offset);
        }
    }
    out->brk = (brk + 0xFFF) & ~0xFFFu;
    return true;
}

// The kernel helper page.
//
// ARM Linux maps a page at 0xFFFF0000 containing three routines every process may
// call: an atomic compare-and-swap, a memory barrier, and a read of the thread
// pointer. They exist because those three operations need instructions that are
// not on every ARM the kernel supports, so the kernel supplies a version that
// works on the CPU it is actually running on -- and Raspberry Pi's own armv6
// toolchain output calls them rather than using LDREX/STREX directly.
//
// They are written here as **real ARM code**, not intercepted in the interpreter.
// That is the better trade twice over: the fast path costs nothing (no per
// instruction check for a magic address), and what runs is the kernel's own
// implementation, so the compare-and-swap goes through this emulator's exclusive
// monitor exactly as a guest's own would.
void map_kuser_helpers(Memory& mem) {
    mem.map(kKuserBase, 0x1000);

    // __kuser_cmpxchg(oldval r0, newval r1, ptr r2) -> r0 = 0 on success, and the
    // C flag set. This is the kernel's own ARMv6+ implementation.
    static const uint32_t cmpxchg[] = {
        0xE1923F9F,   // 1: ldrex r3, [r2]
        0xE0533000,   //    subs  r3, r3, r0
        0x01823F91,   //    strexeq r3, r1, [r2]
        0x03330001,   //    teqeq r3, #1
        0x0AFFFFFA,   //    beq   1b
        0xE2730000,   //    rsbs  r0, r3, #0
        0xE12FFF1E,   //    bx    lr
    };
    for (unsigned i = 0; i < sizeof cmpxchg / 4; ++i)
        mem.write<uint32_t>(kKuserCmpxchg + i * 4, cmpxchg[i]);

    // __kuser_memory_barrier: one CPU, so the barrier is the return.
    static const uint32_t barrier[] = {
        0xF57FF05F,   // dmb ish
        0xE12FFF1E,   // bx lr
    };
    for (unsigned i = 0; i < sizeof barrier / 4; ++i)
        mem.write<uint32_t>(kKuserMemoryBarrier + i * 4, barrier[i]);

    // __kuser_get_tls -> r0. Reads TPIDRURO, which exec_cp15 answers.
    static const uint32_t get_tls[] = {
        0xEE1D0F70,   // mrc p15, 0, r0, c13, c0, 3
        0xE12FFF1E,   // bx lr
    };
    for (unsigned i = 0; i < sizeof get_tls / 4; ++i)
        mem.write<uint32_t>(kKuserGetTls + i * 4, get_tls[i]);

    // __kuser_helper_version, which a libc reads to find out which of the above
    // exist. 5 is "all of them, including cmpxchg64".
    mem.write<uint32_t>(0xFFFF0FFCu, 5);
}

uint32_t build_stack(Memory& mem, uint32_t stack_top, const LoadedImage& img,
                     uint32_t interp_base,
                     const std::vector<std::string>& argv,
                     const std::vector<std::string>& envp) {
    uint32_t p = stack_top;
    auto push_str = [&](const std::string& s) {
        p -= static_cast<uint32_t>(s.size() + 1);
        mem.write_bytes(p, s.c_str(), s.size() + 1);
        return p;
    };

    std::vector<uint32_t> argp, envpp;
    for (const std::string& s : argv) argp.push_back(push_str(s));
    for (const std::string& s : envp) envpp.push_back(push_str(s));

    // AT_RANDOM must point at 16 readable bytes; a libc copies them into its stack
    // guard and TLS canary before anything else runs.
    const uint8_t rnd[16] = {0x5a, 0x0f, 0x1c, 0x77, 0x21, 0x93, 0xb4, 0xe6,
                             0x38, 0xd1, 0x4c, 0x0a, 0x95, 0x62, 0xfe, 0x83};
    p -= sizeof rnd;
    const uint32_t random_addr = p;
    mem.write_bytes(p, rnd, sizeof rnd);

    const uint32_t execfn = argp.empty() ? 0 : argp[0];
    const uint32_t platform = push_str("v7l");

    std::vector<std::pair<uint32_t, uint32_t>> aux = {
        {AT_PHDR, img.phdr_addr}, {AT_PHENT, img.phent}, {AT_PHNUM, img.phnum},
        {AT_PAGESZ, 4096}, {AT_BASE, interp_base}, {AT_FLAGS, 0},
        {AT_ENTRY, img.entry | (img.thumb_entry ? 1u : 0u)},
        {AT_UID, 0}, {AT_EUID, 0}, {AT_GID, 0}, {AT_EGID, 0},
        {AT_HWCAP, kHwcap}, {AT_CLKTCK, 100},
        {AT_RANDOM, random_addr}, {AT_SECURE, 0}, {AT_EXECFN, execfn},
        {AT_HWCAP2, 0}, {AT_PLATFORM, platform},
        {AT_NULL, 0},
    };

    // The block has to leave SP 8-byte aligned: the AAPCS requires it at a public
    // interface, and a `vpush {d8}` in a libc's start-up faults on anything less.
    const size_t words = 1 + argp.size() + 1 + envpp.size() + 1 + aux.size() * 2;
    uint32_t sp = p - static_cast<uint32_t>(words * 4);
    sp &= ~7u;

    uint32_t w = sp;
    auto put = [&](uint32_t v) { mem.write<uint32_t>(w, v); w += 4; };
    put(static_cast<uint32_t>(argp.size()));
    for (uint32_t a : argp) put(a);
    put(0);
    for (uint32_t e : envpp) put(e);
    put(0);
    for (auto& kv : aux) { put(kv.first); put(kv.second); }
    return sp;
}

}  // namespace arm
