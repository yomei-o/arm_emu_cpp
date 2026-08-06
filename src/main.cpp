// armemu -- run a 32-bit ARM Linux binary on whatever this host happens to be.
//
//   armemu [options] program [args...]
//     --root DIR       the directory the guest sees as "/"
//     --trace-sys      log every syscall
//     --max N          stop after N instructions (a runaway guard)
//     --sample N       print the PC every N instructions
//     --watch LO:HI    log every guest access in an address range, with the PC
//     --strict         an access to a page nothing mapped is a fault, not a zero
//     --stats          print instruction count and mapped pages on exit
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "cpu.h"
#include "loader.h"
#include "memory.h"
#include "start.h"
#include "syscalls.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

std::vector<uint8_t> read_file(const char* path) {
    std::FILE* fp = std::fopen(path, "rb");
    if (!fp) return {};
    std::fseek(fp, 0, SEEK_END);
    const long n = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> v(n > 0 ? static_cast<size_t>(n) : 0);
    if (n > 0 && std::fread(v.data(), 1, static_cast<size_t>(n), fp) != static_cast<size_t>(n))
        v.clear();
    std::fclose(fp);
    return v;
}

// The address-space layout, in one place. These are close to what a real ARM Linux
// kernel picks, and the only hard requirement is that the five regions do not
// overlap -- the sibling project lost 90,000 instructions to an mmap arena that
// started where the interpreter had been placed, and the symptom was the loader
// executing zeroes, nowhere near the mmap that caused it.
constexpr uint32_t kPieBase    = 0x00400000u;   // where an ET_DYN program goes
constexpr uint32_t kInterpBase = 0xB6F00000u;   // where ld.so goes, as on real ARM
constexpr uint32_t kMmapBase   = 0x40000000u;   // the arena, growing up
constexpr uint32_t kStackTop   = 0xBEFFF000u;   // just under TASK_SIZE
constexpr uint32_t kStackSize  = 8u << 20;

}  // namespace

int main(int argc, char** argv) {
    using namespace arm;

#ifdef _WIN32
    // The guest's standard streams are byte streams, and Windows opens them in text
    // mode -- which silently deletes a CR before every LF on the way in and inserts
    // one on the way out. A guest that hashes its standard input then gets a digest
    // of *different bytes* from the ones the host would hash: a wrong answer that
    // looks exactly like an emulator bug and is not one.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif

    bool trace_sys = false, stats = false, strict = false;
    uint64_t max_insns = 0, sample = 0;
    uint32_t watch_lo = 0, watch_hi = 0, pc_watch = 0;
    std::vector<std::string> extra_env;
    std::string root = ".";
    int i = 1;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--trace-sys") trace_sys = true;
        else if (a == "--stats") stats = true;
        // --strict: a guest access to a page nothing mapped stops the run and names
        // the address, instead of reading zero and letting the mistake surface later
        // somewhere else. Off by default, because some libc reads speculatively and
        // refusing to run is worse than being permissive.
        else if (a == "--strict") strict = true;
        else if (a == "--watch" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const size_t colon = spec.find(':');
            watch_lo = static_cast<uint32_t>(std::strtoul(spec.c_str(), nullptr, 16));
            watch_hi = colon == std::string::npos
                           ? watch_lo + 0x4000
                           : static_cast<uint32_t>(std::strtoul(spec.c_str() + colon + 1, nullptr, 16));
        }
        else if (a == "--pcwatch" && i + 1 < argc)
            pc_watch = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 16));
        else if (a == "--max" && i + 1 < argc) max_insns = std::strtoull(argv[++i], nullptr, 0);
        else if (a == "--sample" && i + 1 < argc) sample = std::strtoull(argv[++i], nullptr, 0);
        else if (a == "--root" && i + 1 < argc) root = argv[++i];
        else if (a == "--setenv" && i + 1 < argc) extra_env.push_back(argv[++i]);
        else break;
    }
    if (i >= argc) {
        std::fprintf(stderr, "usage: armemu [--root DIR] [--trace-sys] [--stats] "
                             "[--max N] program [args...]\n");
        return 2;
    }

    // The program can be named by its host path or by the path the *guest* would
    // use. Both are natural: `armemu --root tree tree/bin/sh` is what a shell
    // completes, and `armemu --root tree /bin/sh` is what the guest's own scripts
    // say. Trying the host path first and the rooted one second costs one failed
    // open and removes a papercut that otherwise reaches into every test script.
    std::string prog_path = argv[i];
    std::vector<uint8_t> file = read_file(prog_path.c_str());
    if (file.empty() && !root.empty() && root != "." && prog_path[0] == '/') {
        prog_path = root + prog_path;
        file = read_file(prog_path.c_str());
    }
    if (file.empty()) { std::fprintf(stderr, "armemu: cannot read %s\n", argv[i]); return 1; }

    Memory mem;
    Cpu cpu(mem);
    cpu.max_insns = max_insns;
    cpu.sample_every = sample;
    Syscalls sys(cpu, mem);
    sys.trace = trace_sys;
    if (watch_hi) {
        mem.watch_lo = watch_lo;
        mem.watch_hi = watch_hi;
        mem.on_watch = [&cpu](uint32_t addr, unsigned size, uint64_t value, bool is_write) {
            std::fprintf(stderr, "[watch] %s %u @ %08X = %llX   from PC %08X\n",
                         is_write ? "write" : "read ", size, addr,
                         static_cast<unsigned long long>(value), cpu.cur_pc());
        };
    }
    if (pc_watch) {
        cpu.pc_watch = pc_watch;
        cpu.on_pc_watch = [&cpu, &mem]() {
            std::fprintf(stderr, "[pcwatch] %08X lr=%08X", cpu.pc, cpu.lr());
            for (unsigned k = 0; k <= 3; ++k) {
                const uint32_t val = cpu.r[k];
                std::fprintf(stderr, "  r%u=%X", k, val);
                if (val < 0x1000) continue;
                char s[40];
                unsigned n = 0;
                for (; n < sizeof s - 1; ++n) {
                    const uint8_t ch = mem.read<uint8_t>(val + n);
                    if (!ch) break;
                    if (ch < 0x20 || ch > 0x7E) { n = 0; break; }
                    s[n] = static_cast<char>(ch);
                }
                if (n >= 2) { s[n] = 0; std::fprintf(stderr, "(\"%s\")", s); }
            }
            std::fputc('\n', stderr);
        };
    }
    if (strict) {
        mem.strict = true;
        mem.on_unmapped = [&cpu](uint32_t addr, bool is_write) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "unmapped %s at %08X (--strict)",
                          is_write ? "write" : "read", addr);
            throw CpuError{buf, cpu.cur_pc(), 0};
        };
    }
    // An instruction we do not implement is delivered to the guest as SIGILL when it
    // has a handler. CPU-feature probes depend on exactly that.
    cpu.on_undefined = [&sys](uint32_t, uint32_t at) { return sys.deliver_signal(4, at); };
    // The guest's standard input is this process's. Without it every read of fd 0
    // returns end of file, and a guest in a pipeline hashes nothing and reports the
    // digest of the empty string -- which is a valid-looking answer.
    sys.input = [](char* dst, uint64_t len) -> int64_t {
        const size_t n = std::fread(dst, 1, static_cast<size_t>(len), stdin);
        return static_cast<int64_t>(n);
    };

    // argv[0] has to be the path *the guest* would see, not the host path we opened.
    // Python finds its standard library by walking up from argv[0]; hand it a host
    // path and it looks for the library in a directory that does not exist inside
    // the guest, then dies with "Failed to import encodings module".
    auto to_guest_path = [&root](std::string p) {
        auto fwd = [](std::string t) {
            for (char& ch : t) if (ch == '\\') ch = '/';
            return t;
        };
        std::string rt = fwd(root);
        p = fwd(p);
        while (!rt.empty() && rt.back() == '/') rt.pop_back();
        if (!rt.empty() && rt != "." && p.size() > rt.size() && p.compare(0, rt.size(), rt) == 0)
            return p.substr(rt.size());
        if (!p.empty() && p[0] != '/') return "/" + p;
        return p;
    };
    const std::string guest_exe = to_guest_path(prog_path);

    LoadedImage img;
    std::string err;
    if (!load_elf(file, mem, kPieBase, &img, &err)) {
        std::fprintf(stderr, "armemu: %s\n", err.c_str());
        return 1;
    }

    // A dynamically linked program is loaded exactly the way the kernel does it: map
    // the program, map its interpreter, and start at the *interpreter's* entry with
    // AT_BASE saying where it landed and AT_ENTRY where the program's is. The loader
    // does everything after that -- relocations, symbol binding, TLS. Writing our own
    // linker instead would mean reimplementing glibc's and getting the TLS model
    // wrong in some new way; running the real one is both less work and more
    // faithful.
    uint32_t start_pc = 0, interp_base = 0;
    bool interp_thumb = false;
    if (!img.interp.empty()) {
        std::string ipath = img.interp;
        if (!root.empty() && root != "." && !ipath.empty() && ipath[0] == '/') ipath = root + ipath;
        const std::vector<uint8_t> ifile = read_file(ipath.c_str());
        if (ifile.empty()) {
            std::fprintf(stderr,
                         "armemu: %s needs the dynamic loader %s,\n"
                         "        which is not under --root %s\n",
                         prog_path.c_str(), img.interp.c_str(), root.c_str());
            return 1;
        }
        LoadedImage interp;
        if (!load_elf(ifile, mem, kInterpBase, &interp, &err)) {
            std::fprintf(stderr, "armemu: loading %s: %s\n", img.interp.c_str(), err.c_str());
            return 1;
        }
        interp_base = interp.base;
        start_pc = interp.entry;
        interp_thumb = interp.thumb_entry;
    }

    std::vector<std::string> guest_argv, guest_env = {
        "PATH=/usr/bin:/bin", "HOME=/", "LANG=C.UTF-8", "TERM=dumb",
        "PYTHONDONTWRITEBYTECODE=1",
    };
    guest_env.insert(guest_env.end(), extra_env.begin(), extra_env.end());
    guest_argv.push_back(guest_exe);
    for (int k = i + 1; k < argc; ++k) guest_argv.push_back(argv[k]);

    mem.set(kStackTop - kStackSize, 0, kStackSize);
    cpu.set_sp(build_stack(mem, kStackTop, img, interp_base, guest_argv, guest_env));
    prepare_process(cpu, mem, img, start_pc, interp_thumb);
    sys.set_brk(img.brk);
    sys.exe_path = guest_exe;
    sys.files.set_root(root);
    sys.set_mmap_base(kMmapBase);

    // Running a child program. `execve` in a vfork child calls this: the child runs
    // here, to completion, in its own Memory and Cpu, and only then does the parent
    // resume. The descriptors are handed over and handed back, which is what makes a
    // pipe between the two work at all.
    //
    // Recursion is expected: gcc's driver execs cc1, and cc1 could exec something
    // else. The depth is capped because a guest that execs itself would otherwise
    // recurse until the host's stack ran out, and a stack overflow says nothing
    // about why.
    static int spawn_depth = 0;
    sys.spawn = [&](const std::string& prog, const std::vector<std::string>& sargv,
                    const std::vector<std::string>& senvp, Files& parent_files) -> int {
        if (spawn_depth >= 8) {
            std::fprintf(stderr, "armemu: refusing to nest more than 8 processes\n");
            return -1;
        }
        std::string host = prog;
        if (!root.empty() && root != "." && !host.empty() && host[0] == '/') host = root + host;
        const std::vector<uint8_t> cfile = read_file(host.c_str());
        if (cfile.empty()) return -1;

        Memory cmem;
        Cpu ccpu(cmem);
        Syscalls csys(ccpu, cmem);
        csys.trace = trace_sys;
        csys.files = parent_files;            // descriptors survive exec; that is the point
        csys.files.set_root(root);
        csys.spawn = sys.spawn;               // a child may spawn in turn
        ccpu.on_undefined = [&csys](uint32_t, uint32_t at) { return csys.deliver_signal(4, at); };

        LoadedImage cimg;
        std::string cerr_s;
        if (!load_elf(cfile, cmem, kPieBase, &cimg, &cerr_s)) {
            std::fprintf(stderr, "armemu: cannot run %s: %s\n", prog.c_str(), cerr_s.c_str());
            return -1;
        }
        uint32_t cstart = 0, cinterp = 0;
        bool cinterp_thumb = false;
        if (!cimg.interp.empty()) {
            std::string ip = cimg.interp;
            if (!root.empty() && root != "." && ip[0] == '/') ip = root + ip;
            const std::vector<uint8_t> ifile = read_file(ip.c_str());
            if (ifile.empty()) return -1;
            LoadedImage interp2;
            if (!load_elf(ifile, cmem, kInterpBase, &interp2, &cerr_s)) return -1;
            cinterp = interp2.base;
            cstart = interp2.entry;
            cinterp_thumb = interp2.thumb_entry;
        }
        cmem.set(kStackTop - kStackSize, 0, kStackSize);
        ccpu.set_sp(build_stack(cmem, kStackTop, cimg, cinterp, sargv, senvp));
        prepare_process(ccpu, cmem, cimg, cstart, cinterp_thumb);
        csys.set_brk(cimg.brk);
        csys.set_mmap_base(kMmapBase);
        csys.exe_path = prog;

        ++spawn_depth;
        int cstatus;
        try {
            cstatus = run_image(ccpu, cmem, csys);
        } catch (const CpuError& e) {
            std::fflush(stdout);
            std::fprintf(stderr, "\narmemu: in child %s: %s\n", prog.c_str(), e.what.c_str());
            cstatus = 127;
        }
        --spawn_depth;
        // The parent's descriptor table gets the child's back: the child may have
        // written into a pipe, and the buffer lives in the entry.
        parent_files = csys.files;
        return cstatus;
    };

    int rc = 0;
    try {
        rc = run_image(cpu, mem, sys);
    } catch (const CpuError& e) {
        std::fflush(stdout);
        std::fprintf(stderr, "\narmemu: %s  [%llu instructions]\n", e.what.c_str(),
                     static_cast<unsigned long long>(cpu.insns));
        // The registers, because where a guest stopped is rarely as informative as
        // what it was holding when it did.
        for (int k = 0; k < 16; k += 4) {
            std::fprintf(stderr, "  ");
            for (int j = k; j < k + 4; ++j)
                std::fprintf(stderr, "r%-2d=%08X ", j, j == 15 ? cpu.pc : cpu.r[j]);
            std::fputc('\n', stderr);
        }
        std::fprintf(stderr, "  cpsr=%08X (%c%c%c%c) %s  pc=%08X\n", cpu.cpsr(),
                     cpu.n ? 'N' : '-', cpu.z ? 'Z' : '-', cpu.c ? 'C' : '-', cpu.v ? 'V' : '-',
                     cpu.thumb ? "Thumb" : "ARM", cpu.cur_pc());
        // The frame chain. AAPCS keeps r11 pointing at the frame in ARM code and r7
        // in Thumb code, and neither is guaranteed -- so this walk is a hint, not a
        // fact, and it stops on anything that cannot be a frame.
        std::fprintf(stderr, "  frames (r11 chain):\n    %08X  (pc)\n    %08X  (lr)\n",
                     cpu.cur_pc(), cpu.lr());
        uint32_t fp = cpu.r[11];
        for (int depth = 0; depth < 24 && fp > 0x1000; ++depth) {
            const uint32_t next = mem.read<uint32_t>(fp - 12);
            const uint32_t ret = mem.read<uint32_t>(fp - 4);
            if (!ret) break;
            std::fprintf(stderr, "    %08X\n", ret);
            if (next <= fp) break;
            fp = next;
        }
        rc = 1;
    }
    std::fflush(stdout);
    if (stats)
        std::fprintf(stderr, "[stats] %llu instructions, %llu pages mapped (%llu KiB)\n",
                     static_cast<unsigned long long>(cpu.insns),
                     static_cast<unsigned long long>(mem.mapped_pages()),
                     static_cast<unsigned long long>(mem.mapped_pages() * Memory::kPageSize / 1024));
    return rc;
}
