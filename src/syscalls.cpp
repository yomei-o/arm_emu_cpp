// The Linux ARM (EABI) kernel interface.
//
// A guest talks to the outside world through `svc #0`, with the call number in
// **r7** and arguments in r0..r5. That single register is the whole difference in
// shape from the AArch64 sibling; the difference in *content* is much larger, and
// it is the numbering. ARM keeps the original Linux table -- write is 4, open is 5,
// brk is 45 -- rather than the "generic" one AArch64 was given, and it carries
// thirty years of history in it: `stat64` exists alongside `stat` because inodes
// outgrew 32 bits, `mmap2` exists alongside `mmap` because file offsets did, and
// `ugetrlimit` exists because the first `getrlimit` got the type wrong.
//
// Three consequences run through this file:
//
//   - **mmap2 takes its offset in 4096-byte pages**, not bytes. A guest mapping a
//     library at offset 0x1000 means byte 0x1000000, and getting that wrong maps the
//     wrong part of the file -- which reads as a corrupt library rather than as an
//     arithmetic mistake.
//   - **`struct stat64` is 104 bytes with padding in two places** (see files.h).
//   - **Anything with a 64-bit argument passes it in an aligned register pair**, so
//     `pread64(fd, buf, count, offset)` puts the offset in r4:r5, skipping r3.
//     That is the AAPCS rule, not a kernel one, and it is why some calls appear to
//     leave a register unused.
//
// Errors are returned the way Linux returns them: a negative errno in r0.
#include "syscalls.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

namespace arm {

namespace {
constexpr int64_t kEBADF = -9, kENOSYS = -38, kEINVAL = -22;

// The ARM private syscall space. `set_tls` is the one that matters: a guest whose
// libc predates the CP15 thread-pointer register -- which is every Raspberry Pi OS
// binary built for ARMv6 -- sets its thread pointer through this call and reads it
// back through the kernel helper page.
constexpr uint32_t kArmBase = 0x0F0000;
constexpr uint32_t kArmCacheflush = kArmBase + 2;
constexpr uint32_t kArmSetTls = kArmBase + 5;
constexpr uint32_t kArmGetTls = kArmBase + 6;
}  // namespace

Syscalls::Syscalls(Cpu& cpu, Memory& mem) : cpu_(cpu), mem_(mem) {
    cpu_.on_svc = [this](uint32_t imm) { return svc(imm); };
}

bool Syscalls::svc(uint32_t imm) {
    // EABI puts the number in r7 and the immediate is zero. OABI encoded it in the
    // immediate itself, offset by 0x900000, and a few very old binaries still do --
    // supporting both costs one compare and saves an unexplainable failure.
    uint32_t nr = cpu_.r[7];
    if (imm >= 0x900000) nr = imm - 0x900000;

    const uint32_t a0 = cpu_.r[0], a1 = cpu_.r[1], a2 = cpu_.r[2];
    const uint32_t a3 = cpu_.r[3], a4 = cpu_.r[4], a5 = cpu_.r[5];
    int64_t r = kENOSYS;

    if (trace)
        std::fprintf(stderr, "[sys] %u(%X, %X, %X) pc=%08X\n", nr, a0, a1, a2, cpu_.cur_pc());

    switch (nr) {
        // ---- the ARM private calls --------------------------------------------
        case kArmSetTls: cpu_.tpidruro = a0; r = 0; break;
        case kArmGetTls: r = cpu_.tpidruro; break;
        case kArmCacheflush: r = 0; break;      // no caches, and the code is re-read

        // ---- process lifetime --------------------------------------------------
        case 1:                                  // exit
        case 248:                                // exit_group
            // A vfork child that exits before it execs -- the `_exit(127)` after a
            // failed exec. The parent has to come back rather than the whole run
            // ending.
            if (in_vfork_child()) {
                vfork_child_status_ = static_cast<int>(a0 & 0xFF) << 8;
                vfork_returning_ = true;
                cpu_.stop_requested = true;
                return true;
            }
            if (nr == 1) { thread_exit(static_cast<int>(a0 & 0xFF)); return true; }
            cpu_.exit_code = static_cast<int>(a0 & 0xFF);
            cpu_.halted = true;
            return true;
        case 2: case 190:                        // fork, vfork
            cpu_.r[0] = static_cast<uint32_t>(sys_fork());
            return true;
        case 120:                                // clone
            // A thread is CLONE_VM *and* CLONE_THREAD. Everything else that comes
            // through here is a new process: `fork()` passes SIGCHLD alone, and
            // `posix_spawn()` passes CLONE_VM|CLONE_VFORK -- which shares the address
            // space but is emphatically not a thread.
            if (!((a0 & 0x00000100u) && (a0 & 0x00010000u))) {
                cpu_.r[0] = static_cast<uint32_t>(sys_fork());
                return true;
            }
            cpu_.r[0] = static_cast<uint32_t>(sys_clone(a0, a1, a2, a3, a4));
            return true;
        case 11: r = sys_execve(a0, a1, a2); return true;          // execve
        case 114: r = sys_wait4(static_cast<int32_t>(a0), a1); break;
        case 20: r = 1000; break;                                  // getpid
        case 64: r = 999; break;                                   // getppid
        case 224: r = static_cast<int64_t>(current_tid()); break;  // gettid
        case 199: case 200: case 201: case 202: r = 0; break;      // get[ug]id32
        case 24: case 47: case 49: case 50: r = 0; break;          // the OABI aliases
        case 205: r = 0; break;                                    // getgroups32
        case 208: case 210: case 213: case 214: r = 0; break;      // setres[ug]id32, set[ug]id32
        case 57: case 65: case 66: case 132: case 147: r = 0; break;  // setpgid/getpgrp/setsid/...
        case 172: r = 0; break;                                    // prctl
        case 116: mem_.set(a0, 0, 64); r = 0; break;               // sysinfo
        case 77: mem_.set(a1, 0, 72); r = 0; break;                // getrusage
        case 43: mem_.set(a0, 0, 16); r = 0; break;                // times

        // ---- memory ------------------------------------------------------------
        case 45: r = sys_brk(a0); break;
        case 192:                                                  // mmap2
            // The offset is in pages. This is the single most consequential
            // difference between the ARM table and a modern one.
            r = sys_mmap(a0, a1, a2, a3, static_cast<int>(static_cast<int32_t>(a4)),
                         static_cast<uint64_t>(a5) * 4096);
            break;
        case 90: {                                                 // old mmap: a struct
            const uint32_t args = a0;
            r = sys_mmap(mem_.read<uint32_t>(args), mem_.read<uint32_t>(args + 4),
                         mem_.read<uint32_t>(args + 8), mem_.read<uint32_t>(args + 12),
                         static_cast<int>(mem_.read<uint32_t>(args + 16)),
                         mem_.read<uint32_t>(args + 20));
            break;
        }
        case 91: r = 0; break;                                     // munmap: the arena only grows
        case 125: r = 0; break;                                    // mprotect: no protection modelled
        case 163: {                                                // mremap
            const uint32_t old_addr = a0, old_len = a1, new_len = a2;
            if (new_len <= old_len) { r = static_cast<int64_t>(old_addr); break; }
            const int64_t dst = sys_mmap(0, new_len, 0, 0x20, -1, 0);
            std::vector<uint8_t> tmp(old_len);
            mem_.read_bytes(old_addr, tmp.data(), old_len);
            mem_.write_bytes(static_cast<uint32_t>(dst), tmp.data(), old_len);
            r = dst;
            break;
        }
        case 220: r = 0; break;                                    // madvise
        case 219: r = 0; break;                                    // mincore
        case 150: case 151: case 152: case 153: r = 0; break;      // mlock family

        // ---- threads -----------------------------------------------------------
        case 240: sys_futex(a0, static_cast<int>(a1), a2); return true;
        case 158: sys_sched_yield(); return true;
        case 256:                                                  // set_tid_address
            tid_address_ = a0;
            if (!threads_.empty()) threads_[cur_thread_].clear_child_tid = a0;
            r = static_cast<int64_t>(current_tid());
            break;
        case 338: case 339: r = 0; break;                          // set/get_robust_list
        // membarrier. Guest threads are interleaved on one interpreter, never
        // concurrent, so every barrier is already satisfied and 0 is the truth
        // rather than a stub.
        case 389: r = 0; break;
        case 241: case 242: r = 0; break;                          // sched_[gs]etaffinity
        case 156: case 157: r = 0; break;                          // sched_[gs]etscheduler
        case 398: r = kENOSYS; break;                              // rseq: glibc falls back

        // ---- files -------------------------------------------------------------
        case 3: r = sys_read(static_cast<int>(a0), a1, a2); break;
        case 4: r = sys_write(static_cast<int>(a0), a1, a2); break;
        case 145: r = sys_readv(static_cast<int>(a0), a1, a2); break;
        case 146: r = sys_writev(static_cast<int>(a0), a1, a2); break;
        case 5: {                                                  // open
            const std::string path = guest_str(a0);
            r = files.open(path, static_cast<int>(a1), static_cast<int>(a2));
            if (trace)
                std::fprintf(stderr, "[sys]   open(\"%s\") -> %lld\n", path.c_str(),
                             static_cast<long long>(r));
            break;
        }
        case 322: {                                                // openat
            const std::string path = guest_str(a1);
            r = files.open(path, static_cast<int>(a2), static_cast<int>(a3));
            if (trace)
                std::fprintf(stderr, "[sys]   openat(\"%s\") -> %lld\n", path.c_str(),
                             static_cast<long long>(r));
            break;
        }
        case 6: r = files.close(static_cast<int>(a0)); break;
        case 10: r = files.unlink(guest_str(a0)); break;
        case 328: r = files.unlink(guest_str(a1)); break;           // unlinkat
        case 39: r = files.mkdir(guest_str(a0)); break;
        case 323: r = files.mkdir(guest_str(a1)); break;             // mkdirat
        case 38: r = files.rename(guest_str(a0), guest_str(a1)); break;
        case 329: r = files.rename(guest_str(a1), guest_str(a3)); break;  // renameat
        case 19: r = files.lseek(static_cast<int>(a0), static_cast<int32_t>(a1),
                                 static_cast<int>(a2)); break;
        case 140: {                                                // _llseek
            const int64_t off = (static_cast<int64_t>(a1) << 32) | a2;
            const int64_t got = files.lseek(static_cast<int>(a0), off, static_cast<int>(a4));
            if (got < 0) { r = got; break; }
            mem_.write<uint64_t>(a3, static_cast<uint64_t>(got));
            r = 0;
            break;
        }
        case 180: {                                                // pread64
            // The 64-bit offset is in r4:r5 because AAPCS aligns a 64-bit argument
            // to an even register pair, so r3 is skipped.
            std::vector<uint8_t> tmp(a2);
            const uint64_t off = (static_cast<uint64_t>(a5) << 32) | a4;
            r = files.pread(static_cast<int>(a0), tmp.data(), a2, off);
            if (r > 0) mem_.write_bytes(a1, tmp.data(), static_cast<uint64_t>(r));
            break;
        }
        case 195: case 196: {                                      // stat64, lstat64
            uint8_t st[kStatSize];
            const std::string path = guest_str(a0);
            r = files.stat_path(path, st);
            if (r == 0) mem_.write_bytes(a1, st, kStatSize);
            if (trace)
                std::fprintf(stderr, "[sys]   stat64(\"%s\") -> %lld\n", path.c_str(),
                             static_cast<long long>(r));
            break;
        }
        case 197: {                                                // fstat64
            uint8_t st[kStatSize];
            r = files.fstat(static_cast<int>(a0), st);
            if (r == 0) mem_.write_bytes(a1, st, kStatSize);
            break;
        }
        case 327: {                                                // fstatat64
            uint8_t st[kStatSize];
            const std::string path = guest_str(a1);
            // AT_EMPTY_PATH (0x1000) with an empty path means "stat the fd itself".
            // Statting "" as a path instead answers with the sysroot's root
            // directory, whose size makes every library the loader maps look 4096
            // bytes long -- one wrong stat, four unrelated-looking symptoms.
            if (path.empty() && (a3 & 0x1000)) r = files.fstat(static_cast<int>(a0), st);
            else r = files.stat_path(path, st);
            if (r == 0) mem_.write_bytes(a2, st, kStatSize);
            if (trace)
                std::fprintf(stderr, "[sys]   fstatat64(%d, \"%s\") -> %lld\n",
                             static_cast<int>(a0), path.c_str(), static_cast<long long>(r));
            break;
        }
        case 397: r = kENOSYS; break;                              // statx: glibc falls back
        case 217: {                                                // getdents64
            std::vector<uint8_t> tmp(a2);
            r = files.getdents64(static_cast<int>(a0), tmp.data(), a2);
            if (r > 0) mem_.write_bytes(a1, tmp.data(), static_cast<uint64_t>(r));
            break;
        }
        case 33: {                                                 // access
            const std::string path = guest_str(a0);
            r = files.access(path);
            if (trace)
                std::fprintf(stderr, "[sys]   access(\"%s\") -> %lld\n", path.c_str(),
                             static_cast<long long>(r));
            break;
        }
        case 334: case 439: {                                      // faccessat, faccessat2
            const std::string path = guest_str(a1);
            r = files.access(path);
            if (trace)
                std::fprintf(stderr, "[sys]   faccessat(\"%s\") -> %lld\n", path.c_str(),
                             static_cast<long long>(r));
            break;
        }
        case 183: {                                                // getcwd
            const std::string& d = files.cwd;
            if (a1 < d.size() + 1) { r = -34; break; }             // ERANGE
            mem_.write_bytes(a0, d.c_str(), d.size() + 1);
            r = static_cast<int64_t>(d.size() + 1);
            break;
        }
        case 12: files.cwd = guest_str(a0); r = 0; break;           // chdir
        case 85: case 332: {                                       // readlink, readlinkat
            const uint32_t path_arg = nr == 85 ? a0 : a1;
            const uint32_t buf = nr == 85 ? a1 : a2;
            const uint32_t size = nr == 85 ? a2 : a3;
            const std::string path = guest_str(path_arg);
            std::string target;
            if (path == "/proc/self/exe") target = exe_path;
            else target = files.readlink(path);
            if (target.empty()) { r = kEINVAL; break; }            // not a symlink
            const uint64_t n = target.size() < size ? target.size() : size;
            mem_.write_bytes(buf, target.c_str(), n);
            r = static_cast<int64_t>(n);
            break;
        }
        case 41: r = files.dup(static_cast<int>(a0), -1); break;    // dup
        case 63: case 358: r = files.dup(static_cast<int>(a0), static_cast<int>(a1)); break;
        case 42: case 359: {                                       // pipe, pipe2
            int fds[2];
            r = files.pipe2(fds);
            if (r == 0) {
                mem_.write<uint32_t>(a0, static_cast<uint32_t>(fds[0]));
                mem_.write<uint32_t>(a0 + 4, static_cast<uint32_t>(fds[1]));
            }
            break;
        }
        // close_range(first, last, flags). One call instead of a loop, and CPython
        // uses it when it is there -- the difference between a child that execs
        // immediately and one that makes a quarter of a million syscalls first.
        case 436: {
            const uint32_t last = a1 > 4096 ? 4096 : a1;
            for (uint32_t fd = a0; fd <= last; ++fd) files.close(static_cast<int>(fd));
            r = 0;
            break;
        }
        case 55: case 221:                                         // fcntl, fcntl64
            r = 0;
            break;
        case 54: r = -25; break;                                   // ioctl: ENOTTY, no tty here
        case 118: case 148: r = 0; break;                          // fsync, fdatasync
        case 94: case 15: case 333: r = 0; break;                  // fchmod, chmod, fchmodat
        case 60: r = 022; break;                                   // umask
        case 93: case 194: r = 0; break;                           // truncate, ftruncate64
        case 99: case 100: mem_.set(nr == 99 ? a2 : a2, 0, 84); r = 0; break;  // statfs
        case 226: case 229: case 231: r = -61; break;              // xattr: ENODATA

        // epoll is refused on purpose: Python's selectors module tries it first and
        // falls back to poll, and poll is the one that can be answered honestly
        // here -- an epoll set would have to remember registrations and report
        // edges, which is a lot of machinery for a design where the child has
        // always already finished by the time the parent asks.
        case 250: case 357: r = kENOSYS; break;
        case 168: case 336: {                                      // poll, ppoll
            // Nothing here runs two processes at once, so by the time the parent
            // polls, the child has finished and everything it wrote is in the
            // buffer. "Every descriptor asked about is ready" is therefore the
            // truthful answer rather than a convenient one -- a pipe with a dead
            // writer is ready to report end of file even when it is empty.
            const uint32_t count = a1;
            int64_t ready = 0;
            for (uint32_t i = 0; i < count && i < 64; ++i) {
                const uint32_t e = a0 + i * 8;                     // struct pollfd
                const uint16_t events = mem_.read<uint16_t>(e + 4);
                mem_.write<uint16_t>(e + 6, events);
                ++ready;
            }
            r = ready;
            break;
        }
        case 142: case 335: r = 0; break;                          // select: nothing ready

        // ---- signals -----------------------------------------------------------
        // These must *succeed*. Refusing rt_sigprocmask is not neutral: a libc
        // startup treats the failure as fatal-ish and busybox's applet dispatch
        // comes apart a few instructions later, ending up executing data.
        case 174: r = sys_rt_sigaction(static_cast<int>(a0), a1, a2); break;
        case 173: return (void)sys_rt_sigreturn(), true;            // rt_sigreturn
        case 119: return (void)sys_rt_sigreturn(), true;            // sigreturn
        case 175: case 176: case 179: case 186: r = 0; break;       // procmask/pending/suspend/altstack
        case 67: case 126: r = 0; break;                            // sigaction, sigprocmask (old)
        case 37: case 238: case 268: r = 0; break;                  // kill, tkill, tgkill

        // ---- time --------------------------------------------------------------
        case 78: {                                                 // gettimeofday
            const uint64_t ns = host_nanos();
            if (a0) {
                mem_.write<uint32_t>(a0, static_cast<uint32_t>(ns / 1000000000ull));
                mem_.write<uint32_t>(a0 + 4, static_cast<uint32_t>((ns % 1000000000ull) / 1000));
            }
            r = 0;
            break;
        }
        case 13: r = static_cast<int64_t>(host_nanos() / 1000000000ull); break;   // time
        case 263: case 264: {                                      // clock_gettime, clock_getres
            const uint64_t ns = nr == 263 ? host_nanos() : 1;
            mem_.write<uint32_t>(a1, static_cast<uint32_t>(ns / 1000000000ull));
            mem_.write<uint32_t>(a1 + 4, static_cast<uint32_t>(ns % 1000000000ull));
            r = 0;
            break;
        }
        case 403: case 406: {                                      // clock_gettime64, clock_getres64
            const uint64_t ns = nr == 403 ? host_nanos() : 1;
            mem_.write<uint64_t>(a1, ns / 1000000000ull);
            mem_.write<uint32_t>(a1 + 8, static_cast<uint32_t>(ns % 1000000000ull));
            r = 0;
            break;
        }
        case 162: case 265: r = 0; break;                          // nanosleep: return immediately

        // ---- limits ------------------------------------------------------------
        case 191: {                                                // ugetrlimit: 32-bit pairs
            uint32_t cur = 0xFFFFFFFFu, max = 0xFFFFFFFFu;
            if (a0 == 3) { cur = 8u << 20; max = 64u << 20; }      // RLIMIT_STACK
            if (a0 == 7) { cur = 256; max = 4096; }                // RLIMIT_NOFILE
            if (a1) { mem_.write<uint32_t>(a1, cur); mem_.write<uint32_t>(a1 + 4, max); }
            r = 0;
            break;
        }
        // prlimit64(pid, resource, new, old). Answering EINVAL is survivable until
        // something asks how many descriptors it might have: CPython's
        // `_posixsubprocess` closes every fd from 3 to the limit before it execs,
        // and with no answer it falls back to about twenty million.
        case 369: {
            uint64_t cur = ~0ull, max = ~0ull;
            if (a1 == 3) { cur = 8u << 20; max = 64u << 20; }
            if (a1 == 7) { cur = 256; max = 4096; }
            if (a3) { mem_.write<uint64_t>(a3, cur); mem_.write<uint64_t>(a3 + 8, max); }
            r = 0;
            break;
        }
        case 75: r = 0; break;                                     // setrlimit

        case 384: {                                                // getrandom
            for (uint32_t i = 0; i < a1; ++i)
                mem_.write<uint8_t>(a0 + i, static_cast<uint8_t>(0x9E * (i + 1) + 0x37));
            r = static_cast<int64_t>(a1);
            break;
        }
        case 122: {                                                // uname
            // Six 65-byte fields. The release string is *parsed* digit by digit by
            // glibc's `_dl_discover_osversion`, so it has to look like a real kernel
            // version, and `machine` has to agree with AT_PLATFORM or a guest looks
            // for its libraries in one place and reports the other.
            struct { char s[6][65]; } u{};
            std::snprintf(u.s[0], 65, "Linux");
            std::snprintf(u.s[1], 65, "armemu");
            std::snprintf(u.s[2], 65, "6.1.21-v7+");
            std::snprintf(u.s[3], 65, "#1 SMP arm_emu_cpp");
            std::snprintf(u.s[4], 65, "armv7l");
            std::snprintf(u.s[5], 65, "(none)");
            mem_.write_bytes(a0, &u, sizeof u);
            r = 0;
            break;
        }

        default:
            if (unknown) unknown(nr);
            std::fprintf(stderr, "[sys] unimplemented syscall %u at PC %08X\n", nr,
                         cpu_.cur_pc());
            r = kENOSYS;
            break;
    }
    cpu_.r[0] = static_cast<uint32_t>(r);
    // A vfork child that has just execed or exited: ask the run loop to stop so the
    // host can put the parent back. Set after r0, because the value written here
    // belongs to the child and is about to be discarded with it.
    if (vfork_pending()) cpu_.stop_requested = true;
    return true;
}

int64_t Syscalls::sys_write(int fd, uint32_t buf, uint32_t len) {
    if (len == 0) return 0;
    std::vector<uint8_t> tmp(len);
    mem_.read_bytes(buf, tmp.data(), len);
    // Descriptors 0..2 are the console *unless* something redirected them. A child
    // about to exec does exactly that -- `dup2(pipe_w, 1)` -- and sending fd 1 to
    // the terminal regardless makes the child's output appear on the screen while
    // its parent reads an empty pipe and reports failure.
    if (fd >= 0 && fd <= 2 && files.is_open(fd)) return files.write(fd, tmp.data(), len);
    if (fd > 2) return files.write(fd, tmp.data(), len);
    if (output) { output(fd, reinterpret_cast<const char*>(tmp.data()), len); return len; }
    if (fd == 1 || fd == 2) {
        std::FILE* f = fd == 2 ? stderr : stdout;
        std::fwrite(tmp.data(), 1, len, f);
        std::fflush(f);
        return len;
    }
    return kEBADF;
}

int64_t Syscalls::sys_read(int fd, uint32_t buf, uint32_t len) {
    if (fd == 0 && files.is_open(0)) {
        std::vector<uint8_t> tmp(len);
        const int64_t n = files.read(0, tmp.data(), len);
        if (n > 0) mem_.write_bytes(buf, tmp.data(), static_cast<uint64_t>(n));
        return n;
    }
    if (fd > 2) {
        std::vector<uint8_t> tmp(len);
        const int64_t got = files.read(fd, tmp.data(), len);
        if (got > 0) mem_.write_bytes(buf, tmp.data(), static_cast<uint64_t>(got));
        return got;
    }
    if (fd != 0) return kEBADF;
    if (!input) return 0;                                      // EOF
    std::vector<char> tmp(len);
    const int64_t got = input(tmp.data(), len);
    if (got > 0) mem_.write_bytes(buf, tmp.data(), static_cast<uint64_t>(got));
    return got;
}

// writev exists because a libc that buffers will flush through it rather than
// write(2); without it a program that "works" prints nothing. The iovec is two
// 32-bit words on ARM, not two 64-bit ones.
int64_t Syscalls::sys_writev(int fd, uint32_t iov, uint32_t cnt) {
    int64_t total = 0;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t base = mem_.read<uint32_t>(iov + i * 8);
        const uint32_t len = mem_.read<uint32_t>(iov + i * 8 + 4);
        if (!len) continue;
        const int64_t w = sys_write(fd, base, len);
        if (w < 0) return total ? total : w;
        total += w;
    }
    return total;
}

int64_t Syscalls::sys_readv(int fd, uint32_t iov, uint32_t cnt) {
    int64_t total = 0;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t base = mem_.read<uint32_t>(iov + i * 8);
        const uint32_t len = mem_.read<uint32_t>(iov + i * 8 + 4);
        if (!len) continue;
        const int64_t got = sys_read(fd, base, len);
        if (got < 0) return total ? total : got;
        total += got;
        if (static_cast<uint32_t>(got) < len) break;     // a short read ends the vector
    }
    return total;
}

// brk(0) reports the break; brk(x) moves it. Memory is allocated on touch, so there
// is nothing to map -- the only job is to answer consistently, because a libc
// computes its heap size from the difference between two calls.
int64_t Syscalls::sys_brk(uint32_t addr) {
    if (addr && addr >= brk_start_) {
        // Map what the break now covers. With permissive memory nothing had to be
        // done here -- a write allocated its own page -- but that is exactly what
        // `--strict` withdraws, and growing the break is the guest *asking* for the
        // memory.
        if (addr > brk_) mem_.map(brk_, addr - brk_);
        brk_ = addr;
    }
    return static_cast<int64_t>(brk_);
}

// mmap.
//
// A static guest only needs "give me a zeroed region somewhere". A *dynamic* one
// needs the actual contract, because the dynamic loader builds every shared library
// out of it: first one PROT_NONE anonymous reservation spanning the whole object,
// then a MAP_FIXED file mapping over each segment at its own offset. Get MAP_FIXED
// wrong, or ignore the file offset, and ld.so lays libc down in pieces that do not
// join up.
//
// What is *not* modelled: protection (nothing here faults), sharing (a file mapping
// is a private copy), and unmapping (the arena only grows). Those are honest gaps
// rather than approximations.
int64_t Syscalls::sys_mmap(uint32_t addr, uint32_t len, uint32_t prot, uint32_t flags,
                           int fd, uint64_t off) {
    (void)prot;
    constexpr uint32_t kMAP_FIXED = 0x10, kMAP_ANONYMOUS = 0x20;
    const bool anon = (flags & kMAP_ANONYMOUS) || fd < 0;

    uint32_t base;
    if (addr && (flags & kMAP_FIXED)) {
        base = addr;
    } else {
        base = mmap_next_;
        mmap_next_ = (mmap_next_ + len + 0xFFFF) & ~0xFFFFu;
    }

    // Every mapping starts zeroed, including the tail of a file mapping that runs
    // past end-of-file -- which is exactly how a .bss inside a PT_LOAD gets there.
    mem_.set(base, 0, len);
    if (!anon) {
        std::vector<uint8_t> tmp(len);
        const int64_t got = files.pread(fd, tmp.data(), len, off);
        if (got > 0) mem_.write_bytes(base, tmp.data(), static_cast<uint64_t>(got));
    }
    return static_cast<int64_t>(base);
}

uint64_t Syscalls::host_nanos() {
    const auto t = std::clock();
    return static_cast<uint64_t>(t) * (1000000000ull / CLOCKS_PER_SEC);
}

}  // namespace arm
