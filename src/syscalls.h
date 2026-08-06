#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "cpu.h"
#include "files.h"
#include "loader.h"
#include "memory.h"

namespace arm {

class Syscalls {
public:
    Syscalls(Cpu& cpu, Memory& mem);

    // Where the heap and the mmap arena live. Set by the host after loading, so the
    // break starts just above the image.
    void set_brk(uint32_t addr) { brk_ = brk_start_ = addr; }
    void set_mmap_base(uint32_t addr) { mmap_next_ = addr; }

    Files files;
    // What /proc/self/exe should report; a libc reads it to find its own prefix.
    std::string exe_path;
    bool trace = false;
    std::function<void(int fd, const char* data, uint64_t len)> output;
    std::function<int64_t(char* dst, uint64_t len)> input;
    std::function<void(uint32_t nr)> unknown;
    // Run another program to completion and return its exit status, sharing this
    // process's descriptors. The host supplies it because loading a program means
    // reading files and choosing an address-space layout, which is the front end's
    // job -- `main.cpp` and `web/wasm_api.cpp` each have their own.
    //
    // Returning -1 means "could not start it", which `execve` reports as ENOENT.
    std::function<int(const std::string& path,
                      const std::vector<std::string>& argv,
                      const std::vector<std::string>& envp,
                      Files& files)> spawn;

    // Signal delivery, in signals.cpp. Returns false when nothing is installed --
    // then the caller reports the fault, which is the default action anyway.
    bool deliver_signal(int sig, uint32_t fault_addr);

    // The address a handler returns to when the guest supplied no restorer.
    static constexpr uint32_t kSigreturnMagic = 0xDEAD0000u;

    // True when a vfork child has just finished and the parent is waiting to be put
    // back. The run loop checks it; `vfork_resume()` does the putting back.
    bool vfork_pending() const { return vfork_returning_; }
    void vfork_resume();

private:
    bool svc(uint32_t imm);
    int64_t sys_write(int fd, uint32_t buf, uint32_t len);
    int64_t sys_read(int fd, uint32_t buf, uint32_t len);
    int64_t sys_writev(int fd, uint32_t iov, uint32_t cnt);
    int64_t sys_readv(int fd, uint32_t iov, uint32_t cnt);
    int64_t sys_brk(uint32_t addr);
    int64_t sys_mmap(uint32_t addr, uint32_t len, uint32_t prot, uint32_t flags,
                     int fd, uint64_t off);
    int64_t sys_rt_sigaction(int sig, uint32_t act, uint32_t oact);
    int64_t sys_rt_sigreturn();
    std::string guest_str(uint32_t addr) { return mem_.read_cstr(addr); }
    static uint64_t host_nanos();

    // Threads, in threads.cpp. One emulated CPU running one guest thread at a time,
    // switched at futex/yield/exit and by preemption. These write r0 themselves,
    // because the thread that made the call may not be the one running on return.
    struct Thread {
        Cpu::Context ctx;
        uint32_t tid = 0;
        uint32_t clear_child_tid = 0;   // the join word: cleared and woken on exit
        uint32_t wait_addr = 0;
        bool waiting = false;           // blocked in futex(WAIT)
        bool exited = false;
    };
    std::vector<Thread> threads_;       // empty until the first clone
    size_t cur_thread_ = 0;
    uint32_t next_tid_ = 1001;

    void ensure_main_thread();
    void switch_to(size_t idx);
    bool schedule(bool must_move);
    uint32_t current_tid() const;
    int64_t sys_clone(uint32_t flags, uint32_t stack, uint32_t ptid, uint32_t tls,
                      uint32_t ctid);
    int64_t sys_futex(uint32_t addr, int op, uint32_t val);
    int64_t sys_sched_yield();
    void thread_exit(int status);

    // fork/execve/wait4, in process.cpp. See the note there: the child is a *vfork*
    // child -- the parent is suspended and the two never run at the same time.
    int64_t sys_fork();
    int64_t sys_execve(uint32_t path, uint32_t argv, uint32_t envp);
    int64_t sys_wait4(int32_t pid, uint32_t status_addr);
    bool in_vfork_child() const { return vfork_depth_ > 0; }

    std::vector<Cpu::Context> vfork_saved_;
    // The parent's descriptor table, saved across a vfork child.
    //
    // Real vfork *shares* the table, and POSIX therefore forbids a child from
    // touching it -- but every real fork+exec does touch it, because redirecting the
    // child's standard streams is the entire point. Sharing it leaves the *parent*
    // with fd 1 pointing at the pipe after the child is gone. The pipes themselves
    // are shared_ptr, so what the child wrote survives the copy.
    std::vector<Files> vfork_files_;
    int vfork_depth_ = 0;
    std::map<int, int> child_status_;
    int next_pid_ = 2000;
    bool vfork_returning_ = false;
    int vfork_child_pid_ = 0, vfork_child_status_ = 0;

    struct Handler { uint32_t func = 0, flags = 0, restorer = 0; };
    Handler handlers_[64];
    struct SigFrame { uint32_t frame; Cpu::Regs saved; };
    std::vector<SigFrame> sig_frames_;

    Cpu& cpu_;
    Memory& mem_;
    uint32_t brk_ = 0, brk_start_ = 0;
    // Set by the host, which owns the address-space layout. The default is only a
    // fallback; main.cpp places it clear of the image, the interpreter and the stack.
    uint32_t mmap_next_ = 0x40000000u;
    uint32_t tid_address_ = 0;
};

}  // namespace arm
