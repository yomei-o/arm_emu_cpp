// Threads: `clone`, `futex`, and a scheduler that runs one of them at a time.
//
// One emulated CPU, N guest threads, switched by saving and restoring the whole
// register file. That is not how a real kernel does it, but it is enough for the
// thing a guest actually depends on -- that another thread eventually *runs* -- and
// it keeps the interpreter single-threaded, so no memory model question arises.
//
// Two places give the CPU up:
//
//  - **Voluntarily**, at `futex(WAIT)`, `sched_yield`, and thread exit. This is the
//    common path: a libc blocks on a futex whenever a lock is contended.
//  - **By preemption**, every N instructions. A guest that spins instead of
//    blocking -- and one exists in almost every libc, in the short spin before the
//    futex -- would otherwise hold the CPU forever. Preemption is switched on only
//    when a second thread appears, so a single-threaded guest pays one branch.
//
// A switch has to be all-or-nothing: any syscall that may switch writes its own
// return value into r0 *before* switching, and returns without falling into the
// shared `cpu_.r[0] = r` tail -- which would otherwise land in whichever thread
// happens to be current afterwards.
#include "syscalls.h"
#include <cstdio>

namespace arm {

namespace {
constexpr uint32_t kCloneVm     = 0x00000100;
constexpr uint32_t kCloneThread = 0x00010000;
// Long enough that switching is not the dominant cost, short enough that a spin of
// a few thousand iterations still yields. Nothing depends on the exact value.
constexpr uint64_t kPreemptEvery = 20000;
}  // namespace

// The main thread is not created by `clone`, so it does not exist as a Thread until
// the moment a second one does. Materialising it lazily keeps every single-threaded
// guest on exactly the path it had before threads existed.
void Syscalls::ensure_main_thread() {
    if (!threads_.empty()) return;
    Thread t;
    t.tid = 1000;
    t.clear_child_tid = tid_address_;
    threads_.push_back(t);
    cur_thread_ = 0;
}

uint32_t Syscalls::current_tid() const {
    return threads_.empty() ? 1000 : threads_[cur_thread_].tid;
}

void Syscalls::switch_to(size_t idx) {
    if (idx == cur_thread_) return;
    threads_[cur_thread_].ctx = cpu_.save_context();
    cur_thread_ = idx;
    cpu_.load_context(threads_[idx].ctx);
}

// Round-robin from the one after the current, so a thread that yields does not
// immediately get picked again. Returns false when nothing else can run.
bool Syscalls::schedule(bool must_move) {
    if (threads_.size() < 2) return !must_move;
    const size_t n = threads_.size();
    for (size_t k = 1; k <= n; ++k) {
        const size_t i = (cur_thread_ + k) % n;
        if (i == cur_thread_ && must_move) continue;
        if (threads_[i].exited || threads_[i].waiting) continue;
        switch_to(i);
        return true;
    }
    return false;
}

int64_t Syscalls::sys_clone(uint32_t flags, uint32_t stack, uint32_t ptid, uint32_t tls,
                            uint32_t ctid) {
    if (!(flags & kCloneVm) || !(flags & kCloneThread)) {
        std::fprintf(stderr, "[sys] clone(%X) is a fork, not a thread; unsupported\n", flags);
        return -38;                                          // ENOSYS
    }
    ensure_main_thread();

    Thread child;
    child.tid = next_tid_++;
    // The child resumes exactly where the parent is -- just past the SVC -- with a
    // stack of its own, its own thread pointer, and 0 in r0 to tell the two apart.
    child.ctx = cpu_.save_context();
    child.ctx.r[13] = stack;
    child.ctx.tpidruro = tls;
    child.ctx.r[0] = 0;
    child.clear_child_tid = ctid;
    if (ptid) mem_.write<uint32_t>(ptid, child.tid);
    if (ctid) mem_.write<uint32_t>(ctid, child.tid);
    threads_.push_back(child);

    // From here the guest is concurrent, so the CPU has to be taken away from a
    // thread that never asks.
    cpu_.preempt_every = kPreemptEvery;
    cpu_.preempt_left = kPreemptEvery;
    cpu_.on_preempt = [this] { schedule(true); };
    return static_cast<int64_t>(child.tid);
}

int64_t Syscalls::sys_sched_yield() {
    cpu_.r[0] = 0;
    schedule(true);
    return 0;
}

// FUTEX_WAIT and FUTEX_WAKE, which is all a libc needs. The private and
// clock-realtime flags change nothing for a single address space.
int64_t Syscalls::sys_futex(uint32_t addr, int op, uint32_t val) {
    const int base = op & 0x7F;
    if (base == 0 || base == 9) {                            // WAIT, WAIT_BITSET
        // The load and the compare are atomic with respect to the guest because
        // nothing else is running: the whole point of the check is that the value
        // may have changed between the guest's own load and this call.
        if (mem_.read<uint32_t>(addr) != val) {
            cpu_.r[0] = static_cast<uint32_t>(-11);          // EAGAIN
            return 0;
        }
        ensure_main_thread();
        cpu_.r[0] = 0;                                       // woken normally
        threads_[cur_thread_].waiting = true;
        threads_[cur_thread_].wait_addr = addr;
        if (!schedule(true)) {
            // Every thread is blocked on a futex nobody will post. That is a guest
            // deadlock, and reporting it beats spinning forever in a loop that looks
            // exactly like hard work.
            threads_[cur_thread_].waiting = false;
            throw CpuError{"all threads are blocked on a futex: deadlock", cpu_.pc, 0};
        }
        return 0;
    }
    if (base == 1 || base == 10) {                           // WAKE, WAKE_BITSET
        uint32_t woken = 0;
        for (Thread& t : threads_) {
            if (woken >= val) break;
            if (t.waiting && t.wait_addr == addr) { t.waiting = false; ++woken; }
        }
        cpu_.r[0] = woken;
        return 0;
    }
    // FUTEX_REQUEUE and friends. Waking everyone is not what they mean, but it is
    // never *wrong* -- a spurious wake is allowed and the guest rechecks -- whereas
    // ENOSYS would strand the waiters.
    for (Thread& t : threads_) if (t.waiting && t.wait_addr == addr) t.waiting = false;
    cpu_.r[0] = 0;
    return 0;
}

// exit(2) ends one thread; exit_group(2) ends the process. Before threads existed
// the two were the same call, and treating exit as exit_group here would kill the
// program the first time any worker finished.
void Syscalls::thread_exit(int status) {
    if (threads_.size() < 2) {
        cpu_.exit_code = status;
        cpu_.halted = true;
        return;
    }
    Thread& me = threads_[cur_thread_];
    me.exited = true;
    me.waiting = false;
    // pthread_join waits on this word, and the kernel is what clears it and wakes
    // the joiner. Skip it and join() hangs on a thread that has already gone.
    if (me.clear_child_tid) {
        mem_.write<uint32_t>(me.clear_child_tid, 0);
        for (Thread& t : threads_)
            if (t.waiting && t.wait_addr == me.clear_child_tid) t.waiting = false;
    }
    if (!schedule(true)) {
        cpu_.exit_code = status;
        cpu_.halted = true;
    }
}

}  // namespace arm
