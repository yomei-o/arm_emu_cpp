// Signal delivery.
//
// This exists because "unimplemented instruction" is not always an emulator gap to
// be filled. Library code detects CPU features by *executing* them under a SIGILL
// handler -- OpenSSL does it, and so does anything that has to run on both an
// ARMv6 Raspberry Pi and an ARMv7 one from the same binary. There, SIGILL is *the
// answer the guest is asking for*, and the only way to give it is the way the
// kernel does: build a signal frame and call the handler.
//
// The frame is the kernel's `struct rt_sigframe` for ARM: a siginfo, then a
// ucontext whose `uc_mcontext` is `struct sigcontext` -- and that structure starts
// with three words (trap_no, error_code, oldmask) *before* r0, which is the offset
// everything else here is measured from.
#include "syscalls.h"
#include <cstdio>
#include <cstring>

namespace arm {

namespace {
constexpr uint32_t kSiginfoSize = 128;
// struct ucontext: uc_flags, uc_link, then a 12-byte stack_t.
constexpr uint32_t kUcMcontextOff = 4 + 4 + 12;
// struct sigcontext: trap_no, error_code, oldmask, then r0..r15, cpsr, fault_address.
constexpr uint32_t kScRegsOff = 12;
constexpr uint32_t kScCpsrOff = kScRegsOff + 16 * 4;
constexpr uint32_t kScFaultOff = kScCpsrOff + 4;
// Room for the sigmask and the VFP/coprocessor blocks a real frame carries after
// the sigcontext. Nothing here reads them, but a handler that walks the frame
// should not run off the end of mapped memory.
constexpr uint32_t kFrameSize = kSiginfoSize + kUcMcontextOff + 84 + 512;
}  // namespace

bool Syscalls::deliver_signal(int sig, uint32_t fault_addr) {
    if (sig < 0 || sig >= 64) return false;
    const Handler& h = handlers_[sig];
    // 0 = SIG_DFL, 1 = SIG_IGN. Neither is something we can act on usefully: the
    // default for SIGILL is to kill the process, which is what failing already does.
    if (h.func <= 1) return false;

    // The frame goes below the current stack, 8-byte aligned as the AAPCS requires.
    const uint32_t frame = (cpu_.sp() - kFrameSize) & ~7u;
    mem_.set(frame, 0, kFrameSize);

    const uint32_t siginfo = frame;
    const uint32_t uc = frame + kSiginfoSize;
    const uint32_t sc = uc + kUcMcontextOff;

    mem_.write<uint32_t>(siginfo + 0, static_cast<uint32_t>(sig));   // si_signo
    mem_.write<uint32_t>(siginfo + 8, 1);                            // si_code: ILL_ILLOPC
    mem_.write<uint32_t>(siginfo + 12, fault_addr);                  // si_addr

    for (int i = 0; i < 15; ++i) mem_.write<uint32_t>(sc + kScRegsOff + i * 4, cpu_.r[i]);
    // The saved PC is the faulting instruction, not the one after it: a handler that
    // returns is meant to retry, and a probe that longjmps out never looks. Storing
    // `pc` -- which has already advanced -- makes a returning handler skip an
    // instruction, silently.
    mem_.write<uint32_t>(sc + kScRegsOff + 15 * 4, cpu_.cur_pc());
    mem_.write<uint32_t>(sc + kScCpsrOff, cpu_.cpsr());
    mem_.write<uint32_t>(sc + kScFaultOff, fault_addr);

    sig_frames_.push_back({frame, cpu_.save_regs()});
    if (trace)
        std::fprintf(stderr, "[sig] %d at %08X -> handler %08X (frame %08X)\n", sig,
                     fault_addr, h.func, frame);

    cpu_.set_sp(frame);
    cpu_.r[0] = static_cast<uint32_t>(sig);
    cpu_.r[1] = siginfo;
    cpu_.r[2] = uc;
    // Returning from the handler lands on the restorer, which issues rt_sigreturn.
    // A libc always supplies one; if a guest does not, point at a magic address the
    // CPU will trap on rather than letting it run off into whatever is there.
    cpu_.set_lr(h.restorer ? h.restorer : kSigreturnMagic);
    // An interworking branch, because a Thumb handler's address has bit 0 set and
    // entering it in ARM state executes its first two instructions as one.
    cpu_.bx_write_pc(h.func);
    // ITSTATE belongs to the interrupted instruction stream, and the handler starts
    // a new one. Leaving it set makes the handler's first few instructions
    // conditional on a comparison it never made.
    cpu_.itstate = 0;
    return true;
}

// rt_sigreturn: put back everything the frame saved. A probe never gets here -- it
// longjmps out -- but a real handler that returns must resume exactly where it was.
int64_t Syscalls::sys_rt_sigreturn() {
    if (sig_frames_.empty()) return -22;
    const SigFrame f = sig_frames_.back();
    sig_frames_.pop_back();
    const uint32_t sc = f.frame + kSiginfoSize + kUcMcontextOff;
    for (int i = 0; i < 15; ++i) cpu_.r[i] = mem_.read<uint32_t>(sc + kScRegsOff + i * 4);
    const uint32_t saved_cpsr = mem_.read<uint32_t>(sc + kScCpsrOff);
    cpu_.set_cpsr(saved_cpsr);
    cpu_.pc = mem_.read<uint32_t>(sc + kScRegsOff + 15 * 4);
    return static_cast<int64_t>(cpu_.r[0]);
}

// rt_sigaction: remember the handler. The struct is
//   { void *handler; unsigned long flags; void *restorer; sigset_t mask; }
// -- four-byte members on ARM, so the offsets are 0, 4, 8 rather than 0, 8, 16.
int64_t Syscalls::sys_rt_sigaction(int sig, uint32_t act, uint32_t oact) {
    if (sig < 0 || sig >= 64) return -22;
    if (oact) {
        mem_.write<uint32_t>(oact + 0, handlers_[sig].func);
        mem_.write<uint32_t>(oact + 4, handlers_[sig].flags);
        mem_.write<uint32_t>(oact + 8, handlers_[sig].restorer);
    }
    if (act) {
        handlers_[sig].func = mem_.read<uint32_t>(act + 0);
        handlers_[sig].flags = mem_.read<uint32_t>(act + 4);
        handlers_[sig].restorer = mem_.read<uint32_t>(act + 8);
    }
    return 0;
}

}  // namespace arm
