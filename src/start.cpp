// What the host does around a guest, shared by the command-line and WebAssembly
// front ends.
#include "start.h"

namespace arm {

void prepare_process(Cpu& cpu, Memory& mem, const LoadedImage& img, uint32_t start_pc,
                     bool interp_is_thumb) {
    // The kernel helper page. Every ARM Linux process has one, and a guest built for
    // ARMv6 -- which is what Raspberry Pi OS's own toolchain produces by default --
    // reaches its atomics and its thread pointer through it rather than through the
    // instructions.
    map_kuser_helpers(mem);

    // Which instruction set the first instruction is in. It is carried in bit 0 of
    // the entry address, exactly as it is for every other branch target on ARM, and
    // a Thumb entry started in ARM state executes its first two instructions as one
    // -- which does not fault, it computes.
    if (start_pc) {
        cpu.thumb = interp_is_thumb;
        cpu.pc = start_pc;
    } else {
        cpu.thumb = img.thumb_entry;
        cpu.pc = img.entry;
    }
    // The AAPCS says r0..r3 are undefined at process entry and a libc must not read
    // them -- but glibc's `_start` passes r0 to `__libc_start_main` as the "rtld
    // fini" pointer and calls it if it is non-zero. A left-over value there is a
    // call through a wild pointer before `main`.
    for (int i = 0; i < 13; ++i) cpu.r[i] = 0;
    cpu.set_lr(0);
}

int run_image(Cpu& cpu, Memory& mem, Syscalls& sys) {
    (void)mem;
    // A vfork child that execs or exits asks the run loop to stop; this is what puts
    // the parent back so the same loop can carry on with it.
    cpu.on_stop_requested = [&sys] { sys.vfork_resume(); };
    cpu.run();
    return cpu.exit_code;
}

}  // namespace arm
