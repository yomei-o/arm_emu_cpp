#pragma once
#include "cpu.h"
#include "loader.h"
#include "memory.h"
#include "syscalls.h"

namespace arm {

// Everything the host has to do around a guest, shared by the two front ends.
//
// It exists as a function rather than as code in `main.cpp` because the two front
// ends drifted in the sibling project: the command-line one grew each piece of
// start-up as it was discovered and the WebAssembly one grew none of it, so the
// browser build ran a freestanding binary and not a real one, and nothing said so.
void prepare_process(Cpu& cpu, Memory& mem, const LoadedImage& img, uint32_t start_pc,
                     bool interp_is_thumb);

// Run to completion and return the exit code.
int run_image(Cpu& cpu, Memory& mem, Syscalls& sys);

}  // namespace arm
