#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "memory.h"

namespace arm {

// What loading an image produced. One structure for both the program and its
// interpreter, because the kernel treats them the same way: map the PT_LOADs,
// remember where the headers landed, and hand the addresses to the guest through
// the auxiliary vector.
struct LoadedImage {
    uint32_t base = 0;          // where an ET_DYN image was placed; 0 for ET_EXEC
    uint32_t entry = 0;
    uint32_t phdr_addr = 0;     // AT_PHDR: the program headers, in guest memory
    uint32_t phent = 0, phnum = 0;
    uint32_t brk = 0;           // just past the last PT_LOAD, page-aligned
    std::string interp;         // PT_INTERP, empty for a static binary
    // Thumb entry points are marked the way every ARM branch target is: bit 0 of
    // the address. A Thumb-only image (which is what a Cortex-M-style build or a
    // `-mthumb` musl produces) therefore starts with e_entry odd, and starting it
    // in ARM state executes its first instruction as half of something else.
    bool thumb_entry = false;
};

bool load_elf(const std::vector<uint8_t>& f, Memory& mem, uint32_t base,
              LoadedImage* out, std::string* err);

// The initial process image Linux hands a new program: argv, envp and the
// auxiliary vector, laid out on the stack in that order. Returns the SP the guest
// starts with.
uint32_t build_stack(Memory& mem, uint32_t stack_top, const LoadedImage& img,
                     uint32_t interp_base,
                     const std::vector<std::string>& argv,
                     const std::vector<std::string>& envp);

// The kernel helper page, mapped at 0xFFFF0000 in every ARM Linux process. Written
// as real ARM code rather than intercepted in the interpreter -- see the note in
// elf_loader.cpp.
void map_kuser_helpers(Memory& mem);

}  // namespace arm
