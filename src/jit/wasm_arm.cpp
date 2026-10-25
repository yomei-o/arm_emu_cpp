// The ARM guest has no ops of its own that the wasm tier translates yet: its flag ops
// run through their threaded handlers (jit_generic), and the generic ops - moves,
// arithmetic, loads and stores - are emitted by the shared code.
#include "wasm_gen.h"

namespace jit {

bool guest_wasm_emit(BlockGen&) { return false; }
int guest_wasm_cells(const Op&) { return 0; }
void guest_wasm_extras(std::vector<Buf>&) {}

}  // namespace jit
