# Where this is, and what to do next

Working notes for picking the project back up. The README says what the emulator
*is*; this says what is unfinished and what is known about it.

## Handoff — start here

    sh build.sh
    sh tests/run_tests.sh        # 16 cases: 8 sources x {ARM, Thumb}
    # guests/README.md has the one curl that fetches busybox
    sh tests/run_busybox.sh      # 10 cases, against the host's own md5sum/sha256sum

Both should be green. They are the whole quality argument so far, and they are
differential rather than expectational: the host computes the same answers from the
same sources, so "the emulator is right" means "it produced the host's bytes".

**The goal.** A Raspberry Pi 32-bit ARM toolchain — `gcc`, `python3` and a `hello.c`
compiled by the first and run by neither — on an x86 host and in a browser tab. That
is the same shape the AArch64 sibling reached, one architecture generation back.

**Where this is (2026-08-06, first pass).** The CPU is done in the sense that
matters: Alpine's static armv7 busybox, a Thumb-2 binary end to end, produces the
host's digests over a few hundred million instructions. What is left is breadth, not
a wall.

## Bugs from this stretch worth not rediscovering

Every one of these failed *silently*, and every one was found by a diff rather than
by a crash.

- **`ALUWritePC` interworks only in ARM state.** In Thumb, a data-processing write to
  the PC is a plain branch. musl's `__aeabi_uidiv` is a jump table entered with
  `mov pc, r3`, and every entry is at an even address — so interworking there drops
  into ARM state and decodes a table of Thumb instructions four bytes at a time.
  `busybox md5sum` ran 800 million instructions into zeroed memory before this one
  was found. `LoadWritePC` — a load or an LDM into the PC — *does* interwork in both
  states, so the two cases go through different functions (`set_reg` vs
  `set_reg_alu`) and never through a shared one.

- **ITSTATE has to be remembered as it was, not as it becomes.** It is advanced
  before the instruction executes, so by the time a handler asks "am I in an IT
  block" the last instruction of a block already reads as outside one — and 16-bit
  Thumb decides whether to write the flags from exactly that question. `ite ls /
  movls / movhi / bhi` left Z from the MOV and the BHI read it: musl's `strtod` scan
  loop never terminated, and `busybox seq 1 5` printed nothing at all. `Cpu::in_it`
  is the fix and the comment on it is the explanation.

- **The 16-bit Thumb group mask is 0x38, not 0x30.** `00xxx` is the shifts and the
  three-operand adds; `001xx` is the eight-bit immediate group. One test catching
  both sends `subs r3, #4` into the shift decoder, where it becomes a rotate: a
  plausible number, and a loop counter that never reaches its terminating value.

- **BL and B differ by one bit of `op1`, not by its value.** `(op1 & 5) == 5` is BL,
  `== 1` is B, `== 4` is BLX. Testing for the exact values 5 and 7 catches half of
  each, and a `bl` executed as a `b` does not fault — the callee runs correctly and
  returns through whatever was already in LR.

- **BLX (immediate) must compute its target before switching state.** `pc_val()`
  answers +8 in ARM and +4 in Thumb, so clearing or setting `thumb` first moves the
  branch by four bytes — which lands inside the callee, on a real instruction, and
  runs.

- **The Thumb exclusive group is `op1 == 01`, not `00`.** The word-sized
  STREX/LDREX are `op1 == 00`; every byte, halfword and doubleword form plus TBB/TBH
  is `op1 == 01`, split by `op2`. Getting it wrong decodes every sub-word atomic as
  an LDRD, which does not fault: it loads a register pair from the lock's address and
  the compare-and-swap loop spins forever.

- **T32's extend group selects on `op1` directly** (0 SXTH, 1 UXTH, 2 SXTB16,
  3 UXTB16, 4 SXTB, 5 UXTB), unlike the shift group just above it where the low bit
  is the S flag. Shifting op1 right by one for both turns `uxtb16` into `uxth` — a
  mask of 0x0000FFFF where the guest asked for 0x00FF00FF.

- **A32's SBFX and UBFX are keyed on bits 24..21**, because bit 20 belongs to the
  width field. `(op1 & 0x1E) == 0x1A` catches SBFX and misses UBFX entirely.

- **A32's extend family: the width is the low bit of op1 and the signedness is bit
  2.** Reading the signedness from bit 22 and the width from nowhere makes
  `(uint8_t)(h >> 8)` a halfword extract — a plausible byte until the input has
  anything in bits 15..8.

- **Windows opens the standard streams in text mode**, which deletes a CR before
  every LF on the way in. A guest hashing its standard input then digests *different
  bytes* from the ones the host would: a wrong answer that looks exactly like an
  emulator bug and is not one. `main.cpp` sets all three to binary.

## Two habits this project runs on

- **Build every test three times.** ARM, Thumb, and the host. Every bug above was
  present in exactly one of the two ARM builds, and a suite that built only one would
  have reported green.

- **Refuse the unknown; never no-op.** An unimplemented encoding stops the machine
  and prints itself, or becomes SIGILL if the guest installed a handler. A guest that
  keeps running on a stale register fails somewhere unrelated and hours later.

## The tools built for this

    --trace-sys        every syscall, with the path for opens and stats
    --sample N         print the PC every N instructions, with A/T for the state
    --watch LO:HI      log every guest access in a range, with the PC that made it
    --pcwatch ADDR     r0..r3 with any strings they point at, at a function's entry
    --strict           an access to a page nothing mapped is a fault, not a zero
    --max N            stop after N instructions
    --stats            instructions and pages on exit

`--sample` is the one that found four of the bugs above: a guest that stops making
progress looks exactly like one doing a lot of work, and sampling tells them apart in
seconds. When the PC is climbing by four per instruction through zeroed memory, the
last in-image PC in the trace is the instruction that went wrong.

## ⏭ Next, in order

1. **Dynamic linking.** The loader already maps `PT_INTERP` and starts at the
   interpreter's entry with AT_BASE set, exactly as the kernel does — the same path
   the AArch64 sibling uses, where the guest's own `ld.so` does all the relocation.
   It has not been exercised yet. The first guest to try is Alpine's dynamic busybox
   with `ld-musl-armhf.so.1`, because it is the same program that already works
   statically, so any difference is the loader's.

2. **CPython for armhf.** The sibling project's route was
   `python-build-standalone`'s musl build, which is dynamically linked and therefore
   depends on step 1. Expect a round of unimplemented NEON: `neon.cpp` is grown by
   demand and stops with the encoding printed, which is the address to go implement.

3. **Raspberry Pi's gcc.** Alpine's armv7 packages give a real toolchain with no Pi
   and no cross-compiler; the AArch64 sibling's `resume.md` has the package list and
   the two traps (Windows `tar` drops the symlinks, so every soname has to be a
   *copy*; and cc1 installs a SIGILL handler, so an unimplemented instruction becomes
   "internal compiler error: Illegal instruction" with no PC — `--trace-sys` prints
   the `[sig]` line and the faulting PC in it is what to implement).

4. **WebAssembly.** `web/wasm_api.cpp` and `web/build.sh` in the sibling are close to
   directly portable; the only guest-visible difference is that there is no host
   filesystem, so the guest tree is unpacked into emscripten's in-memory FS.

## Also still open

- **NEON is partial.** Implemented: the element and structure load/stores, the
  three-registers-same group, the one-register modified immediates, the shift-by-
  immediate group, the two-register miscellaneous group, VEXT, and the widening
  three-different forms. Not implemented, and refused loudly: the parallel add/sub
  groups, the saturating shifts, VTBL/VTBX, the polynomial multiplies, and the
  by-scalar forms.
- **A32's parallel add/sub (SADD8, UHSUB16 and the rest) is refused.** Nothing has
  needed it; glibc's string routines use NEON instead.
- **`fork` is vfork semantics.** The parent is suspended until the child execs or
  exits, and the child's writes are journalled and rolled back so the parent's memory
  is genuinely unaffected. A child that runs *concurrently* with its parent is not
  supported and is reported rather than faked — see the note in `process.cpp`.
- **Speed.** No decode cache; the interpreter re-decodes every instruction. Measure
  before building one.
- **VCVT between half and single precision, and the fixed-point VCVTs**, return false
  from `exec_vfp` and therefore stop the machine. Nothing has needed them.
