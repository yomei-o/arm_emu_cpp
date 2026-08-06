# Where this is, and what to do next

Working notes for picking the project back up. The README says what the emulator
*is*; this says what is unfinished and what is known about it.

## Handoff — start here

    sh build.sh
    sh tests/run_tests.sh        # 16 cases: 8 sources x {ARM, Thumb}
    # guests/README.md has the downloads for the three guest trees below
    sh tests/run_busybox.sh      # 10 cases, against the host's own md5sum/sha256sum
    sh tests/run_python.sh       #  6 cases, a stock dynamically linked CPython 3.12
    sh tests/run_gcc.sh          #  4 cases, gcc compiling and the result running

All four should be green (16 / 10 / 6 / 4). They are the whole quality argument, and
they are differential rather than expectational: the host computes the same answers
from the same sources, so "the emulator is right" means "it produced the host's
bytes".

**The goal.** A Raspberry Pi 32-bit ARM toolchain — `gcc`, `python3` and a `hello.c`
compiled by the first and run by neither — on an x86 host and in a browser tab. That
is the same shape the AArch64 sibling reached, one architecture generation back.

**Where this is (2026-08-06, first pass).** The goal it was aimed at is met, except
for the browser:

    ./armemu --root guests/sysroot /usr/bin/python3 \
        -c "import sys, platform; print(sys.version.split()[0], platform.machine())"
    3.12.13 armv7l

    ./armemu --root guests/gccroot /usr/bin/gcc /hello.c -o /hello
    ./armemu --root guests/gccroot /hello
    hello from emulated gcc

A stock CPython 3.12 for 32-bit ARM, dynamically linked, with its own
`ld-musl-armhf.so.1` doing every relocation; four guest threads on one lock coming
out at exactly 80,000; and gcc 13 forking cc1, as and ld and producing a binary that
runs under the same interpreter. About 490 million instructions for the CPython
digest case.

What is left is **WebAssembly**, and breadth in NEON.

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

## Two bugs the *guests* found, which the tests could not have

Both are in the file layer rather than the CPU, and neither is ARM-specific — they
are here because a real toolchain is the first thing that does the ordinary POSIX
manoeuvre in question.

- **`dup` shares a file *description*, not a handle.** The two descriptors share the
  position, and closing one must not close the file. Owning the `FILE*` from a single
  entry and nulling the copy's pointer looks right until a guest does
  `fd2 = dup(fd); close(fd)` — GNU ld does exactly that for every archive member it
  caches — and is left with a descriptor with nothing behind it. `gcc hello.c` failed
  at the link step with "Invalid seek". `Files::Entry::fp` is a `shared_ptr` now.

- **`(st_dev, st_ino)` has to be independent of how the path was spelled.** The inode
  is an FNV hash of the host path, so `tree/usr/lib` and `tree/usr/lib/.` were two
  different files. Everything that uses inode identity then gets a wrong answer that
  looks like something else: ld.so's "have I loaded this already", and getcwd(3)'s
  fallback, which walks upward comparing `stat(".")` against `stat("..")` and stops
  when they match. `Files::normalize` resolves the path lexically and clamps `..` at
  the guest root, because `--root` is a chroot.

## ⏭ Next, in order

1. **WebAssembly.** The last piece of the original goal. `web/wasm_api.cpp` and
   `web/build.sh` in the AArch64 sibling are close to directly portable; the only
   guest-visible difference is that there is no host filesystem, so the guest tree is
   unpacked into emscripten's in-memory FS and the `Files` layer works unchanged
   because `fopen` on a MEMFS path behaves like `fopen` anywhere else. The guest to
   ship first is busybox (880 KB) and then the CPython sysroot (~60 MB, which the
   sibling serves as a compressed tarball the page decompresses with
   `DecompressionStream`).

2. **Raspberry Pi OS's own binaries**, as opposed to Alpine's. Those are glibc and
   ARMv6, which exercises two things musl and ARMv7 do not: glibc's IFUNC resolvers
   pick string routines from AT_HWCAP (so the HWCAP word in `elf_loader.cpp` becomes
   load-bearing rather than decorative), and ARMv6 code reaches the kernel helper page
   for its atomics instead of using LDREX/STREX directly.

3. **Speed.** ~490 M instructions for the CPython digest case, which is a few tens of
   seconds. A decode cache keyed on the PC is the standing idea; measure first.

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
