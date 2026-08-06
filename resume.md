# Where this is, and what to do next

Working notes for picking the project back up. The README says what the emulator
*is*; this says what is unfinished and what is known about it.

## Handoff — start here

    sh build.sh
    sh tests/run_tests.sh        # 18 cases: 9 sources x {ARM, Thumb}
    # guests/README.md has the downloads for the three guest trees below
    sh tests/run_busybox.sh      # 10 cases, against the host's own md5sum/sha256sum
    sh tests/run_python.sh       #  6 cases, a stock dynamically linked CPython 3.12
    sh tests/run_gcc.sh          #  4 cases, gcc compiling and the result running
    sh tests/run_rpi.sh          #  9 cases, Raspberry Pi OS's own glibc/python/gcc
    node web/test_node.mjs       #  7 cases, the same guests inside WebAssembly
    node web/test_pages.mjs      #  6 cases, the same again from the deployed URL

All seven should be green (18 / 10 / 6 / 4 / 9 / 7 / 6). They are the whole quality argument, and
they are differential rather than expectational: the host computes the same answers
from the same sources, so "the emulator is right" means "it produced the host's
bytes".

**The goal.** A Raspberry Pi 32-bit ARM toolchain — `gcc`, `python3` and a `hello.c`
compiled by the first and run by neither — on an x86 host and in a browser tab. That
is the same shape the AArch64 sibling reached, one architecture generation back.

**Where this is (2026-08-06, first pass).** The goal it was aimed at is met:


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

And in a browser tab, at <https://yomei-o.github.io/arm_emu_cpp/>: `web/` is the same
emulator as WebAssembly, running the freestanding tests in both instruction sets,
busybox, and that same CPython -- four guest pthreads and all, on one wasm instance
with no SharedArrayBuffer.

`web/test_pages.mjs` checks the *deployed* page rather than the local build: it
fetches `armemu.js` and both tarballs from the live URL and puts them through the
exact code path `index.html` uses. That is what catches the failures a static
deployment actually has -- a missing asset, or a server that sets
`Content-Encoding: gzip` on a `.tar.gz`, which makes `fetch` decompress it and
`DecompressionStream` then choke on plain bytes.

Raspberry Pi OS's own tree works too -- glibc 2.36, CPython 3.11.2 and gcc 12 out of
Raspbian's .debs -- which was the literal target:

    ./armemu --root guests/rpi /usr/bin/gcc-12 /hello.c -o /hello
    ./armemu --root guests/rpi /hello
    hello from raspberry pi gcc

What is left is breadth in NEON, and speed.

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

## What the Raspberry Pi tree needed that Alpine's did not

Three things, and the first is the one worth remembering.

- **The parallel add/subtract group is not optional once glibc is the libc.** ARMv6
  predates NEON, and glibc's ARM string routines still work the way ARMv6 had to:
  `uqsub8` against a register of zeros leaves a non-zero byte exactly where the input
  had one, and `sel` picks the lane out of the GE flags. Refusing the group stopped
  `/bin/echo` 12,890 instructions in, inside `ld.so`'s own `strlen`. Emulating one of
  the pair without the other would have been worse: only the *non*-saturating forms
  write GE, so a `sel` a few instructions later picks the wrong bytes and a string
  routine returns a length that is merely somewhere else.

- **`close(1)` on a descriptor that was never redirected has to succeed.**
  coreutils' `close_stdout` runs `fclose(stdout)` at exit and reports whatever it
  gets, so EBADF made every one of them print "write error: Bad file descriptor"
  *after* its correct output, and exit 1.

- **`sched_getaffinity` must return the size and set a bit.** Zero reads as "a
  machine with no processors": `sysconf(_SC_NPROCESSORS_ONLN)` counts the bits,
  CPython reports `os.cpu_count()` as None, and anything sizing a pool from it makes
  a pool of zero.

Two gcc diagnostics on that tree are the *compiler's* limits and not the emulator's,
and both cost a few minutes to recognise: Raspbian's gcc defaults to ARMv6, so plain
`-mthumb` means Thumb-1 and gcc refuses it with the hard-float ABI; naming `armv7-a`
to fix that then resets the FPU choice and fails with "selected architecture lacks an
FPU". `-mthumb -march=armv7-a+fp` is what a Debian armhf package actually contains.

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

1. **Speed.** ~490 M instructions for the CPython digest case, which is a few tens of
   seconds. A decode cache keyed on the PC is the standing idea; measure first.

## Also still open

- **`tests/parallel.c` had a bug of its own worth remembering**, because it is the
  kind a differential test cannot catch by itself: the host build called
  `show(name, par(...), g_ge)` where `par()` is what sets `g_ge`, and C does not
  specify which argument is evaluated first. Half the cases printed the *previous*
  case's flags, the two builds disagreed by one line, and the diff read as a missing
  instruction. A reference that computes and reports in one expression is a reference
  that can be wrong in a way the emulator cannot be.

- **NEON: what is still refused.** The saturating and rounding shifts (VQSHL,
  VRSHL, VQRSHL, VQSHRN), the saturating doubling multiplies (VQDMULH, VQDMLAL and
  friends), the polynomial multiplies, the by-scalar forms (A7.4.6), VRECPE/VRSQRTE,
  the half-precision converts, and 64-bit lanes for the saturating and halving adds.
  Each returns false and stops the machine with its encoding printed, which is the
  address to go implement.

## How the NEON gaps were found, and how to find the next ones

Not by reading the manual: by making the guest's own compiler emit them.
`tests/run_rpi.sh` builds one source twice with the emulated gcc -- `-O3 -mfpu=neon
-ftree-vectorize` and `-O1 -fno-tree-vectorize` -- and requires the two runs to
agree. gcc chooses the instructions, so coverage follows what a real compiler emits,
and the scalar build is an oracle needing no host compiler.

Adding a loop shape to `vec.c` inside that script is how to reach further. The four
bugs it found on its first run were all silent:

- **The three-registers-same-length group masked out bit 24**, which is `U`. The
  signed half of NEON worked and the unsigned half was refused -- so the failure
  looked like a missing instruction rather than a wrong group test.
- **VQADD/VQSUB did not saturate and VHADD/VHSUB ignored signedness.** `vqsub.u8` is
  how a compiler writes `a > b ? a - b : 0`; a wrapping subtract turns the clamp into
  255-and-counting.
- **VBSL read `Vn` as its mask instead of the destination.** VBSL's selector *is* the
  destination register, so reading the wrong operand inverts every select -- the
  difference appeared exactly where zero should have been.
- **VMOVL halved its element size.** `imm6` already names the source element, so
  deriving a "narrower" one from it widened bytes where halfwords were asked for.

And one from the same session in VFP: the **fixed-point VCVT** was refused. Its
immediate is not the fraction width -- it is that width subtracted from the container
size, which is why the encoding reads as a large number for a small scale.

The two-registers-miscellaneous (A, B) table was read out of the *assembler* rather
than remembered: assemble each mnemonic, read the bits back. That group is a dozen
operations picked by two small fields with no pattern, and the sibling project lost
an evening to a zero-compare table whose opcodes were each off by one.
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
