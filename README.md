# arm_emu_cpp

A user-mode **32-bit ARM** emulator in C++17. It loads an ARM Linux ELF, interprets
the machine code instruction by instruction in both instruction sets — A32 and
Thumb-2 — and services the guest's syscalls on the host, so a Raspberry Pi binary
runs on x86 Windows, on x86 Linux, and in a browser tab.

```console
$ ./armemu tests/hello.elf
hello from arm

$ ./armemu --root . guests/busybox sha256sum src/cpu.cpp
f87b97e603d331b364196680d0752f1c2b669481423ee744705602dc06a837f2  src/cpu.cpp

$ sha256sum src/cpu.cpp
f87b97e603d331b364196680d0752f1c2b669481423ee744705602dc06a837f2 *src/cpu.cpp
```

That is Alpine's static `armv7` busybox — a Thumb-2 binary that has never heard of
this emulator — and the digest is the host's digest of the same bytes.

Sibling of [aarch64_emu_cpp](https://github.com/yomei-o/aarch64_emu_cpp), pointed one
architecture generation back. The goal is **a Raspberry Pi 32-bit toolchain — gcc,
CPython and a `hello.c` compiled by the first and run by neither — on a machine that
is not an ARM**, locally and in WebAssembly.

No dependencies beyond a C++17 standard library.

## Why 32-bit ARM is a different job from AArch64

A64 is a clean field layout: four bytes per instruction, one encoding space, no
condition field, and a program counter that is not a register. A32 is none of those
things, and three differences are where every silent bug in a 32-bit ARM interpreter
comes from.

**R15 is the program counter, and reading it is not reading a register.** In ARM
state R15 reads as the current instruction's address **+ 8**; in Thumb state, **+ 4**
— regardless of whether the Thumb instruction is two bytes or four. That offset is
the pipeline the architecture froze into its ABI, and every PC-relative address in
every real binary depends on it. It exists in exactly one function here, `pc_val()`,
and no handler touches `r[15]`.

**Writing R15 branches, and what kind of branch depends on which instruction and
which state.** A load into the PC always interworks — bit 0 selects Thumb, which is
what makes `pop {pc}` into a Thumb function work with no special case. But a
*data-processing* write to the PC interworks **only in ARM state**; in Thumb it is a
plain branch that stays in Thumb. That distinction is not academic:

```
adr  r2, table
add  r3, r2, n, lsl #4
mov  pc, r3          @ every entry in the table is at an even address
```

is how musl writes `__aeabi_uidiv`, and every entry in that jump table is even
because Thumb instructions are two-byte aligned and nothing marks them. Interworking
there reads bit 0 as "ARM" and the machine starts decoding a table of Thumb
instructions four bytes at a time. `busybox md5sum` ran 800 million instructions into
nowhere before that one was found.

**Conditional execution is carried out of band in Thumb.** Sixteen-bit Thumb has no
condition field, so `IT` loads up to four conditions into ITSTATE and the following
instructions consume them one at a time — and whether a 16-bit instruction *writes
the flags* is decided by the same state, because `movs` and `mov` are the same
encoding inside and outside an IT block. ITSTATE has to be advanced before the
instruction runs and remembered as it was, or the last instruction of every block
sets the flags:

```
ite ls
movls r3, #1
movhi r3, #0        @ this one wrote Z
bhi   done          @ ...and this one read it
```

That is a scan loop in musl's `strtod` that never terminates.

## What runs today

| | status |
| --- | --- |
| A32 integer core: data-processing, the barrel shifter, multiply, divide | ✅ |
| Load/store: all widths and sign extensions, pre/post-index, register offset, LDRD/STRD, LDM/STM in all four modes | ✅ |
| Branches, `BX`/`BLX` and interworking in both directions | ✅ |
| The exclusive monitor: `LDREX`/`STREX` and the byte, halfword and doubleword forms | ✅ |
| Media instructions: bitfield, saturating, packing, extends, `REV`/`RBIT`/`CLZ` | ✅ |
| **Thumb and Thumb-2**, 16- and 32-bit encodings, IT blocks, `TBB`/`TBH` | ✅ |
| **VFPv3/v4** — the hard-float ABI's arithmetic, compares, conversions, `VMRS` | ✅ |
| The **kernel helper page** at `0xFFFF0000`, as real ARM code | ✅ |
| CP15: the thread-pointer registers and the cache-type register | ✅ |
| Static ELF32 loading with a Linux-shaped stack and auxiliary vector | ✅ |
| Linux ARM **EABI** syscalls — the legacy numbering, `mmap2`, `stat64` | ✅ |
| Signal delivery (`SIGILL` frames, `rt_sigreturn`) | ✅ |
| Threads — `clone`, `futex`, a scheduler | ✅ |
| Child processes — `fork`, `execve`, `wait4`, pipes | ✅ |
| A real guest: **Alpine's static armv7 busybox** | ✅ |
| Advanced SIMD (NEON) | grown by demand: load/store, the three-same group, the immediates, shifts, the two-register misc group |
| Dynamic linking through the guest's own `ld.so` | ⏳ loader is in, not yet exercised |
| CPython for armhf | ⏳ |
| Raspberry Pi's gcc | ⏳ |
| WebAssembly | ⏳ |

## Correctness is a diff, not an opinion

Every test is built **three times** from one source: once for ARM state, once for
Thumb state, and once for the host. The emulator runs the first two, the host runs
the third, and all three are compared byte for byte.

The two ARM builds are not thoroughness for its own sake. They are different
instruction sets encoding the same operations, and a real armhf system contains both
— which is why every bug listed above was found in exactly one of them.

```console
$ sh tests/run_tests.sh
ok   arm   arith
ok   thumb arith
ok   arm   atomic
ok   thumb atomic
ok   arm   control
ok   thumb control
ok   arm   file
ok   thumb file
ok   arm   fp
ok   thumb fp
ok   arm   hello
ok   thumb hello
ok   arm   interwork
ok   thumb interwork
ok   arm   mem
ok   thumb mem
16 passed, 0 failed
```

`interwork` is the one worth a second look: two mutually recursive functions, one
forced into ARM state and one into Thumb, so every call and every return crosses.

A second suite runs a **real** guest against the host's own tools:

```console
$ sh tests/run_busybox.sh
ok   busybox md5sum
ok   busybox sha1sum
ok   busybox sha256sum
ok   busybox wc -l
ok   busybox uname
ok   busybox expr
ok   busybox seq
ok   busybox sort -u
ok   busybox od
ok   busybox awk
10 passed, 0 failed
```

A digest is the strongest single check available: every byte of the file travels
through the emulated CPU, and one wrong bit anywhere in a hundred million
instructions changes the answer.

The arithmetic test is deliberately written in 64-bit types. A 32-bit ARM has no
64-bit register, so every line becomes a *sequence* — `ADDS`/`ADC`, `SUBS`/`SBC`,
`UMULL`/`MLA`, a shift built from two shifts and an `ORR` — and each of those depends
on the carry flag being exactly what the architecture says. On AArch64 the same
source compiles to one instruction per line and tests almost nothing.

## Building

```sh
sh build.sh                              # g++ (or CXX=clang++)
CXX=cl sh build.sh                       # MSVC, from a developer prompt
cmake -B build && cmake --build build    # either, through CMake
```

Tests need `clang` with the ARM target and `lld` — no cross-binutils and no sysroot,
because the test programs are freestanding. The two runtime helpers a freestanding
ARM build genuinely cannot do without — `__aeabi_ldivmod` and `__aeabi_uldivmod`,
because ARM has no 64-bit divide instruction — are in `tests/harness.h`.

`guests/README.md` has the one command that fetches busybox.

## Design

- `src/cpu.{h,cpp}` — the A32 interpreter, laid out in the order the ARM ARM's A5.1
  encoding tables are: bits 27..25 pick the group, each group sub-decodes its own
  fields, and the function names are the section titles.
- `src/thumb.cpp` — T16 and T32. Only the *encodings* live here; everything that
  computes is shared with the A32 file, because the two instruction sets encode the
  same operations differently and execute them identically.
- `src/vfp.cpp` — VFPv3/v4. Not optional for this target: Raspberry Pi OS is
  `armhf`, so every `double` crosses a function boundary in a VFP register and a
  libm that cannot do arithmetic returns whatever was left in `d0`.
- `src/neon.cpp` — Advanced SIMD, grown strictly by demand. An unimplemented
  encoding **stops the machine and prints it**; it never silently does nothing,
  because a guest that keeps running on a stale register fails somewhere unrelated
  and hours later.
- `src/memory.h` — 64 KiB pages in a hash map. A 32-bit guest would fit in one flat
  3 GiB array, and that is exactly why it is not one: a hello world touches a few
  dozen pages.
- `src/elf_loader.cpp` — ELF32, the initial process image, and the kernel helper
  page.
- `src/syscalls.cpp` — the kernel interface, in ARM's own legacy numbering (write is
  4, not 64).

## Two things about the ARM syscall interface that bite

**The number is in r7, and the table is the original Linux one.** ARM kept it rather
than taking the "generic" table AArch64 was given, and it carries thirty years of
history: `stat64` exists alongside `stat` because inodes outgrew 32 bits, `mmap2`
exists alongside `mmap` because file offsets did, and `ugetrlimit` exists because the
first `getrlimit` got the type wrong.

**`mmap2` takes its offset in 4096-byte pages, not bytes.** A guest mapping a library
at offset 0x1000 means byte 0x1000000. Getting it wrong maps the wrong part of the
file, which reads as a corrupt library rather than as an arithmetic mistake.

## The kernel helper page is real code

ARM Linux maps a page at `0xFFFF0000` in every process containing three routines any
program may call: an atomic compare-and-swap, a memory barrier, and a read of the
thread pointer. They exist because those operations need instructions that are not on
every ARM the kernel supports — and Raspberry Pi's own ARMv6 toolchain output calls
them rather than using the instructions.

They are written here as **real ARM code**, not intercepted in the interpreter. That
is the better trade twice over: the fast path costs nothing, and what runs is the
kernel's own implementation, so the compare-and-swap goes through this emulator's
exclusive monitor exactly as a guest's own would.
