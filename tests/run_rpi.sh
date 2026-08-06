#!/bin/sh
# Raspberry Pi OS's own binaries: glibc, CPython 3.11 and gcc 12, exactly as
# Raspbian built them.
#
# This is the guest the project was aimed at, and it is a different one from Alpine's
# in two ways that the emulator has to actually satisfy rather than merely survive:
#
#   * **glibc, not musl.** glibc's ARM string routines are the parallel add/subtract
#     instructions -- `uqsub8` against zero to find a zero byte, `sel` to pick the
#     lane out of the GE flags. Those two only work as a pair, and only the
#     *non*-saturating forms write GE. An emulator that implements one and stubs the
#     other returns a string length that is merely somewhere else.
#   * **Raspbian is rebuilt for the original Pi**, so its code does not assume the
#     ARMv7 instructions the Alpine tree uses freely.
#
# `sh guests/fetch_rpi.sh` builds the tree (~250 MB unpacked).
set -e
cd "$(dirname "$0")/.."
EMU=${EMU:-./armemu}
ROOT=guests/rpi
[ -f "$ROOT/usr/bin/python3.11" ] || { echo "skip rpi (no $ROOT -- run: sh guests/fetch_rpi.sh)"; exit 0; }

pass=0; fail=0
check() {   # name, want, got
    if [ "$2" = "$3" ]; then echo "ok   $1"; pass=$((pass+1));
    else echo "FAIL $1"; echo "     want: $2"; echo "     got:  $3"; fail=$((fail+1)); fi
}
# MSYS rewrites a leading-slash argument into a Windows path before the emulator
# sees it. These paths are the *guest's*, so that has to be off.
run() { MSYS_NO_PATHCONV=1 $EMU --root $ROOT "$@" 2>&1 | tr -d '\015'; }

mkdir -p $ROOT/tmp
cp src/cpu.cpp $ROOT/tmp/cpu.cpp

# coreutils, which is where glibc's string routines get their workout.
check "rpi coreutils echo" "hello raspberry pi" "$(run /bin/echo hello raspberry pi)"
check "rpi coreutils sha256sum (against the host's)" \
      "$(sha256sum src/cpu.cpp | tr -d '\015' | cut -d' ' -f1)" \
      "$(run /usr/bin/sha256sum /tmp/cpu.cpp | cut -d' ' -f1)"
check "rpi coreutils wc -l" "$(wc -l < src/cpu.cpp | tr -d ' \015')" \
      "$(run /usr/bin/wc -l /tmp/cpu.cpp | tr -s ' ' | cut -d' ' -f1)"

# CPython 3.11, dynamically linked against that glibc.
check "rpi python version and machine" "3.11.2 armv7l" \
      "$(run /usr/bin/python3.11 -c 'import sys,platform;print(sys.version.split()[0],platform.machine())')"
want=$(python -c "import hashlib;print(hashlib.sha256(b'arm_emu_cpp').hexdigest())" 2>/dev/null \
       || python3 -c "import hashlib;print(hashlib.sha256(b'arm_emu_cpp').hexdigest())")
check "rpi python sha256 (against the host's)" "$want" \
      "$(run /usr/bin/python3.11 -c 'import hashlib;print(hashlib.sha256(b"arm_emu_cpp").hexdigest())')"

# gcc 12: the driver forks cc1, as and collect2/ld, and the binary it links runs
# under the same emulator that built it.
cat > $ROOT/t.c <<'PROG'
#include <stdio.h>
#include <string.h>
static long long fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
int main(void) {
    char buf[64];
    snprintf(buf, sizeof buf, "%lld %.3f", fib(24), 355.0 / 113.0);
    printf("%s %zu\n", buf, strlen(buf));
    return 0;
}
PROG
rm -f $ROOT/t $ROOT/t2
run /usr/bin/gcc-12 -O2 /t.c -o /t > /dev/null
if [ -f "$ROOT/t" ]; then echo "ok   rpi gcc compiles and links"; pass=$((pass+1));
else echo "FAIL rpi gcc compiles and links"; fail=$((fail+1)); fi
check "rpi gcc output runs under the emulator" "46368 3.142 11" "$(run /t)"

# Thumb-2, spelled out in full. Raspbian's gcc defaults to ARMv6, where plain
# `-mthumb` means Thumb-*1* -- and gcc itself refuses that with the hard-float ABI
# ("sorry, unimplemented: Thumb-1 'hard-float' VFP ABI"). Naming `armv7-a` alone then
# fails differently ("selected architecture lacks an FPU"), because an architecture
# name resets the FPU choice and the driver's hard-float ABI has nothing left to use.
# Both are compiler limitations, not emulator ones. `armv7-a+fp` is also the better
# test: it is the encoding a Debian armhf package actually contains.
run /usr/bin/gcc-12 -O2 -mthumb -march=armv7-a+fp /t.c -o /t2 > /dev/null
check "rpi gcc -mthumb (Thumb-2) output runs" "46368 3.142 11" "$(run /t2)"

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
