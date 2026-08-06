#!/bin/sh
# The whole toolchain, inside the emulator: gcc's driver forks and execs cc1, as
# and collect2/ld -- four child processes, pipes and temporary files -- and the
# binary it links **runs under the same emulator that built it**.
#
# That last part is what makes this a closed loop rather than a demo. A compiler
# that produces plausible-looking output proves very little; a compiler whose output
# runs correctly on the same interpreter proves the code generator and the
# interpreter agree about the instruction set.
set -e
cd "$(dirname "$0")/.."
EMU=${EMU:-./armemu}
ROOT=guests/gccroot
GCC=$ROOT/usr/bin/gcc
[ -f "$GCC" ] || { echo "skip gcc (no $GCC -- see guests/README.md)"; exit 0; }

pass=0; fail=0
check() {   # name, want, got
    if [ "$2" = "$3" ]; then echo "ok   $1"; pass=$((pass+1));
    else echo "FAIL $1"; echo "     want: $2"; echo "     got:  $3"; fail=$((fail+1)); fi
}
# MSYS rewrites a leading-slash argument into a Windows path before the emulator
# sees it. These paths are the *guest's*, so that has to be off.
run() { MSYS_NO_PATHCONV=1 $EMU --root $ROOT "$@" 2>&1 | tr -d '\015'; }

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

rm -f $ROOT/t $ROOT/t.s
run $GCC -O2 /t.c -o /t > /dev/null
if [ -f "$ROOT/t" ]; then echo "ok   gcc compiles and links"; pass=$((pass+1));
else echo "FAIL gcc compiles and links"; fail=$((fail+1)); fi

# The output of the emulated compiler, run by the emulator that built it.
check "gcc output runs under the emulator" "46368 3.142 11" "$(run /t)"

# -S stops before the assembler, so this is cc1 alone, and the answer is text: a
# code generator that drifts changes it.
run $GCC -O2 -S /t.c -o /t.s > /dev/null
if grep -q "fib" $ROOT/t.s 2>/dev/null; then echo "ok   gcc -S produces assembly"; pass=$((pass+1));
else echo "FAIL gcc -S produces assembly"; fail=$((fail+1)); fi

# Thumb as well as ARM, because the emulator has to run both and so does gcc.
rm -f $ROOT/t2
run $GCC -O2 -mthumb /t.c -o /t2 > /dev/null
check "gcc -mthumb output runs" "46368 3.142 11" "$(run /t2)"

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
