#!/bin/sh
# Builds each test three times -- once for ARM state, once for Thumb state, once for
# the host -- runs the first two under the emulator and the third directly, and
# diffs. A test passes only if the emulator reproduces the host byte for byte.
#
# The two ARM builds are not thoroughness for its own sake. They are different
# instruction sets encoding the same operations, and a real armhf system contains
# both -- glibc is built for ARM, but its string routines and anything from a
# `-mthumb` package are not. Every decoder bug this project has found was present in
# exactly one of the two, and a suite that built only one would have reported green.
set -e
cd "$(dirname "$0")"
EMU=${EMU:-../armemu}
CLANG=${CLANG:-clang}
HOSTCC=${HOSTCC:-cc}
ARM="--target=armv7-unknown-linux-gnueabihf -mcpu=cortex-a7 -ffreestanding -nostdlib -fuse-ld=lld -static"
pass=0; fail=0

# Both sides are compared with carriage returns stripped. The emulator puts its
# stdout in binary mode on purpose -- a guest that hashes its input has to see the
# bytes the host would -- while the native oracle is an ordinary Windows program
# whose newlines grow a CR on the way out. Normalising here keeps both of those
# correct instead of trading one for the other.
run_one() {   # name, elf, label
    if $EMU "$2" > "$1.actual" 2>"$1.err"; then :; else
        echo "FAIL $3 $1 (emulator stopped)"; sed 's/^/     /' "$1.err"; fail=$((fail+1)); return
    fi
    tr -d '\r' < "$1.expected" > "$1.want"
    tr -d '\r' < "$1.actual"   > "$1.got"
    if cmp -s "$1.want" "$1.got"; then
        echo "ok   $3 $1"; pass=$((pass+1))
    else
        echo "FAIL $3 $1"; diff "$1.want" "$1.got" | head -10; fail=$((fail+1))
    fi
}

for src in *.c; do
    name=${src%.c}
    $CLANG $ARM -marm   -O2 -I. -o "$name.elf"  "$src"
    $CLANG $ARM -mthumb -O2 -I. -o "$name.telf" "$src"
    $HOSTCC -DARM_NATIVE -O2 -I. -o "$name.native" "$src"
    "./$name.native" > "$name.expected"
    run_one "$name" "$name.elf"  "arm  "
    run_one "$name" "$name.telf" "thumb"
done
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
