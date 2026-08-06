#!/bin/sh
# A *real* guest: Alpine's static armv7 busybox, a binary that has never heard of
# this emulator, checked against the host's own tools.
#
# A digest is the strongest single check available. Every byte of the file travels
# through the emulated CPU, and one wrong bit anywhere in a hundred million
# instructions changes the answer -- which is a claim no hand-written test makes.
#
# busybox is also a Thumb-2 binary end to end, so this is where the IT blocks, the
# jump tables and musl's hand-written division get exercised for real.
set -e
cd "$(dirname "$0")/.."
EMU=${EMU:-./armemu}
BB=guests/busybox
[ -f "$BB" ] || { echo "skip busybox (no $BB -- see resume.md for the download)"; exit 0; }

pass=0; fail=0
check() {   # name, want, got
    if [ "$2" = "$3" ]; then echo "ok   $1"; pass=$((pass+1));
    else echo "FAIL $1"; echo "     want: $2"; echo "     got:  $3"; fail=$((fail+1)); fi
}
strip_cr() { tr -d '\015'; }

# The file every digest is taken over: the emulator's own source, which is a few
# hundred kilobytes of text and changes as the project does.
SRC=src/cpu.cpp

for algo in md5sum sha1sum sha256sum; do
    want=$($algo < "$SRC" | strip_cr | cut -d' ' -f1)
    got=$($EMU --root . $BB $algo < "$SRC" 2>/dev/null | strip_cr | cut -d' ' -f1)
    check "busybox $algo" "$want" "$got"
done

want=$(wc -l < "$SRC" | tr -d ' \015')
got=$($EMU --root . $BB wc -l < "$SRC" 2>/dev/null | tr -d ' \015')
check "busybox wc -l" "$want" "$got"

check "busybox uname" "Linux armv7l" \
      "$($EMU --root . $BB uname -s -m 2>/dev/null | strip_cr)"

check "busybox expr" "4611686014132420609" \
      "$($EMU --root . $BB expr 2147483647 '*' 2147483647 2>/dev/null | strip_cr)"

check "busybox seq" "1 2 3 4 5" \
      "$($EMU --root . $BB seq 1 5 2>/dev/null | strip_cr | tr '\n' ' ' | sed 's/ $//')"

check "busybox sort -u" "a b c" \
      "$(printf 'c\na\nb\na\nc\n' | $EMU --root . $BB sort -u 2>/dev/null | strip_cr | tr '\n' ' ' | sed 's/ $//')"

# od exercises the printf path and every byte of a binary file.
want=$(od -An -tx1 -N 64 "$BB" | tr -s ' ' | strip_cr)
got=$($EMU --root . $BB od -An -tx1 -N 64 "$BB" 2>/dev/null | tr -s ' ' | strip_cr)
check "busybox od" "$want" "$got"

# awk: a bytecode interpreter inside the guest, so this is an interpreter running
# an interpreter -- and it uses doubles, which is the VFP path.
check "busybox awk" "338350" \
      "$($EMU --root . $BB awk 'BEGIN{s=0;for(i=1;i<=100;i++)s+=i*i;print s}' 2>/dev/null | strip_cr)"

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
