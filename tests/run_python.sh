#!/bin/sh
# A stock CPython for 32-bit ARM, dynamically linked, running on whatever this host
# is -- with its own `ld-musl-armhf.so.1` doing the relocation and symbol binding.
#
# The digests are the point. Every byte of them travels through the emulated CPU, so
# a single wrong bit anywhere in half a billion instructions changes the answer, and
# the host's own Python computes the same ones. That is a much stronger claim than
# "it started and printed something".
set -e
cd "$(dirname "$0")/.."
EMU=${EMU:-./armemu}
ROOT=guests/sysroot
PY=$ROOT/usr/bin/python3
[ -f "$PY" ] || { echo "skip python (no $PY -- see guests/README.md)"; exit 0; }

pass=0; fail=0
check() {   # name, want, got
    if [ "$2" = "$3" ]; then echo "ok   $1"; pass=$((pass+1));
    else echo "FAIL $1"; echo "     want: $2"; echo "     got:  $3"; fail=$((fail+1)); fi
}
run() { $EMU --root $ROOT $PY -c "$1" 2>/dev/null | tr -d '\015'; }

check "python version and machine" "3.12.13 armv7l" \
      "$(run 'import sys,platform; print(sys.version.split()[0], platform.machine())')"

# The host computes the same digest from the same bytes.
want=$(python -c "import hashlib;print(hashlib.sha256(b'arm_emu_cpp').hexdigest())" 2>/dev/null \
       || python3 -c "import hashlib;print(hashlib.sha256(b'arm_emu_cpp').hexdigest())")
check "python sha256 (against the host's)" "$want" \
      "$(run 'import hashlib; print(hashlib.sha256(b"arm_emu_cpp").hexdigest())')"

# Integer arithmetic well past 64 bits, which is CPython's own bignum code and
# therefore a long run of UMULL/ADC chains.
check "python bignum" "$(python -c 'print(pow(3,1000,10**18))' 2>/dev/null || python3 -c 'print(pow(3,1000,10**18))')" \
      "$(run 'print(pow(3,1000,10**18))')"

# Floating point through VFP, formatted by CPython's own dtoa.
check "python float" "0.1 0.30000000000000004 1.4142135623730951" \
      "$(run 'import math; print(0.1, 0.1+0.2, math.sqrt(2))')"

# Four threads on one lock. The count comes out exactly right only if the exclusive
# monitor is real -- an emulator whose STREX always succeeds loses updates and
# reports a plausible number.
check "python threading" "counter 80000" \
      "$(run 'import threading
c=[0]; lk=threading.Lock()
def w():
    for _ in range(20000):
        with lk: c[0]+=1
ts=[threading.Thread(target=w) for _ in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("counter", c[0])')"

# CPython spawning CPython: fork, execve, a pipe, and wait4.
check "python subprocess (fork/exec/pipe/wait)" "0 b'42\\n'" \
      "$(run 'import subprocess
r = subprocess.run(["/usr/bin/python3","-c","print(6*7)"], capture_output=True)
print(r.returncode, r.stdout)')"

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
