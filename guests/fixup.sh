#!/bin/sh
# Put back the symbolic links Windows `tar` dropped on the way out of an .apk.
#
# An Alpine package is full of them, and the dynamic loader asks for the *soname* --
# `libz.so.1`, not `libz.so.1.3.2`. The failure is never "file not found": it is a
# library that cannot resolve a symbol that is plainly there, or a link that quietly
# falls back to the static archive. Copies rather than links, because a link made on
# a Windows drive reads back as EINVAL from a native process.
#
#   sh guests/fixup.sh guests/sysroot
#
# On a Linux host, where tar got the links right, every step below is a no-op.
set -e
ROOT=${1:-.}
[ -d "$ROOT" ] || { echo "usage: sh guests/fixup.sh <tree>"; exit 2; }

python - "$ROOT" <<'PY'
import os, re, shutil, sys
root = sys.argv[1]
made = 0
for base, _dirs, files in os.walk(root):
    for f in sorted(files):
        m = re.match(r'^(.*\.so)\.(\d+)(\..*)?$', f)
        if not m:
            continue
        src = os.path.join(base, f)
        # Both the soname (libX.so.N) and the linker name (libX.so). The first is
        # what the loader asks for at run time; the second is what `-lX` finds.
        for cand in (m.group(1) + "." + m.group(2), m.group(1)):
            dst = os.path.join(base, cand)
            if f != cand and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                made += 1

# The names that carry no version suffix to derive them from.
pairs = [
    ("usr/bin/python3.12", "usr/bin/python3"),
    # Alpine's /usr/lib/libc.so is a symlink to the loader itself. Without it `-lc`
    # finds libc.a, the link goes static, and `gcc hello.c` fails on link order with
    # "undefined reference to `raise`".
    ("lib/ld-musl-armhf.so.1", "usr/lib/libc.so"),
]
for src, dst in pairs:
    s, d = os.path.join(root, src), os.path.join(root, dst)
    if os.path.exists(s) and not os.path.exists(d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        made += 1
print("fixup: %d name(s) materialised under %s" % (made, root))
PY
