#!/bin/sh
# Fetch a Raspberry Pi OS (Raspbian bookworm) armhf tree: glibc, CPython 3.11 and
# gcc 12, as the Pi's own distribution built them.
#
# This is a different guest from the Alpine one in two ways that matter, and both are
# the reason it is here:
#
#   * **glibc, not musl.** glibc picks its memcpy, strlen and memset through IFUNC
#     resolvers that read AT_HWCAP -- so the HWCAP word in elf_loader.cpp stops being
#     decorative and becomes a promise the emulator has to keep.
#   * **ARMv6, not ARMv7.** Raspbian is rebuilt for the original Pi, so its code may
#     reach the kernel helper page at 0xFFFF0000 for atomics and the thread pointer
#     rather than using LDREX/STREX and CP15 directly.
#
# ~44 MB of .debs. Needs curl and python; the .deb reading is in deb_unpack.py,
# which is there because Windows tar cannot create a symlink and fails the whole
# extraction when a package ships one.
set -e
cd "$(dirname "$0")/.."
DEST=${1:-guests/rpi}
MIRROR=http://raspbian.raspberrypi.com/raspbian
PY=${PYTHON:-python}
mkdir -p "$DEST"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

n=0
# Stripping CR because the list may have been written on Windows, where a stray
# carriage return turns every URL into a 404.
for f in $(tr -d '\r' < guests/rpi-packages.txt); do
    n=$((n + 1))
    printf '  %2d  %s\n' "$n" "$(basename "$f")"
    curl -sfL -o "$tmp/$n.deb" "$MIRROR/$f"
done

# One invocation for all of them, because a symlink in one package routinely points
# into another and the resolver needs the whole set on hand.
"$PY" guests/deb_unpack.py "$DEST" "$tmp"/*.deb
echo "done: $DEST"
