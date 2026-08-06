# guests/

Nothing in here is a source file, and nothing in here is committed. These are real
ARM Linux binaries downloaded from the distributions that built them — the whole
point of them is that they have never heard of this emulator.

The commands below fetch each one. `tests/run_*.sh` skips itself when the tree it
needs is absent, so a fresh clone runs the parts it can and says which parts it
cannot.

## One thing to know before any of this

**Windows `tar` writes a symlink as nothing at all.** An Alpine package is full of
them — `libz.so.1 -> libz.so.1.3.2`, `libc.so -> /lib/ld-musl-armhf.so.1`,
`python3 -> python3.12` — and the dynamic loader asks for the *soname*, which is the
name that went missing. The symptom is not "file not found": it is a library that
fails to resolve a symbol that is plainly there, or a link that silently falls back
to the static archive and then fails on link order.

So after unpacking anything, run `sh guests/fixup.sh`, which turns every
`libX.so.N.M.P` into copies named `libX.so.N` and `libX.so` and fills in the handful
of program names. On a Linux host it is a no-op worth running anyway.

## busybox — a static armv7 musl binary, ~880 KB

```sh
cd guests
curl -Lo bb.apk https://dl-cdn.alpinelinux.org/alpine/v3.20/main/armv7/busybox-static-1.36.1-r31.apk
tar xzf bb.apk && mv bin/busybox.static busybox && rm -rf bb.apk bin .PKGINFO .SIGN*
```

Then `sh tests/run_busybox.sh`. This is the first real guest and a good one: Thumb-2
end to end, with musl's hand-written division jump table, IT blocks everywhere, and
digests that can be checked against the host's own `md5sum`.

## sysroot — CPython 3.12 for armv7, dynamically linked, ~60 MB

```sh
mkdir -p guests/sysroot && cd guests/sysroot
for p in musl-1.2.5-r3 busybox-1.36.1-r31 python3-3.12.13-r0 \
         libffi-3.4.6-r0 zlib-1.3.2-r0 xz-libs-5.8.3-r0 libbz2-1.0.8-r6 \
         sqlite-libs-3.45.3-r3 openssl-3.3.7-r0 ncurses-libs-6.4_p20240420-r2 \
         readline-8.2.10-r0 expat-2.8.1-r0 mpdecimal-4.0.0-r0 gdbm-1.23-r1; do
  curl -sfLO "https://dl-cdn.alpinelinux.org/alpine/v3.20/main/armv7/$p.apk"
  tar xzf "$p.apk"; rm -f "$p.apk"
done
rm -f .PKGINFO .SIGN* .post-* .pre-* .trigger
cd ../.. && sh guests/fixup.sh guests/sysroot
```

Then `sh tests/run_python.sh`. What runs is a stock CPython with its own
`ld-musl-armhf.so.1` doing every relocation — the emulator stays out of the linking
entirely, which is both less work and more faithful than writing a loader.

## gccroot — Alpine's armv7 gcc 13, ~90 MB

```sh
mkdir -p guests/gccroot && cd guests/gccroot
for p in binutils-2.42-r1 gcc-13.2.1_git20240309-r1 musl-dev-1.2.5-r3 musl-1.2.5-r3 \
         libgcc-13.2.1_git20240309-r1 libatomic-13.2.1_git20240309-r1 \
         gmp-6.3.0-r1 mpc1-1.3.1-r1 mpfr4-4.2.1-r0 zlib-1.3.2-r0 isl26-0.26-r1 \
         zstd-libs-1.5.6-r0 jansson-2.14-r4; do
  curl -sfLO "https://dl-cdn.alpinelinux.org/alpine/v3.20/main/armv7/$p.apk"
  tar xzf "$p.apk"; rm -f "$p.apk"
done
rm -f .PKGINFO .SIGN* .post-* .pre-* .trigger
cd ../.. && sh guests/fixup.sh guests/gccroot
```

Then `sh tests/run_gcc.sh`.

`usr/lib/libc.so` is the one name `fixup.sh` cannot derive from a version suffix: on
Alpine it is a symlink to `/lib/ld-musl-armhf.so.1`, and without it `-lc` finds
`libc.a` instead and the link goes *static*. That is not merely slower — it changes
the link order, and `gcc hello.c` then fails with "undefined reference to `raise`"
from libgcc's division-by-zero handler, which is a real hazard of static musl links
and nothing to do with the emulator. `fixup.sh` puts it in place.
