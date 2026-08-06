# guests/

Nothing in here is a source file, and nothing in here is committed. These are real
ARM Linux binaries downloaded from the distributions that built them — the whole
point of them is that they have never heard of this emulator.

The commands below fetch each one. `tests/run_*.sh` skips itself when the tree it
needs is absent, so a fresh clone runs the parts it can and says which parts it
cannot.

## busybox — a static armv7 musl binary, ~880 KB

```sh
cd guests
curl -Lo bb.apk https://dl-cdn.alpinelinux.org/alpine/v3.20/main/armv7/busybox-static-1.36.1-r31.apk
tar xzf bb.apk && mv bin/busybox.static busybox && rm -rf bb.apk bin .PKGINFO .SIGN*
```

Then `sh tests/run_busybox.sh`. This is the first real guest and it is a good one:
Thumb-2 end to end, with musl's hand-written division jump table, IT blocks
everywhere, and digests that can be checked against the host's own `md5sum`.
