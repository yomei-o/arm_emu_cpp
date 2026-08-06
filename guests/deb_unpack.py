#!/usr/bin/env python3
"""Unpack a Debian/Raspbian .deb into a directory tree, resolving symlinks to copies.

    python guests/deb_unpack.py <dest-dir> <file.deb> [file.deb ...]

Two things this does that `ar x` piped into `tar` does not, and both are the reason
it exists:

  * **Symbolic links become copies.** Windows `tar` cannot create one without
    developer mode, and *fails the whole extraction* when it tries -- so a package
    that ships `libc.so.6 -> libc-2.36.so` takes the tree down with it. A link on a
    Windows drive is also read back as EINVAL by a native process, so even where one
    can be made it is not useful. Copying is what the guest actually needs: the
    dynamic loader asks for a soname and has to find a file there.

  * **Links are resolved across packages, in a second pass.** `libgcc-s1` ships a
    link into `gcc-12-base`, which is a different .deb -- so a target that does not
    exist yet is deferred and retried once everything is unpacked.

No `ar`, no `xz` binary, no `dpkg`: a .deb is an ar archive of three members and the
standard library has everything needed to read it.
"""
import lzma
import os
import shutil
import sys
import tarfile
import io


def ar_members(data):
    """Yield (name, bytes) for each member of an ar archive."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("not an ar archive")
    off = 8
    while off + 60 <= len(data):
        hdr = data[off:off + 60]
        name = hdr[0:16].decode("ascii", "replace").strip().rstrip("/")
        size = int(hdr[48:58].decode("ascii", "replace").strip() or 0)
        off += 60
        yield name, data[off:off + size]
        off += size + (size & 1)          # members are padded to an even offset


def open_data_member(name, blob):
    if name.endswith(".xz"):
        return io.BytesIO(lzma.decompress(blob))
    if name.endswith(".gz"):
        import gzip
        return io.BytesIO(gzip.decompress(blob))
    if name.endswith(".zst"):
        import zstandard                  # only if a suite ever uses it
        return io.BytesIO(zstandard.ZstdDecompressor().decompress(blob))
    return io.BytesIO(blob)


def unpack(dest, debs):
    deferred = []                          # (link_path, target_path) to retry later
    for deb in debs:
        with open(deb, "rb") as f:
            data = f.read()
        tar_blob = None
        for name, blob in ar_members(data):
            if name.startswith("data.tar"):
                tar_blob = (name, blob)
        if tar_blob is None:
            raise ValueError("%s has no data member" % deb)
        with tarfile.open(fileobj=open_data_member(*tar_blob), mode="r:") as tf:
            for m in tf.getmembers():
                rel = m.name.lstrip("./")
                if not rel:
                    continue
                out = os.path.join(dest, rel.replace("/", os.sep))
                if m.isdir():
                    os.makedirs(out, exist_ok=True)
                elif m.issym() or m.islnk():
                    # Resolve relative to the link's own directory, the way the
                    # filesystem would.
                    tgt = m.linkname
                    base = os.path.dirname(out) if m.issym() and not tgt.startswith("/") \
                        else dest
                    src = os.path.normpath(os.path.join(base, tgt.lstrip("/").replace("/", os.sep)
                                                        if tgt.startswith("/") else tgt))
                    deferred.append((out, src))
                elif m.isfile():
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with tf.extractfile(m) as fin, open(out, "wb") as fout:
                        shutil.copyfileobj(fin, fout)

    # Second pass: a link may point into a package unpacked later. Repeat until
    # nothing more can be resolved, because links can chain.
    made, unresolved = 0, deferred
    while True:
        again = []
        for out, src in unresolved:
            if os.path.isfile(src) and not os.path.exists(out):
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copyfile(src, out)
                made += 1
            elif not os.path.exists(out):
                again.append((out, src))
        if len(again) == len(unresolved):
            break
        unresolved = again
    print("unpacked %d .deb(s) into %s; %d link(s) copied, %d unresolved"
          % (len(debs), dest, made, len(unresolved)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(unpack(sys.argv[1], sys.argv[2:]))
