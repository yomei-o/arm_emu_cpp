#include "files.h"
#include <cstring>
#include <filesystem>
#include <system_error>

namespace arm {

namespace {
constexpr int64_t kENOENT = -2, kEBADF = -9, kEACCES = -13, kEINVAL = -22;

// Linux open() flags, which are not the host's. O_WRONLY/O_RDWR are the low two
// bits; the rest have to be matched by value, not by the host's <fcntl.h>.
constexpr int kO_WRONLY = 01, kO_RDWR = 02, kO_CREAT = 0100, kO_TRUNC = 01000,
              kO_APPEND = 02000;

// Defined further down with fill_stat, which is its other caller. A directory
// listing and a stat of the same file have to agree about the inode, so both go
// through this.
uint64_t path_ino(const std::string& path);
}  // namespace

// Turn whatever the guest said into one absolute, canonical guest path: relative to
// the cwd if it was relative, with "." dropped and ".." resolved -- and **clamped at
// the guest root**, because `--root` is a chroot and "/.." is "/" inside one.
//
// The lexical resolution is not tidiness. `stat(".")` against `stat("..")` is how
// libc's getcwd(3) recognises that it has reached the root: it walks upward
// comparing (st_dev, st_ino) and stops when the two name the same file. Those inodes
// are hashed from the path here, so without this "tree/." and "tree/.." hash
// differently, the two are never equal, and getcwd walks out of the sysroot looking
// for a root it will never reach.
std::string Files::normalize(const std::string& guest) const {
    std::string p = guest;
    if (p.empty() || p[0] != '/') p = (cwd == "/" ? std::string("/") : cwd + "/") + p;
    std::vector<std::string> parts;
    size_t i = 1;
    while (i <= p.size()) {
        size_t j = p.find('/', i);
        if (j == std::string::npos) j = p.size();
        const std::string c = p.substr(i, j - i);
        if (c == "..") { if (!parts.empty()) parts.pop_back(); }
        else if (!c.empty() && c != ".") parts.push_back(c);
        i = j + 1;
    }
    std::string out;
    for (const std::string& c : parts) { out += '/'; out += c; }
    return out.empty() ? "/" : out;
}

std::string Files::host_path(const std::string& guest) const {
    const std::string p = normalize(guest);
    if (root_.empty()) return p;
    std::string r = root_;
    while (!r.empty() && (r.back() == '/' || r.back() == '\\')) r.pop_back();
    // "/" under a root is the root directory itself, not "root/".
    return p == "/" ? r : r + p;
}

bool Files::is_dir(const std::string& path) const {
    std::error_code ec;
    return std::filesystem::is_directory(host_path(path), ec);
}

std::string Files::readlink(const std::string& path) const {
    std::error_code ec;
    const std::string hp = host_path(path);
    if (!std::filesystem::is_symlink(hp, ec)) return {};
    const auto target = std::filesystem::read_symlink(hp, ec);
    if (ec) return {};
    std::string s = target.string();
    for (char& ch : s) if (ch == '\\') ch = '/';
    return s;
}

int64_t Files::open(const std::string& path, int flags, int mode) {
    (void)mode;
    const std::string hp = host_path(path);
    std::error_code ec;
    if (std::filesystem::is_directory(hp, ec)) {
        // A directory descriptor: snapshot the listing now and serve it from
        // getdents64. Refusing this is what stops CPython -- importlib opens every
        // package directory before it tries to open a single module, so "cannot
        // list a directory" reads as "no such module".
        const int fd = next_fd_++;
        Entry e;
        e.used = true; e.is_directory = true; e.path = hp;
        e.entries.emplace_back(".", true);
        e.entries.emplace_back("..", true);
        for (const auto& dd : std::filesystem::directory_iterator(hp, ec))
            e.entries.emplace_back(dd.path().filename().string(), dd.is_directory(ec));
        open_[fd] = std::move(e);
        return fd;
    }
    const char* m = "rb";
    if (flags & kO_APPEND) m = (flags & kO_RDWR) ? "a+b" : "ab";
    else if (flags & kO_TRUNC) m = (flags & kO_RDWR) ? "w+b" : "wb";
    else if ((flags & 3) == kO_WRONLY) m = (flags & kO_CREAT) ? "wb" : "r+b";
    else if ((flags & 3) == kO_RDWR) m = (flags & kO_CREAT) ? "w+b" : "r+b";

    std::FILE* raw = std::fopen(hp.c_str(), m);
    if (!raw && (flags & kO_CREAT)) raw = std::fopen(hp.c_str(), "w+b");
    if (!raw) return kENOENT;
    const int fd = next_fd_++;
    Entry e;
    e.fp = std::shared_ptr<std::FILE>(raw, [](std::FILE* f) { if (f) std::fclose(f); });
    e.path = hp;
    e.used = true;
    open_[fd] = std::move(e);
    return fd;
}

int64_t Files::close(int fd) {
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    // The FILE* is closed by the last reference going away, not here: another
    // descriptor may still hold it.
    if (it->second.pipe) {
        // Which end is closing matters: a reader that sees no writers left is at
        // end of file, and a reader that sees one is merely waiting.
        if (it->second.writable) --it->second.pipe->writers;
        else --it->second.pipe->readers;
    }
    open_.erase(it);
    return 0;
}

int64_t Files::pipe2(int fds[2]) {
    auto p = std::make_shared<Pipe>();
    p->readers = 1;
    p->writers = 1;
    const int rd = next_fd_++, wr = next_fd_++;
    Entry re; re.used = true; re.pipe = p; re.writable = false;
    Entry we; we.used = true; we.pipe = p; we.writable = true;
    open_[rd] = std::move(re);
    open_[wr] = std::move(we);
    fds[0] = rd;
    fds[1] = wr;
    return 0;
}

int64_t Files::dup(int oldfd, int newfd) {
    auto it = open_.find(oldfd);
    if (it == open_.end()) return kEBADF;
    if (newfd < 0) newfd = next_fd_++;
    else if (newfd != oldfd) close(newfd);          // dup2 silently replaces
    else return newfd;                              // dup2(fd, fd) is a no-op
    Entry e = it->second;
    if (e.pipe) { if (e.writable) ++e.pipe->writers; else ++e.pipe->readers; }
    // `e` is a copy, so the shared_ptr is copied with it and both descriptors now
    // refer to the same open file -- which is what dup(2) means.
    open_[newfd] = std::move(e);
    if (newfd >= next_fd_) next_fd_ = newfd + 1;
    return newfd;
}

int64_t Files::move_fd(int oldfd, int newfd) {
    auto it = open_.find(oldfd);
    if (it == open_.end()) return kEBADF;
    if (newfd == oldfd) return newfd;
    close(newfd);                                   // silently replaces, like dup2
    open_[newfd] = std::move(open_[oldfd]);
    open_.erase(oldfd);
    if (newfd >= next_fd_) next_fd_ = newfd + 1;
    return newfd;
}

// getdents64: pack linux_dirent64 records until the buffer is full.
//   struct linux_dirent64 { u64 d_ino; s64 d_off; u16 d_reclen; u8 d_type; char d_name[]; }
// The record is the same on 32-bit ARM as on 64-bit: d_ino and d_off are explicitly
// 64 bits wide, which is the whole reason the "64" suffix exists.
int64_t Files::getdents64(int fd, void* buf, uint64_t len) {
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    Entry& e = it->second;
    if (!e.is_directory) return -20;                       // ENOTDIR
    auto* out = static_cast<uint8_t*>(buf);
    uint64_t used = 0;
    while (e.pos < e.entries.size()) {
        const std::string& name = e.entries[e.pos].first;
        const uint64_t reclen = (19 + name.size() + 1 + 7) & ~7ull;
        if (used + reclen > len) break;
        std::memset(out + used, 0, static_cast<size_t>(reclen));
        // The same inode `stat` would report, not a sequence number: getcwd(3)'s
        // fallback finds a directory's name by listing its parent and matching
        // inodes, so a listing that invents them cannot be walked.
        const uint64_t ino = path_ino(e.path + "/" + name);
        const int64_t off = static_cast<int64_t>(e.pos + 1);
        const uint16_t rl = static_cast<uint16_t>(reclen);
        const uint8_t type = e.entries[e.pos].second ? 4 : 8;   // DT_DIR : DT_REG
        std::memcpy(out + used + 0, &ino, 8);
        std::memcpy(out + used + 8, &off, 8);
        std::memcpy(out + used + 16, &rl, 2);
        out[used + 18] = type;
        std::memcpy(out + used + 19, name.c_str(), name.size() + 1);
        used += reclen;
        ++e.pos;
    }
    return static_cast<int64_t>(used);
}

int64_t Files::read(int fd, void* dst, uint64_t len) {
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    if (it->second.pipe) {
        Pipe& p = *it->second.pipe;
        const size_t have = p.buf.size() - p.pos;
        if (!have) return 0;                    // see the note on Pipe
        const size_t n = have < len ? have : static_cast<size_t>(len);
        std::memcpy(dst, p.buf.data() + p.pos, n);
        p.pos += n;
        return static_cast<int64_t>(n);
    }
    if (!it->second.fp) return kEBADF;
    return static_cast<int64_t>(std::fread(dst, 1, static_cast<size_t>(len), it->second.fp.get()));
}

int64_t Files::write(int fd, const void* src, uint64_t len) {
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    if (it->second.pipe) {
        Pipe& p = *it->second.pipe;
        const auto* b = static_cast<const uint8_t*>(src);
        p.buf.insert(p.buf.end(), b, b + len);
        return static_cast<int64_t>(len);
    }
    if (!it->second.fp) return kEBADF;
    const size_t n = std::fwrite(src, 1, static_cast<size_t>(len), it->second.fp.get());
    // Flushed now, not at fclose -- because for a spawned child's descriptor there
    // *is* no fclose: the Files copies holding the FILE* go away without one. A
    // child that wrote its output through stdio buffering that nobody drains looks
    // like a child that wrote nothing.
    std::fflush(it->second.fp.get());
    return static_cast<int64_t>(n);
}

int64_t Files::lseek(int fd, int64_t off, int whence) {
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    // A pipe cannot seek, and saying so matters twice over: ESPIPE is what a guest's
    // "is this seekable?" probe expects, and fseek(nullptr) -- a pipe entry has no
    // FILE* -- is instant process death under MSVC's CRT, with no message.
    if (it->second.pipe) return -29;                   // ESPIPE
    if (it->second.is_directory || !it->second.fp) return -29;
    if (std::fseek(it->second.fp.get(), static_cast<long>(off),
                   whence == 1 ? SEEK_CUR : whence == 2 ? SEEK_END : SEEK_SET) != 0)
        return kEINVAL;
    return std::ftell(it->second.fp.get());
}

int64_t Files::pread(int fd, void* dst, uint64_t len, uint64_t off) {
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    if (it->second.pipe || !it->second.fp) return -29;  // ESPIPE (see lseek)
    const long save = std::ftell(it->second.fp.get());
    std::fseek(it->second.fp.get(), static_cast<long>(off), SEEK_SET);
    const size_t got = std::fread(dst, 1, static_cast<size_t>(len), it->second.fp.get());
    std::fseek(it->second.fp.get(), save, SEEK_SET);
    return static_cast<int64_t>(got);
}

namespace {
// The inode is an FNV hash of the host path rather than a constant, and this is not
// cosmetic: musl's ld.so decides "is this library already loaded?" by comparing
// st_dev/st_ino. With every file answering (1,1) the first library loaded stands in
// for all of them, and relocation fails against a library that is sitting right
// there.
uint64_t path_ino(const std::string& path) {
    // Normalised first, so that two spellings of one file hash the same. Without
    // it "tree/usr/lib" and "tree/usr/lib/." are two different inodes, and every
    // consumer of inode identity -- ld.so's "have I loaded this already", getcwd's
    // walk -- gets a wrong answer that looks like something else entirely.
    std::string p = std::filesystem::path(path).lexically_normal().generic_string();
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    uint64_t h = 1469598103934665603ull;
    for (unsigned char ch : p) { h ^= ch; h *= 1099511628211ull; }
    return h ? h : 1;                                  // 0 reads as "no file"
}

// ARM's stat64, by offset. See the note in files.h for why st_size is at 48.
void fill_stat(void* buf, uint64_t size, bool dir, const std::string& path) {
    std::memset(buf, 0, kStatSize);
    auto* p = static_cast<uint8_t*>(buf);
    auto put64 = [&](size_t off, uint64_t v) { std::memcpy(p + off, &v, 8); };
    auto put32 = [&](size_t off, uint32_t v) { std::memcpy(p + off, &v, 4); };
    const uint64_t ino = path_ino(path);
    put64(0, 1);                                       // st_dev
    put32(12, static_cast<uint32_t>(ino));             // __st_ino, the 32-bit one
    put32(16, dir ? 0040755u : 0100755u);              // st_mode
    put32(20, 1);                                      // st_nlink
    put32(24, 1000); put32(28, 1000);                  // st_uid, st_gid
    put64(48, size);                                   // st_size
    put32(56, 4096);                                   // st_blksize
    put64(64, (size + 511) / 512);                     // st_blocks
    put64(96, ino);                                    // st_ino, the 64-bit one
}
}  // namespace

int64_t Files::fstat(int fd, void* statbuf) {
    // The three standard descriptors are not in the table and never will be, but a
    // program still stats them: CPython calls fstat on 0/1/2 while setting up
    // sys.stdin/stdout/stderr, and a zeroed struct reads back as a *directory* --
    // whereupon it refuses to start with "<stdin> is a directory, cannot continue".
    if (fd >= 0 && fd <= 2 && !is_open(fd)) {
        std::memset(statbuf, 0, kStatSize);
        auto* p = static_cast<uint8_t*>(statbuf);
        const uint32_t mode = 0020620u;                // S_IFCHR | rw--w----, like a tty
        const uint32_t nlink = 1, blksz = 4096;
        std::memcpy(p + 16, &mode, 4);
        std::memcpy(p + 20, &nlink, 4);
        std::memcpy(p + 56, &blksz, 4);
        return 0;
    }
    auto it = open_.find(fd);
    if (it == open_.end()) return kEBADF;
    if (it->second.is_directory) {
        std::memset(statbuf, 0, kStatSize);
        auto* p = static_cast<uint8_t*>(statbuf);
        const uint32_t mode = 0040755u;
        std::memcpy(p + 16, &mode, 4);
        return 0;
    }
    if (it->second.pipe) {
        std::memset(statbuf, 0, kStatSize);
        auto* p = static_cast<uint8_t*>(statbuf);
        const uint32_t mode = 0010600u;                // S_IFIFO
        std::memcpy(p + 16, &mode, 4);
        return 0;
    }
    std::error_code ec;
    const auto sz = std::filesystem::file_size(it->second.path, ec);
    fill_stat(statbuf, ec ? 0 : static_cast<uint64_t>(sz), false, it->second.path);
    return 0;
}

int64_t Files::stat_path(const std::string& path, void* statbuf) {
    const std::string hp = host_path(path);
    std::error_code ec;
    if (std::filesystem::is_directory(hp, ec)) { fill_stat(statbuf, 4096, true, hp); return 0; }
    const auto sz = std::filesystem::file_size(hp, ec);
    if (ec) return kENOENT;
    fill_stat(statbuf, static_cast<uint64_t>(sz), false, hp);
    return 0;
}

int64_t Files::unlink(const std::string& path) {
    std::error_code ec;
    const bool gone = std::filesystem::remove(host_path(path), ec);
    if (ec) return kEACCES;
    return gone ? 0 : kENOENT;
}

int64_t Files::mkdir(const std::string& path) {
    const std::string hp = host_path(path);
    std::error_code ec;
    if (std::filesystem::exists(hp, ec)) return -17;   // EEXIST
    return std::filesystem::create_directory(hp, ec) && !ec ? 0 : kEACCES;
}

int64_t Files::rename(const std::string& from, const std::string& to) {
    std::error_code ec;
    // POSIX rename replaces an existing target; std::filesystem::rename does too on
    // POSIX hosts but refuses on Windows, so clear the way first.
    std::filesystem::remove(host_path(to), ec);
    ec.clear();
    std::filesystem::rename(host_path(from), host_path(to), ec);
    return ec ? kENOENT : 0;
}

int64_t Files::access(const std::string& path) {
    // A missing file is ENOENT, not EACCES: a guest distinguishes "not there" from
    // "forbidden" -- gcc probes its search path with access() and treats the two
    // differently.
    std::error_code ec;
    return std::filesystem::exists(host_path(path), ec) ? 0 : kENOENT;
}

}  // namespace arm
