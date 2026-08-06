// The guest's view of the filesystem.
//
// Guest descriptors are small integers the guest chooses nothing about, so they are
// allocated here and mapped to host handles. 0/1/2 are the console and never touch
// the host filesystem layer.
//
// Paths are passed through with only separator normalisation: the guest is a Linux
// program and will say "/usr/lib/python3.11/os.py", so a root directory maps that
// onto the host. Without a root, absolute guest paths are used as-is, which is what
// you want when the host *is* Linux.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace arm {

// ARM's `struct stat64`, which is 104 bytes and is *not* the natural layout of its
// members: `st_size` needs 8-byte alignment and lands at 48 rather than 44, and
// `st_ino` is duplicated -- a 32-bit `__st_ino` at 12 for the old ABI and a 64-bit
// one at 96. A guest libc reads it by offset, so these numbers are the interface.
constexpr unsigned kStatSize = 104;

class Files {
public:
    void set_root(const std::string& r) { root_ = r; }

    // Returns a guest fd, or a negative errno.
    int64_t open(const std::string& path, int flags, int mode);
    int64_t close(int fd);
    int64_t read(int fd, void* dst, uint64_t len);
    int64_t write(int fd, const void* src, uint64_t len);
    int64_t lseek(int fd, int64_t off, int whence);
    // Has this descriptor been opened or redirected here? Descriptors 0..2 are the
    // console by default and are not in the table; a `dup2` onto one puts it there.
    bool is_open(int fd) const { return open_.count(fd) != 0; }
    int64_t unlink(const std::string& path);
    int64_t mkdir(const std::string& path);
    int64_t rename(const std::string& from, const std::string& to);
    int64_t pipe2(int fds[2]);
    // dup/dup2. `newfd` of -1 means "the lowest free one", which is dup(2).
    int64_t dup(int oldfd, int newfd);
    int64_t move_fd(int oldfd, int newfd);
    Files clone_for_exec() const { return *this; }
    int64_t pread(int fd, void* dst, uint64_t len, uint64_t off);
    // Fills an ARM `struct stat64` (kStatSize bytes). Returns 0 or -errno.
    int64_t fstat(int fd, void* statbuf);
    int64_t stat_path(const std::string& path, void* statbuf);
    // Emits linux_dirent64 records into a host buffer; returns bytes written, 0 at
    // end of directory, or -errno.
    int64_t getdents64(int fd, void* buf, uint64_t len);
    int64_t access(const std::string& path);
    bool is_dir(const std::string& path) const;
    // The target of a symbolic link, or empty when it is not one. Needed because a
    // sysroot assembled on Windows has real files where a Linux tree has links, and
    // a guest that resolves one itself has to be told which it is looking at.
    std::string readlink(const std::string& path) const;

    std::string host_path(const std::string& guest) const;
    std::string cwd = "/";

private:
    // A pipe: one buffer, two descriptors. Shared between the two ends and, because
    // descriptors survive `execve`, between a parent and the child it spawned --
    // which is the whole point of having one.
    //
    // The buffer grows without limit, and that is a decision rather than an
    // oversight. A real pipe blocks the writer at 64 KiB and expects the reader to
    // be running concurrently; nothing here runs two processes at once (see
    // `sys_fork`), so a bounded buffer would be a deadlock rather than back-pressure.
    struct Pipe {
        std::vector<uint8_t> buf;
        size_t pos = 0;              // how much the reader has taken
        int readers = 0, writers = 0;
        bool empty() const { return pos >= buf.size(); }
    };

    struct Entry {
        std::FILE* fp = nullptr;
        std::shared_ptr<Pipe> pipe;
        bool writable = false;
        std::string path;
        bool used = false;
        // A directory descriptor holds its listing instead of a file handle.
        // Python's importlib opens every package directory with O_DIRECTORY and
        // reads it with getdents64 to cache what is there before it tries to open
        // any module, so a guest that cannot list a directory cannot import
        // anything.
        bool is_directory = false;
        std::vector<std::pair<std::string, bool>> entries;   // name, is_dir
        size_t pos = 0;
    };
    std::map<int, Entry> open_;
    std::string root_;
    int next_fd_ = 3;
};

}  // namespace arm
