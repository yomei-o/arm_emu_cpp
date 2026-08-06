// WebAssembly entry point.
//
// The emulator core has no host dependencies beyond the C++ standard library, so the
// browser build needs only two things: a way to hand it an ELF plus the files the
// guest will open, and a way to route guest output back into JS.
//
// Files come from Emscripten's MEMFS, which the page populates before calling
// `emu_run`. That means the same `Files` layer works unchanged -- `fopen` on a MEMFS
// path behaves like `fopen` anywhere else -- and a dynamically linked guest can find
// its own interpreter and libraries.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cpu.h"
#include "loader.h"
#include "memory.h"
#include "start.h"
#include "syscalls.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
// Guest output is bytes, not text: hand JS the raw slice and let it decode. The
// callbacks live on globalThis so the reference resolves the same way however the
// module is bundled.
EM_JS(void, js_out, (int fd, const char* p, int n), {
    globalThis.armemuOutput(fd, HEAPU8.slice(p, p + n));
});
EM_JS(void, js_log, (const char* p), { globalThis.armemuLog(UTF8ToString(p)); });
#else
static void js_out(int, const char*, int) {}
static void js_log(const char*) {}
#endif

namespace {

std::string g_error;
uint64_t g_insns = 0;

std::vector<uint8_t> read_file(const std::string& path) {
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return {};
    std::fseek(fp, 0, SEEK_END);
    const long n = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::vector<uint8_t> v(n > 0 ? static_cast<size_t>(n) : 0);
    if (n > 0 && std::fread(v.data(), 1, static_cast<size_t>(n), fp) != static_cast<size_t>(n))
        v.clear();
    std::fclose(fp);
    return v;
}

// The same layout main.cpp uses. Kept in step deliberately: the two front ends
// drifted in the sibling project and the browser build ran a different address space
// from the command line one, which is a difference nothing reports.
constexpr uint32_t kPieBase    = 0x00400000u;
constexpr uint32_t kInterpBase = 0xB6F00000u;
constexpr uint32_t kMmapBase   = 0x40000000u;
constexpr uint32_t kStackTop   = 0xBEFFF000u;
constexpr uint32_t kStackSize  = 8u << 20;

}  // namespace

extern "C" {

// Run `path` (a MEMFS path) with arguments given as a NUL-separated,
// double-NUL-terminated list -- which avoids needing a JSON parser here.
EMSCRIPTEN_KEEPALIVE int emu_run(const char* path, const char* argz, const char* root,
                                 double max_insns) {
    using namespace arm;
    g_error.clear();
    g_insns = 0;

    const std::vector<uint8_t> file = read_file(path);
    if (file.empty()) { g_error = std::string("cannot read ") + path; return -1; }

    Memory mem;
    Cpu cpu(mem);
    cpu.max_insns = static_cast<uint64_t>(max_insns);
    Syscalls sys(cpu, mem);
    sys.output = [](int fd, const char* p, uint64_t n) { js_out(fd, p, static_cast<int>(n)); };
    cpu.on_undefined = [&sys](uint32_t, uint32_t at) { return sys.deliver_signal(4, at); };

    std::vector<std::string> argv;
    for (const char* p = argz; p && *p; p += std::strlen(p) + 1) argv.emplace_back(p);
    if (argv.empty()) argv.emplace_back(path);

    LoadedImage img;
    std::string err;
    if (!load_elf(file, mem, kPieBase, &img, &err)) { g_error = err; return -1; }

    uint32_t interp_base = 0, start_pc = 0;
    bool interp_thumb = false;
    if (!img.interp.empty()) {
        std::string ip = img.interp;
        if (root && *root && ip[0] == '/') ip = std::string(root) + ip;
        const std::vector<uint8_t> ifile = read_file(ip);
        if (ifile.empty()) { g_error = "cannot read interpreter " + ip; return -1; }
        LoadedImage interp;
        if (!load_elf(ifile, mem, kInterpBase, &interp, &err)) { g_error = err; return -1; }
        interp_base = interp.base;
        start_pc = interp.entry;
        interp_thumb = interp.thumb_entry;
    }

    const std::vector<std::string> env = {
        "PATH=/usr/bin:/bin", "HOME=/", "LANG=C.UTF-8", "TERM=dumb",
        "PYTHONDONTWRITEBYTECODE=1",
    };

    mem.set(kStackTop - kStackSize, 0, kStackSize);
    cpu.set_sp(build_stack(mem, kStackTop, img, interp_base, argv, env));
    prepare_process(cpu, mem, img, start_pc, interp_thumb);
    sys.set_brk(img.brk);
    sys.set_mmap_base(kMmapBase);
    sys.exe_path = argv[0];
    sys.files.set_root(root ? root : "");

    {
        char buf[160];
        std::snprintf(buf, sizeof buf, "ELF entry %08X (%s)%s", cpu.pc,
                      cpu.thumb ? "Thumb" : "ARM",
                      img.interp.empty() ? ", static" : ", via the dynamic loader");
        js_log(buf);
    }

    int rc;
    try {
        rc = run_image(cpu, mem, sys);
    } catch (const CpuError& e) {
        g_error = e.what;
        rc = -1;
    }
    g_insns = cpu.insns;
    return rc;
}

EMSCRIPTEN_KEEPALIVE const char* emu_error() { return g_error.c_str(); }
EMSCRIPTEN_KEEPALIVE double emu_instructions() { return static_cast<double>(g_insns); }

}  // extern "C"
