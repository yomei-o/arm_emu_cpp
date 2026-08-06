// Lets one source file be both the guest program and its own oracle.
//
// Built for ARM it is freestanding: no libc, `_start`, and the syscalls it needs
// written by hand. Built natively (-DARM_NATIVE) it is an ordinary program using
// printf. `run_tests.sh` builds both and diffs the output, so "the emulator is
// right" means "it produced the same bytes the host did", not "the number looked
// plausible" -- which is the only claim worth making about an arithmetic core.
//
// Fixed-width types throughout, deliberately: `long` is 32-bit on the ARM side and
// on a Windows host but 64-bit on a Linux one, so a test written with `long` would
// diff for a reason that has nothing to do with the emulator.
#pragma once
#include <stdint.h>

#ifdef ARM_NATIVE

#include <stdio.h>
static void out(const char* s, uint32_t n) { fwrite(s, 1, (size_t)n, stdout); }
static void finish(void) { fflush(stdout); }
#define TEST_MAIN int main(void)

// The file primitives, in both shapes, report failure as a negative value -- so a
// test can be written once and mean the same thing in each build.
//
// The handle is a small integer on both sides, and on the host that means a table
// rather than a cast of the FILE*: a 64-bit pointer squeezed into the 32-bit type
// an ARM guest uses comes back negative about half the time, and negative is how
// this interface spells failure. The host oracle then reports "open failed" for a
// file that opened perfectly, and the diff blames the emulator.
static FILE* t_handles[16];
static int32_t t_open(const char* p) {
    for (int i = 1; i < 16; i++) {
        if (t_handles[i]) continue;
        FILE* f = fopen(p, "rb");
        if (!f) return -1;
        t_handles[i] = f;
        return i;
    }
    return -1;
}
static int32_t t_read(int32_t h, char* b, uint32_t n) {
    return (int32_t)fread(b, 1, (size_t)n, t_handles[h]);
}
static int64_t t_fsize(int32_t h) {
    FILE* f = t_handles[h];
    long cur = ftell(f);
    fseek(f, 0, SEEK_END);
    long e = ftell(f);
    fseek(f, cur, SEEK_SET);
    return (int64_t)e;
}
static void t_close(int32_t h) { fclose(t_handles[h]); t_handles[h] = 0; }
#define TEST_DATA "harness.h"

#else

// EABI: the call number is in r7 and the arguments in r0..r2. `svc #0` with a zero
// immediate -- the number is *not* in the immediate, which is the older OABI form.
static int32_t arm_sys(int32_t n, int32_t a, int32_t b, int32_t c) {
    register int32_t r7 __asm__("r7") = n;
    register int32_t r0 __asm__("r0") = a;
    register int32_t r1 __asm__("r1") = b;
    register int32_t r2 __asm__("r2") = c;
    __asm__ volatile("svc #0" : "+r"(r0) : "r"(r7), "r"(r1), "r"(r2) : "memory");
    return r0;
}
static int32_t arm_sys4(int32_t n, int32_t a, int32_t b, int32_t c, int32_t d) {
    register int32_t r7 __asm__("r7") = n;
    register int32_t r0 __asm__("r0") = a;
    register int32_t r1 __asm__("r1") = b;
    register int32_t r2 __asm__("r2") = c;
    register int32_t r3 __asm__("r3") = d;
    __asm__ volatile("svc #0" : "+r"(r0) : "r"(r7), "r"(r1), "r"(r2), "r"(r3) : "memory");
    return r0;
}
static void out(const char* s, uint32_t n) { arm_sys(4, 1, (int32_t)(intptr_t)s, (int32_t)n); }
static void finish(void) { arm_sys(1, 0, 0, 0); }
#define TEST_MAIN void _start(void)

static int32_t t_open(const char* p) { return arm_sys(5, (int32_t)(intptr_t)p, 0 /*O_RDONLY*/, 0); }
static int32_t t_read(int32_t h, char* b, uint32_t n) {
    return arm_sys(3, h, (int32_t)(intptr_t)b, (int32_t)n);
}
static int64_t t_fsize(int32_t h) {
    uint8_t st[104];                             // ARM's struct stat64
    int32_t r = arm_sys(197 /*fstat64*/, h, (int32_t)(intptr_t)st, 0);
    if (r < 0) return r;
    int64_t size;
    __builtin_memcpy(&size, st + 48, 8);         // st_size -- 48, not 44: see files.h
    return size;
}
static void t_close(int32_t h) { arm_sys(6, h, 0, 0); }
#define TEST_DATA "/harness.h"

// ---------------------------------------------------------------------------
// The two runtime helpers a freestanding ARM build cannot do without.
//
// ARM has no 64-bit divide instruction, so the compiler calls libgcc for `int64 /
// int64` -- and a `-nostdlib` build has no libgcc. Supplying them here keeps the
// tests genuinely freestanding (no toolchain sysroot to find) *and* keeps 64-bit
// division in the arithmetic test, which is worth having: it is a long
// shift-subtract loop and exercises the carry flag, the conditional execution and
// the register pressure all at once.
//
// The EABI signature cannot be written in C -- it returns the quotient in r0:r1 and
// the remainder in r2:r3 -- so the wrapper is naked assembly around a C body that
// leaves the remainder in a global.
// ---------------------------------------------------------------------------
uint64_t arm_divmod_rem;

uint64_t arm_udiv64(uint64_t n, uint64_t d) {
    if (d == 0) { arm_divmod_rem = n; return 0; }
    uint64_t q = 0, r = 0;
    for (int i = 63; i >= 0; --i) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) { r -= d; q |= (uint64_t)1 << i; }
    }
    arm_divmod_rem = r;
    return q;
}

__attribute__((naked)) void __aeabi_uldivmod(void) {
    __asm__ volatile(
        "push  {r4, lr}\n"
        "bl    arm_udiv64\n"
        "movw  r4, :lower16:arm_divmod_rem\n"
        "movt  r4, :upper16:arm_divmod_rem\n"
        "ldr   r2, [r4]\n"
        "ldr   r3, [r4, #4]\n"
        "pop   {r4, pc}\n");
}

uint64_t arm_sdiv64(int64_t n, int64_t d) {
    const int neg_q = (n < 0) != (d < 0), neg_r = n < 0;
    const uint64_t q = arm_udiv64(n < 0 ? (uint64_t)(-n) : (uint64_t)n,
                                  d < 0 ? (uint64_t)(-d) : (uint64_t)d);
    if (neg_r) arm_divmod_rem = (uint64_t)(-(int64_t)arm_divmod_rem);
    return neg_q ? (uint64_t)(-(int64_t)q) : q;
}

__attribute__((naked)) void __aeabi_ldivmod(void) {
    __asm__ volatile(
        "push  {r4, lr}\n"
        "bl    arm_sdiv64\n"
        "movw  r4, :lower16:arm_divmod_rem\n"
        "movt  r4, :upper16:arm_divmod_rem\n"
        "ldr   r2, [r4]\n"
        "ldr   r3, [r4, #4]\n"
        "pop   {r4, pc}\n");
}

#endif

static void hex64(uint64_t v) {
    char b[19];
    b[0] = '0'; b[1] = 'x';
    for (int i = 0; i < 16; i++) b[2 + i] = "0123456789abcdef"[(v >> ((15 - i) * 4)) & 15];
    b[18] = '\n';
    out(b, 19);
}
static void hex32(uint32_t v) { hex64((uint64_t)v); }
static void say(const char* s) {
    uint32_t n = 0;
    while (s[n]) n++;
    out(s, n);
}
// Signed decimal. Shared rather than copied per test, because a bug in the
// formatter is invisible to differential testing -- the host build runs the *same*
// formatter, so the diff agrees and says nothing. Only the harness itself can be
// read.
static void dec(int64_t v) {
    char b[25];
    int i = 24;
    b[24] = '\n';
    uint64_t u = v < 0 ? (uint64_t)(-v) : (uint64_t)v;
    if (!u) b[--i] = '0';
    while (u) { b[--i] = (char)('0' + (uint32_t)(u % 10)); u /= 10; }
    if (v < 0) b[--i] = '-';
    out(b + i, (uint32_t)(25 - i));
}
