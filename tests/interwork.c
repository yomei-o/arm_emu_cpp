/* ARM and Thumb calling each other.
 *
 * A real armhf system is both instruction sets at once: glibc is built for ARM,
 * but plenty of what it links against is Thumb, and every call between them is an
 * interworking branch that carries the target's instruction set in bit 0 of the
 * address. Getting that wrong does not fault -- it decodes the callee's first two
 * ARM instructions as one Thumb pair, or the reverse, and computes.
 *
 * Two mutually recursive functions, forced into different states, so every call
 * and every return crosses. */
#include "harness.h"

#ifdef ARM_NATIVE
static uint32_t f_a(uint32_t x);
static uint32_t f_t(uint32_t x);
static uint32_t f_a(uint32_t x) { return x < 2 ? x : f_t(x - 1) + f_t(x - 2); }
static uint32_t f_t(uint32_t x) { return x < 2 ? x : f_a(x - 1) + f_a(x - 2); }
#else
__attribute__((target("arm")))   static uint32_t f_a(uint32_t x);
__attribute__((target("thumb"))) static uint32_t f_t(uint32_t x);
__attribute__((target("arm")))   static uint32_t f_a(uint32_t x) { return x < 2 ? x : f_t(x - 1) + f_t(x - 2); }
__attribute__((target("thumb"))) static uint32_t f_t(uint32_t x) { return x < 2 ? x : f_a(x - 1) + f_a(x - 2); }
#endif

TEST_MAIN {
    for (uint32_t k = 0; k <= 20; k++) dec((int64_t)f_a(k));
    for (uint32_t k = 0; k <= 20; k++) dec((int64_t)f_t(k));
    finish();
}
