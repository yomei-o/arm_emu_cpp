/* The integer core, checked against the host computing the same expressions.
 *
 * Written in 64-bit types on purpose. A 32-bit ARM has no 64-bit register, so
 * every line below becomes a *sequence* -- ADDS/ADC, SUBS/SBC, UMULL/MLA, a
 * shift built from two shifts and an ORR -- and each of those depends on the
 * carry flag being what the architecture says it is. On AArch64 the same source
 * compiles to one instruction per line and tests almost nothing. Here it is the
 * densest available test of the flags. */
#include "harness.h"
TEST_MAIN {
    volatile int64_t a = -1234567890123LL, b = 7654321;
    hex64((uint64_t)(a + b));
    hex64((uint64_t)(a - b));
    hex64((uint64_t)(a * b));
    hex64((uint64_t)(a / b));
    hex64((uint64_t)(a % b));
    hex64((uint64_t)a / (uint64_t)b);
    hex64((uint64_t)a % (uint64_t)b);
    hex64((uint64_t)(a >> 13));
    hex64((uint64_t)a >> 13);
    hex64((uint64_t)(a << 7));
    hex64((uint64_t)(a << 40));                 /* a shift past the halfway point */
    hex64((uint64_t)a >> 40);
    hex64((uint64_t)(a ^ b));
    hex64((uint64_t)(a & 0x00ff00ff00ff00ffLL));
    hex64((uint64_t)(a | 0x0f0f0f0fLL));
    hex64((uint64_t)(~a));
    hex64((uint64_t)(-a));

    volatile int32_t x = -70000, y = 300;
    hex64((uint32_t)(x / y));
    hex64((uint32_t)(x % y));
    hex64((uint32_t)(x * y));
    hex64((uint32_t)(x >> 3));
    hex64((uint32_t)x >> 3);
    hex64((uint32_t)(x + y));
    hex64((uint32_t)(x ^ y));
    hex64((uint32_t)x / (uint32_t)y);
    hex64((uint32_t)x % (uint32_t)y);

    /* Comparisons, which is where the subtract-carry convention shows up: on ARM
     * the carry after a subtract means "no borrow", the opposite of the naive
     * reading, and an unsigned compare is exactly that flag. */
    hex64((uint64_t)(a < b));
    hex64((uint64_t)((uint64_t)a < (uint64_t)b));
    hex64((uint64_t)(a <= b));
    hex64((uint64_t)(a > b));
    hex64((uint64_t)(x < y ? 111 : 222));
    hex64((uint64_t)((uint32_t)x < (uint32_t)y ? 111 : 222));
    hex64((uint64_t)(a == -1234567890123LL));

    hex64((uint64_t)__builtin_clzl((uint32_t)b));
    hex64((uint64_t)__builtin_ctzl((uint32_t)b));
    hex64((uint64_t)__builtin_bswap32((uint32_t)x));
    hex64((uint64_t)__builtin_bswap16((uint16_t)x));

    /* The widening multiply, which is UMULL/SMULL and its accumulate forms -- the
     * whole of how a 32-bit machine multiplies two 64-bit numbers. */
    volatile uint32_t p = 0xdeadbeefu, q = 0xcafef00du;
    hex64((uint64_t)p * q);
    hex64((uint64_t)(int64_t)(int32_t)p * (int32_t)q);
    hex64((uint64_t)((int64_t)x * (int64_t)y));

    /* Rotates and byte extraction: what a hash loop is made of, and where the
     * barrel shifter's ROR and the sign/zero extends live. */
    volatile uint32_t h = 0x12345678u;
    hex64((uint64_t)((h >> 7) | (h << 25)));
    hex64((uint64_t)(uint8_t)(h >> 8));
    hex64((uint64_t)(int8_t)(h >> 16));
    hex64((uint64_t)(int16_t)h);
    finish();
}
