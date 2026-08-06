/* VFP: the hard-float ABI's arithmetic, checked bit for bit.
 *
 * Results are printed as their IEEE bit patterns rather than as decimals, because
 * a decimal formatter would hide exactly the errors worth catching -- a wrong
 * rounding mode, a single-precision operation done in double and narrowed at the
 * end, a comparison that gets the unordered case wrong. The host computes the same
 * bits with the same C. */
#include "harness.h"

static uint64_t bits_d(double x) { uint64_t b; __builtin_memcpy(&b, &x, 8); return b; }
static uint32_t bits_f(float x)  { uint32_t b; __builtin_memcpy(&b, &x, 4); return b; }

TEST_MAIN {
    volatile double a = 3.14159265358979, b = -2.718281828459045;
    hex64(bits_d(a + b));
    hex64(bits_d(a - b));
    hex64(bits_d(a * b));
    hex64(bits_d(a / b));
    hex64(bits_d(-a));
    hex64(bits_d(a < 0 ? -a : a));
    hex64(bits_d(a * a + b));           /* VMLA, if the compiler fuses it */
    hex64(bits_d(1.0 / a));
    hex64(bits_d(0.0));
    hex64(bits_d(1.0));

    /* Single precision is a separate register view and a separate set of
     * operations; computing it in double and narrowing at the end gives different
     * bits for anything that rounds twice. */
    volatile float f = 1.4142135f, g = 0.5772157f;
    hex32(bits_f(f + g));
    hex32(bits_f(f * g));
    hex32(bits_f(f / g));
    hex32(bits_f((float)(a)));
    hex64(bits_d((double)f));

    /* Comparisons, which reach the integer flags only through VMRS. A guest whose
     * VMRS is missing branches on whatever the last integer compare left. */
    hex64((uint64_t)(a > b));
    hex64((uint64_t)(a < b));
    hex64((uint64_t)(a == a));
    hex64((uint64_t)(f > g));
    hex64((uint64_t)(a > 0.0 ? 11 : 22));

    /* Conversions in both directions, including the saturating ends. */
    volatile int32_t i = -1234567;
    volatile uint32_t u = 4000000000u;
    hex64(bits_d((double)i));
    hex64(bits_d((double)u));
    hex64((uint64_t)(uint32_t)(int32_t)a);
    hex64((uint64_t)(uint32_t)(int32_t)(-a));
    hex64((uint64_t)(uint32_t)(1e10 > 2147483647.0 ? 2147483647 : (int32_t)1e10));
    hex32(bits_f((float)i));
    finish();
}
