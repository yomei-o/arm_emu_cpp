/* The parallel (SIMD-within-a-register) add and subtract group, and the GE flags.
 *
 * These are the instructions ARMv6 had instead of NEON, and glibc's ARM string
 * routines still use them: `uqsub8` against a register of zeros leaves a non-zero
 * byte exactly where the input had one, and `sel` picks the lane out of the GE
 * flags. The two only work as a pair, which is why the last case here is that exact
 * idiom rather than one more arithmetic check.
 *
 * The subtle part -- and the reason every case prints GE as well as its result -- is
 * that only the *plain* forms write GE. The saturating and halving forms leave it
 * alone, so a test that compared only results would pass with GE stuck at whatever
 * the last plain operation left behind, and `sel` would keep working by accident.
 *
 * Each case names the instruction *and* its four properties. The ARM build assembles
 * the mnemonic; the host build hands the properties to one reference function. That
 * way the two sides cannot drift into agreeing through a shared special case, which
 * is what happens when a reference is written by parsing the mnemonic.
 *
 * GE is read back with `mrs` inside the same asm block, because it belongs to the
 * instruction just executed and anything scheduled in between could clobber it. Its
 * bits are CPSR<19:16>.
 */
#include "harness.h"

/* Lane width, signedness, variant, and which lanes subtract. */
#define W8   1
#define W16  0
#define SGN  0
#define UNS  1
#define PLAIN 0
#define SAT   1
#define HALVE 2
#define F_ADD 0     /* every lane adds        */
#define F_SUB 1     /* every lane subtracts   */
#define F_ASX 2     /* low lane subtracts, operands crossed  */
#define F_SAX 3     /* high lane subtracts, operands crossed */

static uint32_t g_ge;

static void show(const char* name, uint32_t res, uint32_t ge) {
    say(name);
    say(" = ");
    hex32(res);
    say("     ge = ");
    hex32(ge);
}

#ifdef ARM_NATIVE

static uint32_t par(int bytes, int uns, int kind, int form, uint32_t a, uint32_t b) {
    const int lanes = bytes ? 4 : 2, bits = bytes ? 8 : 16;
    const uint32_t mask = bytes ? 0xFFu : 0xFFFFu;
    uint32_t res = 0, ge = 0;
    for (int i = 0; i < lanes; i++) {
        const int sh = i * bits;
        const int cross = (form == F_ASX || form == F_SAX);
        const int bsh = cross ? (i == 0 ? 16 : 0) : sh;
        const uint32_t au = (a >> sh) & mask, bu = (b >> bsh) & mask;
        const int32_t as = (int32_t)(au << (32 - bits)) >> (32 - bits);
        const int32_t bs = (int32_t)(bu << (32 - bits)) >> (32 - bits);
        const int sub = form == F_SUB || (form == F_ASX && i == 0) ||
                        (form == F_SAX && i == 1);
        const int64_t truth = uns ? (sub ? (int64_t)au - bu : (int64_t)au + bu)
                                  : (sub ? (int64_t)as - bs : (int64_t)as + bs);
        uint32_t lane;
        if (kind == SAT) {
            const int64_t hi = uns ? (int64_t)mask : ((int64_t)1 << (bits - 1)) - 1;
            const int64_t lo = uns ? 0 : -((int64_t)1 << (bits - 1));
            lane = (uint32_t)(truth > hi ? hi : (truth < lo ? lo : truth)) & mask;
        } else if (kind == HALVE) {
            lane = (uint32_t)(truth >> 1) & mask;
        } else {
            lane = (uint32_t)truth & mask;
            /* Signed, and unsigned subtract: "the true result was non-negative".
             * Unsigned *add*: "there was a carry out" -- so one whose result fits
             * sets no GE bit at all, which is the opposite of what the name reads
             * like. GE is per byte even for the halfword forms, because SEL is. */
            const int set = (uns && !sub) ? (truth >= ((int64_t)1 << bits)) : (truth >= 0);
            if (set) {
                const int first = bytes ? i : i * 2, count = bytes ? 1 : 2;
                for (int k = 0; k < count; k++) ge |= 1u << (first + k);
            }
        }
        res |= lane << sh;
    }
    if (kind == PLAIN) g_ge = ge;
    return res;
}

/* The result goes into a temporary first. Written as one call, `par(...)` and the
 * `g_ge` argument are evaluated in an order C does not specify -- and `par` is what
 * sets `g_ge`, so half the cases printed the *previous* case's flags. The ARM build
 * has no such freedom, so the two disagreed by one line and the diff read as a
 * missing instruction. */
#define P(op, bytes, uns, kind, form, av, bv) do {                                 \
        const uint32_t r_ = par((bytes), (uns), (kind), (form),                    \
                                (uint32_t)(av), (uint32_t)(bv));                   \
        show(#op, r_, g_ge);                                                       \
    } while (0)

static uint32_t sel_(uint32_t a, uint32_t b, uint32_t ge) {
    uint32_t r = 0;
    for (int i = 0; i < 4; i++) r |= ((ge >> i) & 1 ? a : b) & (0xFFu << (i * 8));
    return r;
}

#else

#define P(op, bytes, uns, kind, form, av, bv) do {                                 \
        uint32_t r_, c_;                                                           \
        __asm__ volatile(#op " %0, %2, %3\n\tmrs %1, cpsr"                         \
                         : "=&r"(r_), "=r"(c_)                                     \
                         : "r"((uint32_t)(av)), "r"((uint32_t)(bv)) : "cc");       \
        if ((kind) == PLAIN) g_ge = (c_ >> 16) & 0xF;                              \
        show(#op, r_, g_ge);                                                       \
    } while (0)

static uint32_t sel_(uint32_t a, uint32_t b, uint32_t ge) {
    uint32_t r_;
    /* Put GE back where SEL reads it, then select. `APSR_g` is the write that
     * carries the GE field and nothing else. */
    __asm__ volatile("msr APSR_g, %3\n\tsel %0, %1, %2"
                     : "=r"(r_) : "r"(a), "r"(b), "r"(ge << 16) : "cc");
    return r_;
}

#endif

TEST_MAIN {
    /* Chosen so every lane lands somewhere different: one that overflows, one that
     * borrows, one negative, one exact. */
    const uint32_t a = 0x7F01FE80u, b = 0x0102FF7Fu;
    const uint32_t c = 0x8000FFFFu, d = 0x00017FFFu;

    P(sadd8,   W8,  SGN, PLAIN, F_ADD, a, b);
    P(uadd8,   W8,  UNS, PLAIN, F_ADD, a, b);
    P(ssub8,   W8,  SGN, PLAIN, F_SUB, a, b);
    P(usub8,   W8,  UNS, PLAIN, F_SUB, a, b);
    P(sadd16,  W16, SGN, PLAIN, F_ADD, c, d);
    P(uadd16,  W16, UNS, PLAIN, F_ADD, c, d);
    P(ssub16,  W16, SGN, PLAIN, F_SUB, c, d);
    P(usub16,  W16, UNS, PLAIN, F_SUB, c, d);

    P(qadd8,   W8,  SGN, SAT,   F_ADD, a, b);
    P(uqadd8,  W8,  UNS, SAT,   F_ADD, a, b);
    P(qsub8,   W8,  SGN, SAT,   F_SUB, a, b);
    P(uqsub8,  W8,  UNS, SAT,   F_SUB, a, b);
    P(qadd16,  W16, SGN, SAT,   F_ADD, c, d);
    P(uqadd16, W16, UNS, SAT,   F_ADD, c, d);
    P(qsub16,  W16, SGN, SAT,   F_SUB, c, d);
    P(uqsub16, W16, UNS, SAT,   F_SUB, c, d);

    P(shadd8,  W8,  SGN, HALVE, F_ADD, a, b);
    P(uhadd8,  W8,  UNS, HALVE, F_ADD, a, b);
    P(shsub8,  W8,  SGN, HALVE, F_SUB, a, b);
    P(uhsub8,  W8,  UNS, HALVE, F_SUB, a, b);
    P(shadd16, W16, SGN, HALVE, F_ADD, c, d);
    P(uhadd16, W16, UNS, HALVE, F_ADD, c, d);
    P(shsub16, W16, SGN, HALVE, F_SUB, c, d);
    P(uhsub16, W16, UNS, HALVE, F_SUB, c, d);

    P(sasx,    W16, SGN, PLAIN, F_ASX, c, d);
    P(ssax,    W16, SGN, PLAIN, F_SAX, c, d);
    P(uasx,    W16, UNS, PLAIN, F_ASX, c, d);
    P(usax,    W16, UNS, PLAIN, F_SAX, c, d);

    /* glibc's idiom, end to end: `usub8` sets GE for every byte of the word that did
     * not borrow, and SEL turns that into a mask. */
    P(usub8, W8, UNS, PLAIN, F_SUB, 0x41004243u, 0x01010101u);
    say("sel        = ");
    hex32(sel_(0xFFFFFFFFu, 0x00000000u, g_ge));
    finish();
}
