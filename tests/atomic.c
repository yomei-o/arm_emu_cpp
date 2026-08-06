/* The exclusive monitor: LDREX/STREX, and the kernel helper page's compare-and-swap.
 *
 * Single-threaded, so nothing here can *fail* by racing -- what it checks is that
 * the instructions compute the right values and that a store-exclusive after a
 * load-exclusive to the same address succeeds. The racing case is thread.c, which
 * is the one that fails if STREX always succeeds. */
#include "harness.h"

static volatile uint32_t cell;
static volatile uint64_t cell64;

TEST_MAIN {
    cell = 100;
    hex64(__atomic_fetch_add(&cell, 7, __ATOMIC_SEQ_CST));
    hex64(cell);
    hex64(__atomic_fetch_sub(&cell, 3, __ATOMIC_SEQ_CST));
    hex64(cell);
    hex64(__atomic_fetch_or(&cell, 0xF0, __ATOMIC_SEQ_CST));
    hex64(cell);
    hex64(__atomic_fetch_and(&cell, 0x3C, __ATOMIC_SEQ_CST));
    hex64(cell);
    hex64(__atomic_fetch_xor(&cell, 0xFF, __ATOMIC_SEQ_CST));
    hex64(cell);
    hex64(__atomic_exchange_n(&cell, 42, __ATOMIC_SEQ_CST));
    hex64(cell);

    uint32_t want = 42;
    hex64((uint64_t)__atomic_compare_exchange_n(&cell, &want, 99, 0,
                                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    hex64(cell);
    want = 1;                                   /* now a mismatch: must fail */
    hex64((uint64_t)__atomic_compare_exchange_n(&cell, &want, 7, 0,
                                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    hex64(cell);
    hex64(want);                                /* the caller is told what was there */

    /* 64-bit, which is LDREXD/STREXD -- a register *pair*, and a monitor that
     * ignores the second word leaves half of it stale. */
    cell64 = 0x0123456789ABCDEFull;
    hex64(__atomic_load_n(&cell64, __ATOMIC_SEQ_CST));
    __atomic_store_n(&cell64, 0xFEDCBA9876543210ull, __ATOMIC_SEQ_CST);
    hex64(cell64);
    finish();
}
