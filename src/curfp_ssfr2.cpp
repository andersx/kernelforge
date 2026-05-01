/*
 * curfpSsfr2 — Symmetric Rank-2 update in RFP format (single precision)
 *
 * Computes:  C := alpha * x * y^T + alpha * y * x^T + C
 *
 * Only TRANSR=N (CURFP_OP_N) with UPLO=L (CURFP_FILL_MODE_LOWER) is
 * implemented.  All other combinations return CURFP_STATUS_NOT_SUPPORTED.
 *
 * Implementation uses the same 2-block sub-decomposition as curfp_ssfrk.cpp
 * (LAPACK ssfrk / ssyr2k theory), replacing:
 *   2× cublasSsyrk  →  2× cublasSsyr2   (symmetric sub-blocks)
 *   1× cublasSgemm  →  2× cublasSger    (rectangular off-diagonal block)
 *
 * Sub-block layout (TRANSR=N, UPLO=L):
 *
 *   Odd n  (n1=ceil(n/2), n2=floor(n/2), ldc=n):
 *     Block1  LOWER n1×n1 at C+0    → cublasSsyr2 on x[0..n1-1],  y[0..n1-1]
 *     Block2  UPPER n2×n2 at C+n    → cublasSsyr2 on x[n1..n-1], y[n1..n-1]
 *     Offdiag n2×n1     at C+n1    → sger(x[n1..n-1], y[0..n1-1])
 *                                   + sger(y[n1..n-1], x[0..n1-1])
 *
 *   Even n (nk=n/2, ldc=n+1):
 *     Block1  LOWER nk×nk at C+1    → cublasSsyr2 on x[0..nk-1],  y[0..nk-1]
 *     Block2  UPPER nk×nk at C+0    → cublasSsyr2 on x[nk..n-1], y[nk..n-1]
 *     Offdiag nk×nk     at C+nk+1  → sger(x[nk..n-1], y[0..nk-1])
 *                                   + sger(y[nk..n-1], x[0..nk-1])
 */

#include "curfp_internal.h"

curfpStatus_t curfpSsfr2(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const float     *alpha,
    const float     *x,
    int              incx,
    const float     *y,
    int              incy,
    float           *arf)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || incx == 0 || incy == 0) return CURFP_STATUS_INVALID_VALUE;
    if (!alpha || !x || !y || !arf)       return CURFP_STATUS_INVALID_VALUE;
    if (n == 0 || *alpha == 0.0f)         return CURFP_STATUS_SUCCESS;

    /* Only TRANSR=N, UPLO=L supported */
    if (transr != CURFP_OP_N || uplo != CURFP_FILL_MODE_LOWER)
        return CURFP_STATUS_INVALID_VALUE;

    cublasHandle_t cb = handle->cublas;

    const int nisodd = (n & 1);

    if (nisodd) {
        /* ------------------------------------------------------------------ *
         * Odd n: n1 = ceil(n/2), n2 = floor(n/2), ldc = n                   *
         * ------------------------------------------------------------------ */
        const int n1  = n - n / 2;   /* ceil */
        const int n2  = n / 2;       /* floor */
        const int ldc = n;

        /* Pointers to the second half of x and y */
        const float *x2 = x + (long)n1 * incx;
        const float *y2 = y + (long)n1 * incy;

        /* Block1: LOWER n1×n1 at C+0 */
        CURFP_CHECK_CUBLAS(cublasSsyr2(cb,
            CUBLAS_FILL_MODE_LOWER, n1,
            alpha,
            x,   incx,
            y,   incy,
            arf + 0, ldc));

        /* Block2: UPPER n2×n2 at C+n */
        if (n2 > 0) {
            CURFP_CHECK_CUBLAS(cublasSsyr2(cb,
                CUBLAS_FILL_MODE_UPPER, n2,
                alpha,
                x2, incx,
                y2, incy,
                arf + n, ldc));
        }

        /* Off-diagonal n2×n1 at C+n1:  x2*y1^T + y2*x1^T */
        if (n1 > 0 && n2 > 0) {
            CURFP_CHECK_CUBLAS(cublasSger(cb,
                n2, n1,
                alpha,
                x2, incx,
                y,  incy,
                arf + n1, ldc));

            CURFP_CHECK_CUBLAS(cublasSger(cb,
                n2, n1,
                alpha,
                y2, incy,
                x,  incx,
                arf + n1, ldc));
        }

    } else {
        /* ------------------------------------------------------------------ *
         * Even n: nk = n/2, ldc = n+1                                        *
         * ------------------------------------------------------------------ */
        const int nk  = n / 2;
        const int ldc = n + 1;

        /* Pointers to the second half of x and y */
        const float *x2 = x + (long)nk * incx;
        const float *y2 = y + (long)nk * incy;

        /* Block1: LOWER nk×nk at C+1 */
        CURFP_CHECK_CUBLAS(cublasSsyr2(cb,
            CUBLAS_FILL_MODE_LOWER, nk,
            alpha,
            x,   incx,
            y,   incy,
            arf + 1, ldc));

        /* Block2: UPPER nk×nk at C+0 */
        CURFP_CHECK_CUBLAS(cublasSsyr2(cb,
            CUBLAS_FILL_MODE_UPPER, nk,
            alpha,
            x2, incx,
            y2, incy,
            arf + 0, ldc));

        /* Off-diagonal nk×nk at C+nk+1:  x2*y1^T + y2*x1^T */
        CURFP_CHECK_CUBLAS(cublasSger(cb,
            nk, nk,
            alpha,
            x2, incx,
            y,  incy,
            arf + nk + 1, ldc));

        CURFP_CHECK_CUBLAS(cublasSger(cb,
            nk, nk,
            alpha,
            y2, incy,
            x,  incx,
            arf + nk + 1, ldc));
    }

    return CURFP_STATUS_SUCCESS;
}
