/*
 * curfpSsfrk — Symmetric Rank-K update in RFP format (single precision)
 *
 * Direct translation of LAPACK ssfrk.f to CUDA using cuBLAS.
 *
 * There are 8 RFP storage variants: N parity × transr × uplo.
 * Each variant computes 2× cublasSsyrk + 1× cublasSgemm on sub-blocks.
 *
 * Instead of 8 copy-pasted code blocks we parameterize the sub-block layout
 * in a single struct and run one generic code path.
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Per-case parameters for the 3 cuBLAS calls.
 *
 * Notation (all pointer offsets in elements, not bytes):
 *   C1   = C + off1   (first  ssyrk output: off-diagonal block)
 *   C2   = C + off2   (second ssyrk output: diagonal block)
 *   Cg   = C + offg   (sgemm output)
 *   A2_n = A + a2_notrans  (second A block when trans=N, row offset)
 *   A2_t = A + a2_trans    (second A block when trans=T, col offset × lda)
 *
 * sgemm computes: Cg := alpha * op(Ag1) * op(Ag2)^T + beta * Cg
 * where (Ag1, Ag2) = (A1, A2) when gemm_a1_first=1, else (A2, A1).
 * ------------------------------------------------------------------------- */
typedef struct {
    /* ssyrk1: fill1, dim1 */
    cublasFillMode_t  fill1;
    int               dim1;
    long              off1;    /* C offset for ssyrk1 */

    /* ssyrk2: fill2, dim2 */
    cublasFillMode_t  fill2;
    int               dim2;
    long              off2;    /* C offset for ssyrk2 */

    /* sgemm */
    int               gemm_m, gemm_n;   /* output dims of sgemm */
    long              offg;             /* C offset for sgemm */
    int               gemm_a1_first;   /* 1: (A1,A2), 0: (A2,A1) as sgemm (op(A), op(B)) */

    /* A second-block offset */
    long              a2_notrans;  /* A + a2_notrans when trans=N */
    long              a2_trans_k;  /* A + a2_trans_k * lda when trans=T */

    /* leading dimension of RFP sub-blocks */
    int               ldc;
} ssfrk_params_t;

static void get_ssfrk_params(int nisodd, int normaltransr, int lower,
                              int n, int n1, int n2, int nk,
                              ssfrk_params_t *p)
{
    if (nisodd) {
        if (normaltransr) {
            p->ldc = n;
            if (lower) {
                /* Case 1: odd, TRANSR=N, UPLO=L */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = 0;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n;
                p->gemm_m = n2; p->gemm_n = n1; p->offg = n1;
                p->gemm_a1_first = 0; /* sgemm(A2, A1) */
                p->a2_notrans = n1; p->a2_trans_k = n1;
            } else {
                /* Case 2: odd, TRANSR=N, UPLO=U */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = n1; p->off1 = n2;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = n2; p->off2 = n1;
                p->gemm_m = n1; p->gemm_n = n2; p->offg = 0;
                p->gemm_a1_first = 1; /* sgemm(A1, A2) */
                p->a2_notrans = n2 - 1; p->a2_trans_k = n2 - 1;
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 3: odd, TRANSR=T, UPLO=L, ldc=n1 */
                p->ldc = n1;
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = 0;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = 1;
                p->gemm_m = n1; p->gemm_n = n2; p->offg = (long)n1 * n1;
                p->gemm_a1_first = 1; /* sgemm(A1, A2) */
                p->a2_notrans = n1; p->a2_trans_k = n1;
            } else {
                /* Case 4: odd, TRANSR=T, UPLO=U, ldc=n2 */
                p->ldc = n2;
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = n1; p->off1 = (long)n2 * n2;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = n2; p->off2 = (long)n1 * n2;
                p->gemm_m = n2; p->gemm_n = n1; p->offg = 0;
                p->gemm_a1_first = 0; /* sgemm(A2, A1) */
                p->a2_notrans = n1; p->a2_trans_k = n1;
            }
        }
    } else {
        /* Even N */
        if (normaltransr) {
            p->ldc = n + 1;
            if (lower) {
                /* Case 5: even, TRANSR=N, UPLO=L */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = 1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = 0;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = nk + 1;
                p->gemm_a1_first = 0; /* sgemm(A2, A1) */
                p->a2_notrans = nk; p->a2_trans_k = nk;
            } else {
                /* Case 6: even, TRANSR=N, UPLO=U */
                p->fill1 = CUBLAS_FILL_MODE_LOWER; p->dim1 = nk; p->off1 = nk + 1;
                p->fill2 = CUBLAS_FILL_MODE_UPPER; p->dim2 = nk; p->off2 = nk;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = 0;
                p->gemm_a1_first = 1; /* sgemm(A1, A2) */
                p->a2_notrans = nk; p->a2_trans_k = nk;
            }
        } else {
            /* TRANSR=T */
            p->ldc = nk;
            if (lower) {
                /* Case 7: even, TRANSR=T, UPLO=L */
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = nk;
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = 0;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = (long)(nk + 1) * nk;
                p->gemm_a1_first = 1; /* sgemm(A1, A2) */
                p->a2_notrans = nk; p->a2_trans_k = nk;
            } else {
                /* Case 8: even, TRANSR=T, UPLO=U */
                p->fill1 = CUBLAS_FILL_MODE_UPPER; p->dim1 = nk; p->off1 = (long)nk * (nk + 1);
                p->fill2 = CUBLAS_FILL_MODE_LOWER; p->dim2 = nk; p->off2 = (long)nk * nk;
                p->gemm_m = nk; p->gemm_n = nk; p->offg = 0;
                p->gemm_a1_first = 0; /* sgemm(A2, A1) */
                p->a2_notrans = nk; p->a2_trans_k = nk;
            }
        }
    }
}

curfpStatus_t curfpSsfrk(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpOperation_t trans,
    int              n,
    int              k,
    const float     *alpha,
    const float     *A,
    int              lda,
    const float     *beta,
    float           *C)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || k < 0 || lda < 1) return CURFP_STATUS_INVALID_VALUE;
    if (!alpha || !beta)            return CURFP_STATUS_INVALID_VALUE;
    if (n == 0)                     return CURFP_STATUS_SUCCESS;

    /* Quick returns */
    if ((*alpha == 0.0f || k == 0) && *beta == 1.0f)
        return CURFP_STATUS_SUCCESS;

    if (*alpha == 0.0f && *beta == 0.0f) {
        long ntotal = (long)n * (n + 1) / 2;
        CURFP_CHECK_CUDA(cudaMemset(C, 0, ntotal * sizeof(float)));
        return CURFP_STATUS_SUCCESS;
    }

    cublasHandle_t cb = handle->cublas;

    const int notrans      = (trans  == CURFP_OP_N);
    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd       = (n % 2 != 0);

    const cublasOperation_t opN = CUBLAS_OP_N;
    const cublasOperation_t opT = CUBLAS_OP_T;
    const cublasOperation_t opA = notrans ? opN : opT;
    const cublasOperation_t opAt = notrans ? opT : opN;

    /* Sub-block dimensions */
    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        nk = n / 2; n1 = nk; n2 = nk;
    }

    ssfrk_params_t p;
    get_ssfrk_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    /* A block pointers */
    const float *A1 = A;
    const float *A2 = notrans ? A + p.a2_notrans
                              : A + p.a2_trans_k * (long)lda;

    /* ssyrk on block 1 */
    CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
        p.fill1, opA,
        p.dim1, k, alpha, A1, lda, beta, C + p.off1, p.ldc));

    /* ssyrk on block 2 */
    CURFP_CHECK_CUBLAS(cublasSsyrk(cb,
        p.fill2, opA,
        p.dim2, k, alpha, A2, lda, beta, C + p.off2, p.ldc));

    /* sgemm on off-diagonal block */
    const float *Ag1 = p.gemm_a1_first ? A1 : A2;
    const float *Ag2 = p.gemm_a1_first ? A2 : A1;
    CURFP_CHECK_CUBLAS(cublasSgemm(cb,
        opA, opAt,
        p.gemm_m, p.gemm_n, k,
        alpha, Ag1, lda, Ag2, lda,
        beta,  C + p.offg, p.ldc));

    return CURFP_STATUS_SUCCESS;
}
