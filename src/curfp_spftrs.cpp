/*
 * curfpSpftrs — Triangular solve using RFP Cholesky factor (single precision)
 *
 * Direct translation of LAPACK spftrs.f to CUDA using cuBLAS STRSM/SGEMM.
 *
 * Given the RFP Cholesky factor from curfpSpftrf, solves A * X = B in-place.
 *   uplo=L:  (L * L^T) * X = B
 *   uplo=U:  (U^T * U) * X = B
 *
 * Each of the 8 RFP storage variants (N parity × transr × uplo) requires
 * 4 STRSM + 2 SGEMM calls, following LAPACK spftrs.f exactly.
 * All pointer offsets are converted from 1-based Fortran to 0-based C.
 *
 * B is (n × nrhs) in column-major with leading dimension ldb.
 * B1 and B2 denote the two n/2-row blocks of B (possibly swapped per case).
 */

#include "curfp_internal.h"

/* -------------------------------------------------------------------------
 * Parameters for the 6 cuBLAS calls (4 TRSM + 2 GEMM) per case.
 *
 * Following LAPACK spftrs.f structure:
 *   1. STRSM: forward solve block 11        (b_off = b_fwd_off)
 *   2. SGEMM: off-diagonal update (forward) (A off-diag, result → other block)
 *   3. STRSM: forward solve block 22        (b_off = b_bwd_off)
 *   4. STRSM: backward solve block 22
 *   5. SGEMM: off-diagonal update (backward)
 *   6. STRSM: backward solve block 11
 * ------------------------------------------------------------------------- */
typedef struct {
    int  dim_fwd, dim_bwd;   /* sub-block sizes (nk/n1/n2) */

    /* TRSM 1 & 6: block 11 (forward N, backward T or vice versa) */
    cublasFillMode_t  t16_fill;
    long              t16_a_off;
    int               t16_lda;
    cublasOperation_t t1_op;     /* op for forward */
    cublasOperation_t t6_op;     /* op for backward */
    long              t16_b_off; /* B offset for block 11 */

    /* TRSM 3 & 4: block 22 */
    cublasFillMode_t  t34_fill;
    long              t34_a_off;
    int               t34_lda;
    cublasOperation_t t3_op;     /* op for forward */
    cublasOperation_t t4_op;     /* op for backward */
    long              t34_b_off; /* B offset for block 22 */

    /* GEMM 2: off-diagonal forward update (result goes to bwd block) */
    cublasOperation_t g2_op;
    long              g2_a_off;
    int               g2_lda;
    /* GEMM 5: off-diagonal backward update (result goes to fwd block) */
    cublasOperation_t g5_op;
    long              g5_a_off;
    /* (g5 uses same lda and off-diagonal as g2, just transposed op) */
} spftrs_params_t;

static void get_spftrs_params(int nisodd, int normaltransr, int lower,
                               int n, int n1, int n2, int nk,
                               spftrs_params_t *p)
{
    /* Convert LAPACK 1-based array indices to 0-based C offsets.
     * B(1) → B+0, B(N1+1) → B+N1, B(NK+1) → B+NK, etc.             */

    if (nisodd) {
        if (normaltransr) {
            /* lda_rfp = N */
            if (lower) {
                /* Case 1: odd, TRANSR=N, UPLO=L
                 * 1. STRSM('L','L','N','N', N1,NRHS, 1, A(1),   N, B(1),    LDB)
                 * 2. SGEMM('N','N', N2,NRHS,N1,-1,   A(N1+1),N, B(1),LDB, 1,B(N1+1),LDB)
                 * 3. STRSM('L','U','T','N', N2,NRHS, 1, A(N+1), N, B(N1+1), LDB)
                 * 4. STRSM('L','U','N','N', N2,NRHS, 1, A(N+1), N, B(N1+1), LDB)
                 * 5. SGEMM('T','N', N1,NRHS,N2,-1,   A(N1+1),N, B(N1+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','L','T','N', N1,NRHS, 1, A(1),   N, B(1),    LDB) */
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = 0;   p->t16_lda = n;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = n;   p->t34_lda = n;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = n1; p->g2_lda = n;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = n1;
            } else {
                /* Case 2: odd, TRANSR=N, UPLO=U
                 * From STFSM (.NOT.NOTRANS = TRANS='T', forward solve U^T*Y=B):
                 * 1. STRSM('L','L','N','N', N1,NRHS, 1, A(N2+1), N, B(1),    LDB)
                 * 2. SGEMM('T','N', N2,NRHS,N1,-1,   A(1),N,      B(1),LDB, 1,B(N1+1),LDB)
                 * 3. STRSM('L','U','T','N', N2,NRHS, 1, A(N1+1),  N, B(N1+1), LDB)
                 * From STFSM (ELSE = TRANS='N', backward solve U*X=Y):
                 * 4. STRSM('L','U','N','N', N2,NRHS, 1, A(N1+1),  N, B(N1+1), LDB)
                 * 5. SGEMM('N','N', N1,NRHS,N2,-1,   A(1),N,      B(N1+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','L','T','N', N1,NRHS, 1, A(N2+1),  N, B(1),    LDB) */
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = n2;  p->t16_lda = n;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = n1;  p->t34_lda = n;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = 0; p->g2_lda = n;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = 0;
            }
        } else {
            /* TRANSR=T */
            if (lower) {
                /* Case 3: odd, TRANSR=T, UPLO=L, lda_rfp=N1
                 * 1. STRSM('L','U','T','N', N1,NRHS, 1, A(1),      N1, B(1),    LDB)
                 * 2. SGEMM('T','N', N2,NRHS,N1,-1,   A(N1*N1+1),N1, B(1),LDB, 1,B(N1+1),LDB)
                 * 3. STRSM('L','L','N','N', N2,NRHS, 1, A(2),      N1, B(N1+1), LDB)
                 * 4. STRSM('L','L','T','N', N2,NRHS, 1, A(2),      N1, B(N1+1), LDB)
                 * 5. SGEMM('N','N', N1,NRHS,N2,-1,   A(N1*N1+1),N1, B(N1+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','U','N','N', N1,NRHS, 1, A(1),      N1, B(1),    LDB) */
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = 0;          p->t16_lda = n1;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = 1;          p->t34_lda = n1;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = (long)n1*n1; p->g2_lda = n1;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = (long)n1*n1;
            } else {
                /* Case 4: odd, TRANSR=T, UPLO=U, lda_rfp=N2
                 * From STFSM (.NOT.NOTRANS = TRANS='T', forward solve U^T*Y=B):
                 * 1. STRSM('L','U','T','N', N1,NRHS, 1, A(N2*N2+1), N2, B(1),    LDB)
                 * 2. SGEMM('N','N', N2,NRHS,N1,-1,   A(1),N2,       B(1),LDB, 1,B(N1+1),LDB)
                 * 3. STRSM('L','L','N','N', N2,NRHS, 1, A(N1*N2+1), N2, B(N1+1), LDB)
                 * From STFSM (ELSE = TRANS='N', backward solve U*X=Y):
                 * 4. STRSM('L','L','T','N', N2,NRHS, 1, A(N1*N2+1), N2, B(N1+1), LDB)
                 * 5. SGEMM('T','N', N1,NRHS,N2,-1,   A(1),N2,       B(N1+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','U','N','N', N1,NRHS, 1, A(N2*N2+1), N2, B(1),    LDB) */
                p->dim_fwd = n1; p->dim_bwd = n2;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = (long)n2*n2; p->t16_lda = n2;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = (long)n1*n2; p->t34_lda = n2;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = n1;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = 0; p->g2_lda = n2;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = 0;
            }
        }
    } else {
        /* Even N */
        if (normaltransr) {
            /* lda_rfp = N+1 */
            if (lower) {
                /* Case 5: even, TRANSR=N, UPLO=L
                 * 1. STRSM('L','L','N','N', NK,NRHS, 1, A(2),    N+1, B(1),    LDB)
                 * 2. SGEMM('N','N', NK,NRHS,NK,-1,   A(NK+2),N+1, B(1),LDB, 1,B(NK+1),LDB)
                 * 3. STRSM('L','U','T','N', NK,NRHS, 1, A(1),    N+1, B(NK+1), LDB)
                 * 4. STRSM('L','U','N','N', NK,NRHS, 1, A(1),    N+1, B(NK+1), LDB)
                 * 5. SGEMM('T','N', NK,NRHS,NK,-1,   A(NK+2),N+1, B(NK+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','L','T','N', NK,NRHS, 1, A(2),    N+1, B(1),    LDB) */
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = 1;    p->t16_lda = n+1;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = 0;    p->t34_lda = n+1;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = nk+1; p->g2_lda = n+1;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = nk+1;
            } else {
                /* Case 6: even, TRANSR=N, UPLO=U
                 * From STFSM (.NOT.NOTRANS = TRANS='T', forward solve U^T*Y=B):
                 * 1. STRSM('L','L','N','N', NK,NRHS, 1, A(NK+2), N+1, B(1),    LDB)
                 * 2. SGEMM('T','N', NK,NRHS,NK,-1,   A(1),N+1,   B(1),LDB, 1,B(NK+1),LDB)
                 * 3. STRSM('L','U','T','N', NK,NRHS, 1, A(NK+1), N+1, B(NK+1), LDB)
                 * From STFSM (ELSE = TRANS='N', backward solve U*X=Y):
                 * 4. STRSM('L','U','N','N', NK,NRHS, 1, A(NK+1), N+1, B(NK+1), LDB)
                 * 5. SGEMM('N','N', NK,NRHS,NK,-1,   A(1),N+1,   B(NK+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','L','T','N', NK,NRHS, 1, A(NK+2), N+1, B(1),    LDB) */
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_LOWER; p->t16_a_off = nk+1; p->t16_lda = n+1;
                p->t1_op = CUBLAS_OP_N; p->t6_op = CUBLAS_OP_T; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_UPPER; p->t34_a_off = nk;   p->t34_lda = n+1;
                p->t3_op = CUBLAS_OP_T; p->t4_op = CUBLAS_OP_N; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = 0; p->g2_lda = n+1;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = 0;
            }
        } else {
            /* TRANSR=T, lda_rfp=NK */
            if (lower) {
                /* Case 7: even, TRANSR=T, UPLO=L
                 * 1. STRSM('L','U','T','N', NK,NRHS, 1, A(NK+1),      NK, B(1),    LDB)
                 * 2. SGEMM('T','N', NK,NRHS,NK,-1,   A(NK*(NK+1)+1),NK, B(1),LDB, 1,B(NK+1),LDB)
                 * 3. STRSM('L','L','N','N', NK,NRHS, 1, A(1),          NK, B(NK+1), LDB)
                 * 4. STRSM('L','L','T','N', NK,NRHS, 1, A(1),          NK, B(NK+1), LDB)
                 * 5. SGEMM('N','N', NK,NRHS,NK,-1,   A(NK*(NK+1)+1),NK, B(NK+1),LDB, 1,B(1),LDB)
                 * 6. STRSM('L','U','N','N', NK,NRHS, 1, A(NK+1),      NK, B(1),    LDB) */
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = nk;           p->t16_lda = nk;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = 0;            p->t34_lda = nk;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_T; p->g2_a_off = (long)nk*(nk+1); p->g2_lda = nk;
                p->g5_op = CUBLAS_OP_N; p->g5_a_off = (long)nk*(nk+1);
            } else {
                /* Case 8: even, TRANSR=T, UPLO=U  (optimal default case)
                 * From STFSM (TRANS='T', .NOT.NOTRANS=TRUE, forward solve U^T*Y=B):
                 * 1. STRSM('L','U','T','N', K, N, 1, A(K*(K+1)), K, B(0),   LDB)
                 * 2. SGEMM('N','N', K,N,K, -1,       A(0),       K, B(0),   LDB, 1, B(K), LDB)
                 * 3. STRSM('L','L','N','N', K, N, 1, A(K*K),     K, B(K),   LDB)
                 * From STFSM (TRANS='N', .NOT.NOTRANS=FALSE, backward solve U*X=Y):
                 * 4. STRSM('L','L','T','N', K, N, 1, A(K*K),     K, B(K),   LDB)
                 * 5. SGEMM('T','N', K,N,K, -1,       A(0),       K, B(K),   LDB, 1, B(0), LDB)
                 * 6. STRSM('L','U','N','N', K, N, 1, A(K*(K+1)), K, B(0),   LDB) */
                p->dim_fwd = nk; p->dim_bwd = nk;
                p->t16_fill = CUBLAS_FILL_MODE_UPPER; p->t16_a_off = (long)nk*(nk+1); p->t16_lda = nk;
                p->t1_op = CUBLAS_OP_T; p->t6_op = CUBLAS_OP_N; p->t16_b_off = 0;
                p->t34_fill = CUBLAS_FILL_MODE_LOWER; p->t34_a_off = (long)nk*nk;    p->t34_lda = nk;
                p->t3_op = CUBLAS_OP_N; p->t4_op = CUBLAS_OP_T; p->t34_b_off = nk;
                p->g2_op = CUBLAS_OP_N; p->g2_a_off = 0; p->g2_lda = nk;
                p->g5_op = CUBLAS_OP_T; p->g5_a_off = 0;
            }
        }
    }
}

curfpStatus_t curfpSpftrs(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    int              nrhs,
    const float     *A,
    float           *B,
    int              ldb)
{
    CURFP_CHECK_HANDLE(handle);
    if (n < 0 || nrhs < 0 || ldb < 1) return CURFP_STATUS_INVALID_VALUE;
    if (n == 0 || nrhs == 0)          return CURFP_STATUS_SUCCESS;

    cublasHandle_t cb = handle->cublas;

    const int normaltransr = (transr == CURFP_OP_N);
    const int lower        = (uplo   == CURFP_FILL_MODE_LOWER);
    const int nisodd       = (n % 2 != 0);

    int n1 = 0, n2 = 0, nk = 0;
    if (nisodd) {
        if (lower) { n2 = n / 2; n1 = n - n2; }
        else       { n1 = n / 2; n2 = n - n1; }
    } else {
        nk = n / 2; n1 = nk; n2 = nk;
    }

    spftrs_params_t p;
    get_spftrs_params(nisodd, normaltransr, lower, n, n1, n2, nk, &p);

    const float one  =  1.0f;
    const float mone = -1.0f;

    /* Pointers into B for the two row-blocks (column-major, stride ldb per column) */
    float *Bfwd = B + p.t16_b_off;  /* B block for trsm 1&6 */
    float *Bbwd = B + p.t34_b_off;  /* B block for trsm 3&4 */

    /* GEMM output goes to the OTHER block */
    float *Bg2 = B + p.t34_b_off;   /* gemm2 writes to the bwd block */
    float *Bg5 = B + p.t16_b_off;   /* gemm5 writes to the fwd block */

    /* 1. Forward solve: block 11 */
    CURFP_CHECK_CUBLAS(cublasStrsm(cb,
        CUBLAS_SIDE_LEFT, p.t16_fill, p.t1_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_fwd, nrhs, &one,
        A + p.t16_a_off, p.t16_lda,
        Bfwd, ldb));

    /* 2. GEMM: off-diagonal update (forward) → update bwd block */
    CURFP_CHECK_CUBLAS(cublasSgemm(cb,
        p.g2_op, CUBLAS_OP_N,
        p.dim_bwd, nrhs, p.dim_fwd,
        &mone, A + p.g2_a_off, p.g2_lda,
        Bfwd, ldb,
        &one, Bg2, ldb));

    /* 3. Forward solve: block 22 */
    CURFP_CHECK_CUBLAS(cublasStrsm(cb,
        CUBLAS_SIDE_LEFT, p.t34_fill, p.t3_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_bwd, nrhs, &one,
        A + p.t34_a_off, p.t34_lda,
        Bbwd, ldb));

    /* 4. Backward solve: block 22 */
    CURFP_CHECK_CUBLAS(cublasStrsm(cb,
        CUBLAS_SIDE_LEFT, p.t34_fill, p.t4_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_bwd, nrhs, &one,
        A + p.t34_a_off, p.t34_lda,
        Bbwd, ldb));

    /* 5. GEMM: off-diagonal update (backward) → update fwd block */
    CURFP_CHECK_CUBLAS(cublasSgemm(cb,
        p.g5_op, CUBLAS_OP_N,
        p.dim_fwd, nrhs, p.dim_bwd,
        &mone, A + p.g5_a_off, p.g2_lda,
        Bbwd, ldb,
        &one, Bg5, ldb));

    /* 6. Backward solve: block 11 */
    CURFP_CHECK_CUBLAS(cublasStrsm(cb,
        CUBLAS_SIDE_LEFT, p.t16_fill, p.t6_op, CUBLAS_DIAG_NON_UNIT,
        p.dim_fwd, nrhs, &one,
        A + p.t16_a_off, p.t16_lda,
        Bfwd, ldb));

    return CURFP_STATUS_SUCCESS;
}
