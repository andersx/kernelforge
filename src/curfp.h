#ifndef CURFP_H
#define CURFP_H

#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Status codes
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_STATUS_SUCCESS          = 0,
    CURFP_STATUS_NOT_INITIALIZED  = 1,
    CURFP_STATUS_ALLOC_FAILED     = 2,
    CURFP_STATUS_INVALID_VALUE    = 3,
    CURFP_STATUS_EXECUTION_FAILED = 4
} curfpStatus_t;

/* ---------------------------------------------------------------------------
 * Operation type (controls TRANSR and TRANS parameters)
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_OP_N = 0,  /* Non-transpose */
    CURFP_OP_T = 1   /* Transpose     */
} curfpOperation_t;

/* ---------------------------------------------------------------------------
 * Fill mode (upper or lower triangle)
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_FILL_MODE_LOWER = 0,
    CURFP_FILL_MODE_UPPER = 1
} curfpFillMode_t;

/* ---------------------------------------------------------------------------
 * Side mode (left or right) for curfpSsfmm
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_SIDE_LEFT  = 0,  /* C = alpha * A * B + beta * C */
    CURFP_SIDE_RIGHT = 1   /* C = alpha * B * A + beta * C */
} curfpSideMode_t;

/* ---------------------------------------------------------------------------
 * Norm type for curfpSlansf
 * ---------------------------------------------------------------------------*/
typedef enum {
    CURFP_NORM_MAX = 0,  /* max(|A[i,j]|)             */
    CURFP_NORM_ONE = 1,  /* max column sum = 1-norm   */
    CURFP_NORM_FRO = 2   /* Frobenius norm             */
} curfpNormType_t;

/* ---------------------------------------------------------------------------
 * Opaque handle
 * ---------------------------------------------------------------------------*/
typedef struct curfpContext *curfpHandle_t;

/* ---------------------------------------------------------------------------
 * Handle management
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpCreate(curfpHandle_t *handle);
curfpStatus_t curfpDestroy(curfpHandle_t handle);
curfpStatus_t curfpSetStream(curfpHandle_t handle, cudaStream_t stream);
curfpStatus_t curfpGetStream(curfpHandle_t handle, cudaStream_t *stream);

/* ---------------------------------------------------------------------------
 * curfpSsfrk — Symmetric Rank-K update in RFP format (single precision)
 *
 * Computes:
 *   trans == CURFP_OP_N:  C := alpha * A * A^T + beta * C   (A is N x K)
 *   trans == CURFP_OP_T:  C := alpha * A^T * A + beta * C   (A is K x N)
 *
 * C is an N x N symmetric matrix stored in RFP format (N*(N+1)/2 floats on
 * device).  transr selects the RFP storage variant (normal or transposed).
 * alpha and beta are host pointers; A and C are device pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfrk(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    curfpOperation_t trans,    /* operation on A: CURFP_OP_N or CURFP_OP_T       */
    int              n,        /* order of symmetric matrix C                    */
    int              k,        /* number of columns (trans=N) or rows (trans=T)  */
    const float     *alpha,    /* host pointer                                   */
    const float     *A,        /* device pointer, leading dimension lda          */
    int              lda,
    const float     *beta,     /* host pointer                                   */
    float           *C         /* device pointer, RFP format, N*(N+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSpftrf — Cholesky factorization in RFP format (single precision)
 *
 * Computes:
 *   uplo == CURFP_FILL_MODE_LOWER:  A = L * L^T
 *   uplo == CURFP_FILL_MODE_UPPER:  A = U^T * U
 *
 * A is an N x N symmetric positive definite matrix stored in RFP format
 * (N*(N+1)/2 floats on device).  On exit A is overwritten with the factor.
 * info is a host pointer: 0 = success, >0 = leading minor of order info is
 * not positive definite.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSpftrf(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of matrix A                              */
    float           *A,        /* device pointer, RFP format, N*(N+1)/2 floats  */
    int             *info      /* host pointer: 0 = success                     */
);

/* ---------------------------------------------------------------------------
 * curfpSpftrs — Triangular solve using RFP Cholesky factor (single precision)
 *
 * Solves A * X = B where A is the RFP Cholesky factor from curfpSpftrf.
 *   uplo=FILL_LOWER:  (L * L^T) * X = B
 *   uplo=FILL_UPPER:  (U^T * U) * X = B
 *
 * B is (n × nrhs) column-major with leading dimension ldb, overwritten with X.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSpftrs(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    int              nrhs,
    const float     *A,    /* device pointer, RFP Cholesky factor */
    float           *B,    /* device pointer, (n × nrhs), overwritten with X */
    int              ldb
);

/* ---------------------------------------------------------------------------
 * curfpSstrttf — Copy triangular matrix from full format to RFP (single prec.)
 *
 * Converts the triangle of A specified by uplo into packed RFP format ARF.
 * A is n×n row-major with leading dimension lda >= n.
 * ARF is n*(n+1)/2 floats on device.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSstrttf(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* which triangle of A to read                    */
    int              n,        /* order of matrix A                              */
    const float     *A,        /* device pointer, n×n row-major, lda=n          */
    int              lda,      /* leading dimension of A (>= n)                  */
    float           *arf       /* device pointer, RFP format, n*(n+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSsfmv — Symmetric matrix-vector multiply in RFP format (single prec.)
 *
 * Computes:  y := alpha * A * x + beta * y
 *
 * A is an n×n symmetric matrix in RFP format (n*(n+1)/2 floats on device).
 * x and y are device vectors of length n.
 * alpha and beta are host pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfmv(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of matrix A                              */
    const float     *alpha,    /* host pointer                                   */
    const float     *arf,      /* device pointer, RFP format, N*(N+1)/2 floats  */
    const float     *x,        /* device pointer, vector of length n             */
    int              incx,     /* stride for x (usually 1)                       */
    const float     *beta,     /* host pointer                                   */
    float           *y,        /* device pointer, vector of length n (in/out)    */
    int              incy      /* stride for y (usually 1)                       */
);

/* ---------------------------------------------------------------------------
 * curfpSpftri — Compute inverse of SPD matrix from RFP Cholesky factor (s.p.)
 *
 * Given the RFP Cholesky factor produced by curfpSpftrf, overwrites ARF with
 * A^{-1} in the same RFP storage (same transr/uplo convention).
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSpftri(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of matrix                                */
    float           *arf       /* device pointer, RFP format, N*(N+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSstfttr — Copy triangular matrix from RFP format to full (single prec.)
 *
 * Inverse of curfpSstrttf.  Fills the triangle of A specified by uplo.
 * The other triangle of A is not written (zero-initialize A before calling
 * if you need a full dense matrix).
 * ARF is n*(n+1)/2 floats on device.
 * A is n×n row-major with leading dimension lda >= n.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSstfttr(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* which triangle of A to write                   */
    int              n,        /* order of matrix A                              */
    const float     *arf,      /* device pointer, RFP format, n*(n+1)/2 floats  */
    float           *A,        /* device pointer, n×n row-major, lda=n          */
    int              lda       /* leading dimension of A (>= n)                  */
);

/* ---------------------------------------------------------------------------
 * curfpSlansf — Norm of a symmetric matrix in RFP format (single precision)
 *
 * Computes one of:
 *   CURFP_NORM_MAX: max(|A[i,j]|)
 *   CURFP_NORM_ONE: max column absolute sum (= 1-norm = inf-norm for symmetric)
 *   CURFP_NORM_FRO: Frobenius norm
 *
 * arf is the RFP-packed matrix (NOT a Cholesky factor — use the raw symmetric
 * matrix before calling spftrf).
 * *result is a host pointer output.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSlansf(
    curfpHandle_t    handle,
    curfpNormType_t  norm,     /* norm type: MAX, ONE, or FRO                   */
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of matrix A                              */
    const float     *arf,      /* device pointer, RFP format, n*(n+1)/2 floats  */
    float           *result    /* host pointer output: computed norm             */
);

/* ---------------------------------------------------------------------------
 * curfpSpfcon — Reciprocal condition number estimate from RFP Cholesky factor
 *               (single precision).
 *
 * Estimates rcond = 1 / ( ||A^{-1}||_1 * anorm ) using the Hager–Higham
 * iterative 1-norm estimator (LAPACK SLACN2), applied directly to the RFP
 * Cholesky factor without unpacking.
 *
 * arf must be the Cholesky factor produced by curfpSpftrf (same transr/uplo).
 * anorm must be the 1-norm of the original (pre-factorization) SPD matrix,
 * supplied by the caller.
 * *rcond is a host pointer output: 0.0 if A is singular or anorm == 0.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSpfcon(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant — must match spftrf       */
    curfpFillMode_t  uplo,     /* triangle stored — must match spftrf           */
    int              n,        /* order of matrix                               */
    const float     *arf,      /* device pointer, RFP Cholesky factor           */
    float            anorm,    /* 1-norm of original matrix before factorization */
    float           *rcond     /* host pointer output: reciprocal condition no. */
);

/* ---------------------------------------------------------------------------
 * curfpSsfr — Symmetric Rank-1 update in RFP format (single precision)
 *
 * Computes:  C := alpha * x * x^T + C
 *
 * C is an n×n symmetric matrix stored in RFP format (n*(n+1)/2 floats on
 * device).  x is a device vector of length n.
 * alpha is a host pointer; x and arf are device pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfr(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of symmetric matrix C                    */
    const float     *alpha,    /* host pointer                                   */
    const float     *x,        /* device pointer, vector of length n             */
    int              incx,     /* stride for x (usually 1)                       */
    float           *arf       /* device pointer, RFP format, n*(n+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSsfr2 — Symmetric Rank-2 update in RFP format (single precision)
 *
 * Computes:  C := alpha * x * y^T + alpha * y * x^T + C
 *
 * C is an n×n symmetric matrix stored in RFP format (n*(n+1)/2 floats on
 * device).  x and y are device vectors of length n.
 * alpha is a host pointer; x, y, and arf are device pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfr2(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    int              n,        /* order of symmetric matrix C                    */
    const float     *alpha,    /* host pointer                                   */
    const float     *x,        /* device pointer, vector of length n             */
    int              incx,     /* stride for x (usually 1)                       */
    const float     *y,        /* device pointer, vector of length n             */
    int              incy,     /* stride for y (usually 1)                       */
    float           *arf       /* device pointer, RFP format, n*(n+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSsfr2k — Symmetric Rank-2K update in RFP format (single precision)
 *
 * Computes:
 *   trans == CURFP_OP_N:  C := alpha*A*B^T + alpha*B*A^T + beta*C  (A,B are n×k)
 *   trans == CURFP_OP_T:  C := alpha*A^T*B + alpha*B^T*A + beta*C  (A,B are k×n)
 *
 * C is an n×n symmetric matrix stored in RFP format (n*(n+1)/2 floats on
 * device).  A and B are dense matrices with the same shape and leading dim.
 * alpha and beta are host pointers; A, B, and C are device pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfr2k(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    curfpOperation_t trans,    /* operation on A and B: CURFP_OP_N or CURFP_OP_T */
    int              n,        /* order of symmetric matrix C                    */
    int              k,        /* inner dimension                                */
    const float     *alpha,    /* host pointer                                   */
    const float     *A,        /* device pointer, leading dimension lda          */
    int              lda,
    const float     *B,        /* device pointer, leading dimension ldb          */
    int              ldb,
    const float     *beta,     /* host pointer                                   */
    float           *C         /* device pointer, RFP format, n*(n+1)/2 floats  */
);

/* ---------------------------------------------------------------------------
 * curfpSsfmm — Symmetric matrix-matrix multiply in RFP format (single prec.)
 *
 * Computes:
 *   side == CURFP_SIDE_LEFT:   C := alpha * A * B + beta * C
 *   side == CURFP_SIDE_RIGHT:  C := alpha * B * A + beta * C
 *
 * A is a symmetric matrix in RFP format.  B and C are dense m×n matrices
 * with leading dimensions ldb and ldc respectively.
 * If side == CURFP_SIDE_LEFT,  A is m×m (n*(n+1)/2 = m*(m+1)/2 floats).
 * If side == CURFP_SIDE_RIGHT, A is n×n (n*(n+1)/2 floats).
 * alpha and beta are host pointers; arf, B, and C are device pointers.
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpSsfmm(
    curfpHandle_t    handle,
    curfpOperation_t transr,   /* RFP storage variant: CURFP_OP_N or CURFP_OP_T */
    curfpFillMode_t  uplo,     /* CURFP_FILL_MODE_LOWER or CURFP_FILL_MODE_UPPER */
    curfpSideMode_t  side,     /* CURFP_SIDE_LEFT or CURFP_SIDE_RIGHT            */
    int              m,        /* rows of B and C                                */
    int              n,        /* cols of B and C                                */
    const float     *alpha,    /* host pointer                                   */
    const float     *arf,      /* device pointer, RFP format                    */
    const float     *B,        /* device pointer, m×n column-major, ldb >= m    */
    int              ldb,
    const float     *beta,     /* host pointer                                   */
    float           *C,        /* device pointer, m×n column-major, ldc >= m    */
    int              ldc
);

/* ===========================================================================
 * Double-precision (D-prefix) variants
 * ===========================================================================*/

/* ---------------------------------------------------------------------------
 * curfpDsfrk — Symmetric Rank-K update in RFP format (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDsfrk(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpOperation_t trans,
    int              n,
    int              k,
    const double    *alpha,
    const double    *A,
    int              lda,
    const double    *beta,
    double          *C
);

/* ---------------------------------------------------------------------------
 * curfpDpftrf — Cholesky factorization in RFP format (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDpftrf(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    double          *A,
    int             *info
);

/* ---------------------------------------------------------------------------
 * curfpDpftrs — Triangular solve using RFP Cholesky factor (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDpftrs(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    int              nrhs,
    const double    *A,
    double          *B,
    int              ldb
);

/* ---------------------------------------------------------------------------
 * curfpDstrttf — Copy triangular matrix from full format to RFP (double prec.)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDstrttf(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *A,
    int              lda,
    double          *arf
);

/* ---------------------------------------------------------------------------
 * curfpDsfmv — Symmetric matrix-vector multiply in RFP format (double prec.)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDsfmv(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *alpha,
    const double    *arf,
    const double    *x,
    int              incx,
    const double    *beta,
    double          *y,
    int              incy
);

/* ---------------------------------------------------------------------------
 * curfpDpftri — Compute inverse of SPD matrix from RFP Cholesky factor (d.p.)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDpftri(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    double          *arf
);

/* ---------------------------------------------------------------------------
 * curfpDstfttr — Copy triangular matrix from RFP format to full (double prec.)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDstfttr(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *arf,
    double          *A,
    int              lda
);

/* ---------------------------------------------------------------------------
 * curfpDlansf — Norm of a symmetric matrix in RFP format (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDlansf(
    curfpHandle_t    handle,
    curfpNormType_t  norm,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *arf,
    double          *result
);

/* ---------------------------------------------------------------------------
 * curfpDpfcon — Reciprocal condition number estimate from RFP Cholesky factor
 *               (double precision).
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDpfcon(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *arf,
    double           anorm,
    double          *rcond
);

/* ---------------------------------------------------------------------------
 * curfpDsfr — Symmetric Rank-1 update in RFP format (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDsfr(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *alpha,
    const double    *x,
    int              incx,
    double          *arf
);

/* ---------------------------------------------------------------------------
 * curfpDsfr2 — Symmetric Rank-2 update in RFP format (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDsfr2(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    int              n,
    const double    *alpha,
    const double    *x,
    int              incx,
    const double    *y,
    int              incy,
    double          *arf
);

/* ---------------------------------------------------------------------------
 * curfpDsfr2k — Symmetric Rank-2K update in RFP format (double precision)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDsfr2k(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpOperation_t trans,
    int              n,
    int              k,
    const double    *alpha,
    const double    *A,
    int              lda,
    const double    *B,
    int              ldb,
    const double    *beta,
    double          *C
);

/* ---------------------------------------------------------------------------
 * curfpDsfmm — Symmetric matrix-matrix multiply in RFP format (double prec.)
 * ---------------------------------------------------------------------------*/
curfpStatus_t curfpDsfmm(
    curfpHandle_t    handle,
    curfpOperation_t transr,
    curfpFillMode_t  uplo,
    curfpSideMode_t  side,
    int              m,
    int              n,
    const double    *alpha,
    const double    *arf,
    const double    *B,
    int              ldb,
    const double    *beta,
    double          *C,
    int              ldc
);

#ifdef __cplusplus
}
#endif

#endif /* CURFP_H */
