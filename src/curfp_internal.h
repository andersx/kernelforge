#ifndef CURFP_INTERNAL_H
#define CURFP_INTERNAL_H

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "curfp.h"

/* ---------------------------------------------------------------------------
 * Internal handle structure
 * ---------------------------------------------------------------------------*/
struct curfpContext {
    cublasHandle_t     cublas;
    cusolverDnHandle_t cusolver;
};

/* ---------------------------------------------------------------------------
 * Enum converters
 * ---------------------------------------------------------------------------*/
static inline cublasFillMode_t to_cublas_fill(curfpFillMode_t uplo)
{
    return (uplo == CURFP_FILL_MODE_LOWER) ? CUBLAS_FILL_MODE_LOWER
                                           : CUBLAS_FILL_MODE_UPPER;
}

static inline cublasOperation_t to_cublas_op(curfpOperation_t op)
{
    return (op == CURFP_OP_N) ? CUBLAS_OP_N : CUBLAS_OP_T;
}

static inline cublasSideMode_t to_cublas_side(curfpSideMode_t side)
{
    return (side == CURFP_SIDE_LEFT) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
}

/* ---------------------------------------------------------------------------
 * Map cuBLAS status to curfp status
 * ---------------------------------------------------------------------------*/
static inline curfpStatus_t from_cublas_status(cublasStatus_t s)
{
    if (s == CUBLAS_STATUS_SUCCESS)         return CURFP_STATUS_SUCCESS;
    if (s == CUBLAS_STATUS_NOT_INITIALIZED) return CURFP_STATUS_NOT_INITIALIZED;
    if (s == CUBLAS_STATUS_ALLOC_FAILED)    return CURFP_STATUS_ALLOC_FAILED;
    if (s == CUBLAS_STATUS_INVALID_VALUE)   return CURFP_STATUS_INVALID_VALUE;
    return CURFP_STATUS_EXECUTION_FAILED;
}

/* ---------------------------------------------------------------------------
 * Map cuSOLVER status to curfp status
 * ---------------------------------------------------------------------------*/
static inline curfpStatus_t from_cusolver_status(cusolverStatus_t s)
{
    if (s == CUSOLVER_STATUS_SUCCESS)         return CURFP_STATUS_SUCCESS;
    if (s == CUSOLVER_STATUS_NOT_INITIALIZED) return CURFP_STATUS_NOT_INITIALIZED;
    if (s == CUSOLVER_STATUS_ALLOC_FAILED)    return CURFP_STATUS_ALLOC_FAILED;
    if (s == CUSOLVER_STATUS_INVALID_VALUE)   return CURFP_STATUS_INVALID_VALUE;
    return CURFP_STATUS_EXECUTION_FAILED;
}

/* ---------------------------------------------------------------------------
 * Convenience macros for returning on error
 * ---------------------------------------------------------------------------*/
#define CURFP_CHECK_HANDLE(h)                                       \
    do {                                                             \
        if (!(h)) return CURFP_STATUS_NOT_INITIALIZED;              \
    } while (0)

#define CURFP_CHECK_CUBLAS(expr)                                     \
    do {                                                             \
        cublasStatus_t _s = (expr);                                  \
        if (_s != CUBLAS_STATUS_SUCCESS)                             \
            return from_cublas_status(_s);                           \
    } while (0)

#define CURFP_CHECK_CUSOLVER(expr)                                   \
    do {                                                             \
        cusolverStatus_t _s = (expr);                                \
        if (_s != CUSOLVER_STATUS_SUCCESS)                           \
            return from_cusolver_status(_s);                         \
    } while (0)

#define CURFP_CHECK_CUDA(expr)                                       \
    do {                                                             \
        cudaError_t _e = (expr);                                     \
        if (_e != cudaSuccess) return CURFP_STATUS_EXECUTION_FAILED; \
    } while (0)

#endif /* CURFP_INTERNAL_H */
