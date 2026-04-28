#include <stdlib.h>
#include "curfp_internal.h"

curfpStatus_t curfpCreate(curfpHandle_t *handle)
{
    if (!handle) return CURFP_STATUS_INVALID_VALUE;

    curfpContext *ctx = (curfpContext *)malloc(sizeof(curfpContext));
    if (!ctx) return CURFP_STATUS_ALLOC_FAILED;

    cublasStatus_t cs = cublasCreate(&ctx->cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return from_cublas_status(cs);
    }

    cusolverStatus_t ss = cusolverDnCreate(&ctx->cusolver);
    if (ss != CUSOLVER_STATUS_SUCCESS) {
        cublasDestroy(ctx->cublas);
        free(ctx);
        return from_cusolver_status(ss);
    }

    *handle = ctx;
    return CURFP_STATUS_SUCCESS;
}

curfpStatus_t curfpDestroy(curfpHandle_t handle)
{
    CURFP_CHECK_HANDLE(handle);
    cublasDestroy(handle->cublas);
    cusolverDnDestroy(handle->cusolver);
    free(handle);
    return CURFP_STATUS_SUCCESS;
}

curfpStatus_t curfpSetStream(curfpHandle_t handle, cudaStream_t stream)
{
    CURFP_CHECK_HANDLE(handle);
    CURFP_CHECK_CUBLAS(cublasSetStream(handle->cublas, stream));
    CURFP_CHECK_CUSOLVER(cusolverDnSetStream(handle->cusolver, stream));
    return CURFP_STATUS_SUCCESS;
}

curfpStatus_t curfpGetStream(curfpHandle_t handle, cudaStream_t *stream)
{
    CURFP_CHECK_HANDLE(handle);
    if (!stream) return CURFP_STATUS_INVALID_VALUE;
    CURFP_CHECK_CUBLAS(cublasGetStream(handle->cublas, stream));
    return CURFP_STATUS_SUCCESS;
}
