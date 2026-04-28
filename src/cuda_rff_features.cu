// cuda_rff_features.cu — CUDA Random Fourier Features for global descriptors (FP32)

#include "cuda_rff_features.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "curfp.h"

namespace kf::rff_cuda {

namespace {

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t _err = (call);                                                                  \
        TORCH_CHECK(_err == cudaSuccess, "CUDA error: ", cudaGetErrorString(_err));                \
    } while (0)

#define CUBLAS_CHECK(call)                                                                          \
    do {                                                                                            \
        cublasStatus_t _st = (call);                                                                \
        TORCH_CHECK(_st == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", static_cast<int>(_st));        \
    } while (0)

#define CURFP_CHECK(call)                                                                           \
    do {                                                                                            \
        curfpStatus_t _st = (call);                                                                 \
        TORCH_CHECK(_st == CURFP_STATUS_SUCCESS, "cuRFP error: ", static_cast<int>(_st));          \
    } while (0)

cublasHandle_t s_cublas = nullptr;
curfpHandle_t s_curfp = nullptr;

void ensure_cublas() {
    if (s_cublas == nullptr) {
        CUBLAS_CHECK(cublasCreate(&s_cublas));
    }
}

void ensure_curfp() {
    if (s_curfp == nullptr) {
        CURFP_CHECK(curfpCreate(&s_curfp));
    }
}

void check_cuda_float32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__global__ void add_bias_cos_scale_kernel(float *Z, const float *b, int N, int D, float scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * D;
    if (idx >= total) return;
    int d = (int)(idx % D);
    Z[idx] = cosf(Z[idx] + b[d]) * scale;
}

__global__ void add_bias_sin_neg_scale_kernel(float *P, const float *b, int N, int D, float scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * D;
    if (idx >= total) return;
    int d = (int)(idx % D);
    P[idx] = sinf(P[idx] + b[d]) * scale;
}

__global__ void fill_rfp_diag_kernel(float *arf, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // TRANSR=N, UPLO=L. Same convention as cuda_global_kernels rfp_potrf.
    long long idx;
    if (n % 2 == 0) {
        int k = n / 2;
        if (i < k) {
            idx = 1LL + (long long)i * (n + 1);
        } else {
            int j = i - k;
            idx = (long long)j * (n + 1);
        }
    } else {
        int n1 = n - n / 2;
        if (i < n1) {
            idx = (long long)i * n;
        } else {
            int j = i - n1;
            idx = (long long)n + (long long)j * n + j;
        }
    }
    arf[idx] = value;
}

__global__ void elemental_phase_kernel(
    float *phase,
    const float *X,
    const int *Q,
    const int *N,
    const float *W,
    const float *b,
    int start,
    int nmol,
    int max_atoms,
    int rep_size,
    int nelements,
    int D
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nmol * max_atoms * D;
    if (idx >= total) return;

    int d = (int)(idx % D);
    long long atom_linear = idx / D;
    int a = (int)(atom_linear % max_atoms);
    int m_local = (int)(atom_linear / max_atoms);
    int m = start + m_local;
    int q = Q[m * max_atoms + a];
    if (a >= N[m] || q < 0 || q >= nelements) {
        phase[idx] = 0.0f;
        return;
    }

    const float *x = X + ((long long)m * max_atoms + a) * rep_size;
    const float *w = W + ((long long)q * rep_size * D) + d;
    float s = b[q * D + d];
    for (int r = 0; r < rep_size; ++r) s += x[r] * w[(long long)r * D];
    phase[idx] = s;
}

__global__ void elemental_features_from_phase_kernel(
    float *LZ,
    const float *phase,
    const int *Q,
    const int *N,
    int start,
    int nmol,
    int max_atoms,
    int D,
    float scale
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nmol * D;
    if (idx >= total) return;
    int d = (int)(idx % D);
    int m_local = (int)(idx / D);
    int m = start + m_local;
    float s = 0.0f;
    int n_atoms = N[m];
    for (int a = 0; a < n_atoms; ++a) {
        if (Q[m * max_atoms + a] >= 0) {
            s += cosf(phase[((long long)m_local * max_atoms + a) * D + d]) * scale;
        }
    }
    LZ[idx] = s;
}

__global__ void elemental_gradient_from_phase_kernel(
    float *G,
    const float *phase,
    const float *dX,
    const int *Q,
    const int *N,
    const int *offsets,
    const float *W,
    int start,
    int nmol,
    int max_atoms,
    int rep_size,
    int nelements,
    int D,
    float scale
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_start = offsets[start];
    int chunk_end = offsets[start + nmol];
    int ngrads = chunk_end - chunk_start;
    long long total = (long long)ngrads * D;
    if (idx >= total) return;

    int d = (int)(idx % D);
    int local_g = (int)(idx / D);
    int global_g = chunk_start + local_g;
    int m = start;
    while (m + 1 <= start + nmol && offsets[m + 1] <= global_g) ++m;
    int m_local = m - start;
    int rel = global_g - offsets[m];
    int coord_atom = rel / 3;
    int xyz = rel % 3;
    int n_atoms = N[m];
    if (coord_atom >= n_atoms) {
        G[idx] = 0.0f;
        return;
    }

    float s = 0.0f;
    for (int center = 0; center < n_atoms; ++center) {
        int q = Q[m * max_atoms + center];
        if (q < 0 || q >= nelements) continue;
        float coeff = sinf(phase[((long long)m_local * max_atoms + center) * D + d]) * scale;
        const float *w = W + ((long long)q * rep_size * D) + d;
        const float *dx =
            dX + (((long long)m * max_atoms + center) * rep_size * max_atoms * 3) +
            coord_atom * 3 + xyz;
        for (int r = 0; r < rep_size; ++r) {
            s += coeff * w[(long long)r * D] * dx[(long long)r * max_atoms * 3];
        }
    }
    G[idx] = s;
}

torch::Tensor make_offsets_cuda(const torch::Tensor &N) {
    int nmol = (int)N.size(0);
    std::vector<int> sizes(nmol);
    CUDA_CHECK(cudaMemcpy(sizes.data(), N.data_ptr<int>(), (size_t)nmol * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> offsets(nmol + 1, 0);
    for (int i = 0; i < nmol; ++i) offsets[i + 1] = offsets[i] + 3 * sizes[i];
    auto offsets_cuda = torch::empty({nmol + 1}, N.options());
    CUDA_CHECK(cudaMemcpy(offsets_cuda.data_ptr<int>(), offsets.data(), (size_t)(nmol + 1) * sizeof(int), cudaMemcpyHostToDevice));
    return offsets_cuda;
}

torch::Tensor rff_features_chunk(
    const torch::Tensor &X, const torch::Tensor &W, const torch::Tensor &b, int start, int nc
) {
    ensure_cublas();

    int M = (int)X.size(1);
    int D = (int)b.size(0);
    auto Z = torch::empty({nc, D}, X.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major (nc, M) @ (M, D) -> row-major (nc, D). Interpreted as
    // col-major Z^T(D,nc) = W_rowmajor_as_colmajor(D,M) * X^T(M,nc).
    CUBLAS_CHECK(cublasSgemm(
        s_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D,
        nc,
        M,
        &alpha,
        W.data_ptr<float>(),
        D,
        X.data_ptr<float>() + (long long)start * M,
        M,
        &beta,
        Z.data_ptr<float>(),
        D
    ));

    const float scale = std::sqrt(2.0f / (float)D);
    int threads = 256;
    int blocks = (int)(((long long)nc * D + threads - 1) / threads);
    add_bias_cos_scale_kernel<<<blocks, threads>>>(Z.data_ptr<float>(), b.data_ptr<float>(), nc, D, scale);
    CUDA_CHECK(cudaGetLastError());
    return Z;
}

torch::Tensor rff_gradient_chunk(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc
) {
    ensure_cublas();

    int M = (int)X.size(1);
    int ncoords = (int)dX.size(1);
    int D = (int)b.size(0);
    int ngrads = nc * ncoords;

    auto phase = torch::empty({nc, D}, X.options());
    auto G = torch::empty({ngrads, D}, X.options());

    const float one = 1.0f;
    const float zero = 0.0f;

    // phase = X_chunk @ W + b, row-major (nc, D).
    CUBLAS_CHECK(cublasSgemm(
        s_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D,
        nc,
        M,
        &one,
        W.data_ptr<float>(),
        D,
        X.data_ptr<float>() + (long long)start * M,
        M,
        &zero,
        phase.data_ptr<float>(),
        D
    ));

    const float scale = -std::sqrt(2.0f / (float)D);
    int threads = 256;
    int blocks = (int)(((long long)nc * D + threads - 1) / threads);
    add_bias_sin_neg_scale_kernel<<<blocks, threads>>>(
        phase.data_ptr<float>(), b.data_ptr<float>(), nc, D, scale
    );
    CUDA_CHECK(cudaGetLastError());

    // G = dX_chunk @ W, where dX chunk is row-major (ngrads, M).
    CUBLAS_CHECK(cublasSgemm(
        s_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D,
        ngrads,
        M,
        &one,
        W.data_ptr<float>(),
        D,
        dX.data_ptr<float>() + (long long)start * ncoords * M,
        M,
        &zero,
        G.data_ptr<float>(),
        D
    ));

    auto phase_view = phase.view({nc, 1, D}).expand({nc, ncoords, D}).reshape({ngrads, D});
    G.mul_(phase_view);
    return G;
}

torch::Tensor rff_features_elemental_chunk(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc
) {
    int max_atoms = (int)X.size(1);
    int rep_size = (int)X.size(2);
    int nelements = (int)W.size(0);
    int D = (int)b.size(1);
    auto phase = torch::empty({nc, max_atoms, D}, X.options());
    auto LZ = torch::empty({nc, D}, X.options());
    int threads = 256;
    int phase_blocks = (int)(((long long)nc * max_atoms * D + threads - 1) / threads);
    elemental_phase_kernel<<<phase_blocks, threads>>>(
        phase.data_ptr<float>(),
        X.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        start,
        nc,
        max_atoms,
        rep_size,
        nelements,
        D
    );
    CUDA_CHECK(cudaGetLastError());
    const float scale = std::sqrt(2.0f / (float)D);
    int lz_blocks = (int)(((long long)nc * D + threads - 1) / threads);
    elemental_features_from_phase_kernel<<<lz_blocks, threads>>>(
        LZ.data_ptr<float>(),
        phase.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        start,
        nc,
        max_atoms,
        D,
        scale
    );
    CUDA_CHECK(cudaGetLastError());
    return LZ;
}

torch::Tensor rff_gradient_elemental_chunk(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc
) {
    int max_atoms = (int)X.size(1);
    int rep_size = (int)X.size(2);
    int nelements = (int)W.size(0);
    int D = (int)b.size(1);
    auto offsets = make_offsets_cuda(N);
    int chunk_start = 0;
    int chunk_end = 0;
    CUDA_CHECK(cudaMemcpy(&chunk_start, offsets.data_ptr<int>() + start, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&chunk_end, offsets.data_ptr<int>() + start + nc, sizeof(int), cudaMemcpyDeviceToHost));
    int ngrads = chunk_end - chunk_start;
    auto phase = torch::empty({nc, max_atoms, D}, X.options());
    auto G = torch::empty({ngrads, D}, X.options());
    int threads = 256;
    int phase_blocks = (int)(((long long)nc * max_atoms * D + threads - 1) / threads);
    elemental_phase_kernel<<<phase_blocks, threads>>>(
        phase.data_ptr<float>(),
        X.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        start,
        nc,
        max_atoms,
        rep_size,
        nelements,
        D
    );
    CUDA_CHECK(cudaGetLastError());
    const float scale = std::sqrt(2.0f / (float)D);
    int g_blocks = (int)(((long long)ngrads * D + threads - 1) / threads);
    elemental_gradient_from_phase_kernel<<<g_blocks, threads>>>(
        G.data_ptr<float>(),
        phase.data_ptr<float>(),
        dX.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        offsets.data_ptr<int>(),
        W.data_ptr<float>(),
        start,
        nc,
        max_atoms,
        rep_size,
        nelements,
        D,
        scale
    );
    CUDA_CHECK(cudaGetLastError());
    return G;
}

}  // namespace

torch::Tensor rff_features_cuda(
    const torch::Tensor &X, const torch::Tensor &W, const torch::Tensor &b
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0 && b.size(0) > 0, "zero dimension");

    return rff_features_chunk(X, W, b, 0, (int)X.size(0));
}

std::tuple<torch::Tensor, torch::Tensor> rff_gramian_symm_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(Y.dim() == 1, "Y must be 1D (N,)");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(Y.size(0) == X.size(0), "Y.shape[0] must equal X.shape[0]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();
    ensure_curfp();

    int N = (int)X.size(0);
    int D = (int)b.size(0);
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());

    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < N; cs += chunk_size) {
        int nc = std::min(chunk_size, N - cs);
        auto Z = rff_features_chunk(X, W, b, cs, nc);

        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }

        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));

        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));

        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
    std::fprintf(
        stderr,
        "  [rff_gramian_symm_rfp]        chunks=%d  chunk_size=%d  D=%d  total=%.1f ms\n",
        (N + chunk_size - 1) / chunk_size,
        chunk_size,
        D,
        total_ms
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    return {ZtZ_rfp, ZtY};
}

std::tuple<torch::Tensor, torch::Tensor> rff_full_gramian_symm_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    const torch::Tensor &F,
    int energy_chunk,
    int force_chunk
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    check_cuda_float32(F, "F");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(dX.dim() == 3, "dX must be 3D (N, ncoords, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(Y.dim() == 1, "Y must be 1D (N,)");
    TORCH_CHECK(F.dim() == 1, "F must be 1D (N*ncoords,)");
    TORCH_CHECK(dX.size(0) == X.size(0), "dX.shape[0] must equal X.shape[0]");
    TORCH_CHECK(dX.size(2) == X.size(1), "dX.shape[2] must equal X.shape[1]");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(Y.size(0) == X.size(0), "Y.shape[0] must equal X.shape[0]");
    TORCH_CHECK(F.size(0) == X.size(0) * dX.size(1), "F.shape[0] must equal N*ncoords");
    TORCH_CHECK(energy_chunk > 0 && force_chunk > 0, "chunk sizes must be positive");

    ensure_cublas();
    ensure_curfp();

    int N = (int)X.size(0);
    int D = (int)b.size(0);
    int ncoords = (int)dX.size(1);
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());

    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_energy, ev_force;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_energy));
    CUDA_CHECK(cudaEventCreate(&ev_force));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < N; cs += energy_chunk) {
        int nc = std::min(energy_chunk, N - cs);
        auto Z = rff_features_chunk(X, W, b, cs, nc);

        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }

        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));

        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));

        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_energy));

    for (int cs = 0; cs < N; cs += force_chunk) {
        int nc = std::min(force_chunk, N - cs);
        int ngrads = nc * ncoords;
        auto G = rff_gradient_chunk(X, dX, W, b, cs, nc);

        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            &one,
            ZtZ_rfp.data_ptr<float>()
        ));

        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            F.data_ptr<float>() + (long long)cs * ncoords,
            1,
            &one,
            ZtY.data_ptr<float>(),
            1
        ));
    }

    CUDA_CHECK(cudaEventRecord(ev_force));
    CUDA_CHECK(cudaEventSynchronize(ev_force));
    float energy_ms = 0.0f;
    float force_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&energy_ms, ev_start, ev_energy));
    CUDA_CHECK(cudaEventElapsedTime(&force_ms, ev_energy, ev_force));
    std::fprintf(
        stderr,
        "  [rff_full_gramian_symm_rfp]   energy=%.1f ms  force=%.1f ms  total=%.1f ms\n",
        energy_ms,
        force_ms,
        energy_ms + force_ms
    );
    std::fprintf(
        stderr,
        "  [rff_full_gramian_symm_rfp]   N=%d  ncoords=%d  M=%d  D=%d  "
        "energy_chunk=%d  force_chunk=%d\n",
        N,
        ncoords,
        (int)X.size(1),
        D,
        energy_chunk,
        force_chunk
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_energy));
    CUDA_CHECK(cudaEventDestroy(ev_force));

    return {ZtZ_rfp, ZtY};
}

torch::Tensor rff_predict_energy_cuda(
    const torch::Tensor &X,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D (D,)");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(weights.size(0) == b.size(0), "weights.shape[0] must equal b.shape[0]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();

    int N = (int)X.size(0);
    int D = (int)b.size(0);
    auto E = torch::empty({N}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;

    for (int cs = 0; cs < N; cs += chunk_size) {
        int nc = std::min(chunk_size, N - cs);
        auto Z = rff_features_chunk(X, W, b, cs, nc);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            E.data_ptr<float>() + cs,
            1
        ));
    }

    return E;
}

torch::Tensor rff_predict_force_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(dX.dim() == 3, "dX must be 3D (N, ncoords, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D (D,)");
    TORCH_CHECK(dX.size(0) == X.size(0), "dX.shape[0] must equal X.shape[0]");
    TORCH_CHECK(dX.size(2) == X.size(1), "dX.shape[2] must equal X.shape[1]");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(weights.size(0) == b.size(0), "weights.shape[0] must equal b.shape[0]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();

    int N = (int)X.size(0);
    int ncoords = (int)dX.size(1);
    int D = (int)b.size(0);
    auto F = torch::empty({N * ncoords}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;

    for (int cs = 0; cs < N; cs += chunk_size) {
        int nc = std::min(chunk_size, N - cs);
        int ngrads = nc * ncoords;
        auto G = rff_gradient_chunk(X, dX, W, b, cs, nc);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            F.data_ptr<float>() + (long long)cs * ncoords,
            1
        ));
    }

    return F;
}

torch::Tensor rff_features_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 3, "X must be 3D (nmol, max_atoms, rep_size)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D (nmol, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1D (nmol,)");
    TORCH_CHECK(W.dim() == 3, "W must be 3D (nelements, rep_size, D)");
    TORCH_CHECK(b.dim() == 2, "b must be 2D (nelements, D)");
    TORCH_CHECK(Q.size(0) == X.size(0) && Q.size(1) == X.size(1), "Q shape mismatch");
    TORCH_CHECK(N.size(0) == X.size(0), "N.shape[0] must equal X.shape[0]");
    TORCH_CHECK(W.size(1) == X.size(2), "W.shape[1] must equal X.shape[2]");
    TORCH_CHECK(W.size(0) == b.size(0) && W.size(2) == b.size(1), "W/b shape mismatch");

    return rff_features_elemental_chunk(X, Q, N, W, b, 0, (int)X.size(0));
}

std::tuple<torch::Tensor, torch::Tensor> rff_gramian_elemental_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    TORCH_CHECK(Y.dim() == 1 && Y.size(0) == X.size(0), "Y must be 1D (nmol,)");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    ensure_cublas();
    ensure_curfp();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());
    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc = std::min(chunk_size, nmol - cs);
        auto LZ = rff_features_elemental_chunk(X, Q, N, W, b, cs, nc);
        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }
        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));
        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
    std::fprintf(
        stderr,
        "  [rff_gramian_elemental_rfp]   chunks=%d  chunk_size=%d  D=%d  total=%.1f ms\n",
        (nmol + chunk_size - 1) / chunk_size,
        chunk_size,
        D,
        total_ms
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    return {ZtZ_rfp, ZtY};
}

torch::Tensor rff_predict_energy_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(weights.dim() == 1 && weights.size(0) == b.size(1), "weights must be 1D (D,)");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    ensure_cublas();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    auto E = torch::empty({nmol}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;
    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc = std::min(chunk_size, nmol - cs);
        auto LZ = rff_features_elemental_chunk(X, Q, N, W, b, cs, nc);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            E.data_ptr<float>() + cs,
            1
        ));
    }
    return E;
}

std::tuple<torch::Tensor, torch::Tensor> rff_full_gramian_elemental_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    const torch::Tensor &F,
    int energy_chunk,
    int force_chunk
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    check_cuda_float32(F, "F");
    TORCH_CHECK(dX.dim() == 5, "dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    TORCH_CHECK(Y.dim() == 1 && Y.size(0) == X.size(0), "Y must be 1D (nmol,)");
    TORCH_CHECK(F.dim() == 1, "F must be 1D");
    TORCH_CHECK(energy_chunk > 0 && force_chunk > 0, "chunk sizes must be positive");
    ensure_cublas();
    ensure_curfp();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    auto offsets = make_offsets_cuda(N);
    auto offsets_cpu = offsets.cpu();
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());
    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_energy, ev_force;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_energy));
    CUDA_CHECK(cudaEventCreate(&ev_force));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < nmol; cs += energy_chunk) {
        int nc = std::min(energy_chunk, nmol - cs);
        auto LZ = rff_features_elemental_chunk(X, Q, N, W, b, cs, nc);
        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }
        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));
        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_energy));

    for (int cs = 0; cs < nmol; cs += force_chunk) {
        int nc = std::min(force_chunk, nmol - cs);
        auto G = rff_gradient_elemental_chunk(X, dX, Q, N, W, b, cs, nc);
        int ngrads = (int)G.size(0);
        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            &one,
            ZtZ_rfp.data_ptr<float>()
        ));
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            F.data_ptr<float>() + offsets_cpu.data_ptr<int>()[cs],
            1,
            &one,
            ZtY.data_ptr<float>(),
            1
        ));
    }

    CUDA_CHECK(cudaEventRecord(ev_force));
    CUDA_CHECK(cudaEventSynchronize(ev_force));
    float energy_ms = 0.0f;
    float force_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&energy_ms, ev_start, ev_energy));
    CUDA_CHECK(cudaEventElapsedTime(&force_ms, ev_energy, ev_force));
    std::fprintf(
        stderr,
        "  [rff_full_gramian_elemental_rfp] energy=%.1f ms  force=%.1f ms  total=%.1f ms\n",
        energy_ms,
        force_ms,
        energy_ms + force_ms
    );
    std::fprintf(
        stderr,
        "  [rff_full_gramian_elemental_rfp] nmol=%d  max_atoms=%d  rep=%d  D=%d  "
        "energy_chunk=%d  force_chunk=%d\n",
        nmol,
        (int)X.size(1),
        (int)X.size(2),
        D,
        energy_chunk,
        force_chunk
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_energy));
    CUDA_CHECK(cudaEventDestroy(ev_force));
    return {ZtZ_rfp, ZtY};
}

torch::Tensor rff_predict_force_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    ensure_cublas();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    auto offsets = make_offsets_cuda(N);
    auto offsets_cpu = offsets.cpu();
    int total_grads = offsets_cpu.data_ptr<int>()[nmol];
    auto F = torch::empty({total_grads}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;
    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc = std::min(chunk_size, nmol - cs);
        auto G = rff_gradient_elemental_chunk(X, dX, Q, N, W, b, cs, nc);
        int ngrads = (int)G.size(0);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            F.data_ptr<float>() + offsets_cpu.data_ptr<int>()[cs],
            1
        ));
    }
    return F;
}

}  // namespace kf::rff_cuda
