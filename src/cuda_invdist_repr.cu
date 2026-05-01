// cuda_invdist_repr.cu — GPU batched inverse-distance representation + Jacobian (FP32)
//
// Kernel layout
// -------------
// Grid  : (nm,)  — one block per molecule
// Block : 128 threads — threads stride over M = n_atoms*(n_atoms-1)/2 pairs
//
// Each thread handles one pair (i<j) and writes:
//   Forward:  X[m, p]               = 1 / r_{ij}
//   Jacobian: dX[m, 3*i+d, p]  and  dX[m, 3*j+d, p]   for d in {0,1,2}
//
// No atomics are needed: for different pairs p, writes go to distinct columns
// in dX (layout is (nm, D, M) row-major, so column index is the last dimension).

#include "cuda_invdist_repr.hpp"

#include <cuda_runtime.h>

namespace kf {
namespace invdist_cuda {

namespace {

constexpr int BLOCK_SZ = 128;

// Map flat pair index p (0 <= p < M) to atom indices (ii, jj) with ii < jj.
//
// Pairs are enumerated in row-major order of the strict upper triangle:
//   (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
//
// For row ii, the starting pair index is:
//   start(ii) = ii*(2*n - ii - 1) / 2
// We scan rows until we find the right one.  n_atoms is typically <= 30,
// so the loop is O(n_atoms) and fast in practice.
__device__ static void pair_to_ij(int p, int n_atoms, int &ii, int &jj) {
    int cumul = 0;
    for (int a = 0; a < n_atoms - 1; ++a) {
        int row_len = n_atoms - 1 - a;
        if (cumul + row_len > p) {
            ii = a;
            jj = a + 1 + (p - cumul);
            return;
        }
        cumul += row_len;
    }
    // Unreachable for valid p, but keeps the compiler happy.
    ii = 0;
    jj = 1;
}

// ---------------------------------------------------------------------------
// Forward kernel: X[m, p] = 1 / r_{ij}
// ---------------------------------------------------------------------------
__global__ void invdist_kernel(
    const float *__restrict__ coords,  // (nm, n_atoms, 3)
    int nm,
    int n_atoms,
    int M,   // n_atoms*(n_atoms-1)/2
    float eps2,
    float *__restrict__ X  // (nm, M)
) {
    int m = blockIdx.x;
    if (m >= nm) return;

    const float *c = coords + m * n_atoms * 3;
    float *x = X + m * M;

    for (int p = threadIdx.x; p < M; p += BLOCK_SZ) {
        int ii, jj;
        pair_to_ij(p, n_atoms, ii, jj);

        float dx = c[ii * 3 + 0] - c[jj * 3 + 0];
        float dy = c[ii * 3 + 1] - c[jj * 3 + 1];
        float dz = c[ii * 3 + 2] - c[jj * 3 + 2];
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < eps2) r2 = eps2;
        x[p] = rsqrtf(r2);
    }
}

// ---------------------------------------------------------------------------
// Jacobian kernel: X[m, p] and dX[m, 3*a+d, p]
//
// dX layout: (nm, D, M) row-major, D = 3*n_atoms.
// For pair p=(ii<jj):
//   dX[m, 3*ii + d, p] = -(R_{ii} - R_{jj})_d / r^3   (d in {0,1,2})
//   dX[m, 3*jj + d, p] = +(R_{ii} - R_{jj})_d / r^3
//
// No write conflicts: each thread owns a unique column p.
// dX is pre-zeroed (torch::zeros) so unset entries remain zero.
// ---------------------------------------------------------------------------
__global__ void invdist_jacobian_kernel(
    const float *__restrict__ coords,  // (nm, n_atoms, 3)
    int nm,
    int n_atoms,
    int M,   // n_atoms*(n_atoms-1)/2
    int D,   // 3*n_atoms
    float eps2,
    float *__restrict__ X,   // (nm, M)
    float *__restrict__ dX   // (nm, D, M)
) {
    int m = blockIdx.x;
    if (m >= nm) return;

    const float *c = coords + m * n_atoms * 3;
    float *x = X + m * M;
    // Cast to int64_t to prevent overflow for large nm*D*M products.
    float *jac = dX + (int64_t)m * D * M;

    for (int p = threadIdx.x; p < M; p += BLOCK_SZ) {
        int ii, jj;
        pair_to_ij(p, n_atoms, ii, jj);

        float rx = c[ii * 3 + 0] - c[jj * 3 + 0];
        float ry = c[ii * 3 + 1] - c[jj * 3 + 1];
        float rz = c[ii * 3 + 2] - c[jj * 3 + 2];
        float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 < eps2) r2 = eps2;

        float inv_r = rsqrtf(r2);
        float inv_r3 = inv_r / r2;  // 1/r^3

        x[p] = inv_r;

        float gx = rx * inv_r3;
        float gy = ry * inv_r3;
        float gz = rz * inv_r3;

        // dX[m, 3*ii + d, p] = -grad_d,   dX[m, 3*jj + d, p] = +grad_d
        jac[(ii * 3 + 0) * M + p] = -gx;
        jac[(ii * 3 + 1) * M + p] = -gy;
        jac[(ii * 3 + 2) * M + p] = -gz;

        jac[(jj * 3 + 0) * M + p] = +gx;
        jac[(jj * 3 + 1) * M + p] = +gy;
        jac[(jj * 3 + 2) * M + p] = +gz;
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Host drivers
// ---------------------------------------------------------------------------

torch::Tensor inverse_distance_upper_cuda(
    const torch::Tensor &coords, int n_atoms, float eps
) {
    TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(coords.dtype() == torch::kFloat32, "coords must be float32");
    TORCH_CHECK(
        coords.dim() == 3 && coords.size(1) == n_atoms && coords.size(2) == 3,
        "coords must have shape (nm, n_atoms, 3)"
    );
    TORCH_CHECK(n_atoms >= 2, "n_atoms must be >= 2");

    const int nm = (int)coords.size(0);
    const int M = n_atoms * (n_atoms - 1) / 2;

    auto opts = coords.options();
    torch::Tensor X = torch::zeros({nm, M}, opts);

    invdist_kernel<<<nm, BLOCK_SZ>>>(
        coords.data_ptr<float>(), nm, n_atoms, M, eps * eps, X.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "invdist_kernel launch failed");

    return X;
}

std::tuple<torch::Tensor, torch::Tensor> inverse_distance_upper_and_jacobian_cuda(
    const torch::Tensor &coords, int n_atoms, float eps
) {
    TORCH_CHECK(coords.is_cuda(), "coords must be a CUDA tensor");
    TORCH_CHECK(coords.dtype() == torch::kFloat32, "coords must be float32");
    TORCH_CHECK(
        coords.dim() == 3 && coords.size(1) == n_atoms && coords.size(2) == 3,
        "coords must have shape (nm, n_atoms, 3)"
    );
    TORCH_CHECK(n_atoms >= 2, "n_atoms must be >= 2");

    const int nm = (int)coords.size(0);
    const int M = n_atoms * (n_atoms - 1) / 2;
    const int D = 3 * n_atoms;

    auto opts = coords.options();
    torch::Tensor X = torch::zeros({nm, M}, opts);
    torch::Tensor dX = torch::zeros({nm, D, M}, opts);

    invdist_jacobian_kernel<<<nm, BLOCK_SZ>>>(
        coords.data_ptr<float>(),
        nm,
        n_atoms,
        M,
        D,
        eps * eps,
        X.data_ptr<float>(),
        dX.data_ptr<float>()
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "invdist_jacobian_kernel launch failed");

    return {X, dX};
}

}  // namespace invdist_cuda
}  // namespace kf
