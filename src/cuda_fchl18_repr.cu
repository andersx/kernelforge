// CUDA FCHL18 representation generation.
//
// One block per molecule; threads cover centre atoms. Each centre atom collects
// neighbours within cut_distance, insertion-sorts by distance (stable for ties
// by atom index), and writes the packed (r, Z, dx, dy, dz) channels.

#include "cuda_fchl18_repr.hpp"

#include <cmath>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_fchl18_common.cuh"

namespace kf {
namespace fchl18 {

namespace {

template <typename T>
struct ReprPad;

template <>
struct ReprPad<float> {
    static __device__ __host__ inline float value() { return 1.0e30f; }
};

template <>
struct ReprPad<double> {
    static __device__ __host__ inline double value() { return 1.0e100; }
};

constexpr int kReprMaxSize = 128;

template <typename T>
__global__ void generate_fchl18_kernel(
    const T *__restrict__ coords,  // (nm, max_size, 3)
    const int *__restrict__ z,     // (nm, max_size)
    const int *__restrict__ n,     // (nm,)
    int nm,
    int max_size,
    T cut_distance,
    T *__restrict__ x_out,  // (nm, max_size, 5, max_size)
    int *__restrict__ nn_out  // (nm, max_size)
) {
    const int mol = static_cast<int>(blockIdx.x);
    if (mol >= nm) {
        return;
    }

    const int n_atoms = n[mol];
    const T *c = coords + static_cast<long long>(mol) * max_size * 3;
    const int *zz = z + static_cast<long long>(mol) * max_size;
    T *x_mol = x_out + static_cast<long long>(mol) * max_size * 5 * max_size;
    int *nn_mol = nn_out + static_cast<long long>(mol) * max_size;

    const int stride_atom = 5 * max_size;
    const int stride_chan = max_size;
    const T pad = ReprPad<T>::value();

    // Zero / pad the whole molecule first (cooperative).
    const long long mol_elems = static_cast<long long>(max_size) * 5 * max_size;
    for (long long t = threadIdx.x; t < mol_elems; t += blockDim.x) {
        x_mol[t] = pad;
    }
    for (int t = static_cast<int>(threadIdx.x); t < max_size; t += static_cast<int>(blockDim.x)) {
        nn_mol[t] = 0;
    }
    __syncthreads();

    for (int i = static_cast<int>(threadIdx.x); i < n_atoms; i += static_cast<int>(blockDim.x)) {
        const T xi = c[i * 3 + 0];
        const T yi = c[i * 3 + 1];
        const T zi = c[i * 3 + 2];

        // Collect (r, j) then insertion-sort (matches CPU std::sort by r, then j).
        T rs[kReprMaxSize];
        int js[kReprMaxSize];
        int nn = 0;

        for (int j = 0; j < n_atoms; ++j) {
            const T dx = c[j * 3 + 0] - xi;
            const T dy = c[j * 3 + 1] - yi;
            const T dz = c[j * 3 + 2] - zi;
            const T r = Math<T>::sqrt_(dx * dx + dy * dy + dz * dz);
            if (r < cut_distance) {
                int pos = nn;
                while (pos > 0 && (rs[pos - 1] > r || (rs[pos - 1] == r && js[pos - 1] > j))) {
                    rs[pos] = rs[pos - 1];
                    js[pos] = js[pos - 1];
                    --pos;
                }
                rs[pos] = r;
                js[pos] = j;
                ++nn;
            }
        }

        const int nn_use = nn < max_size ? nn : max_size;
        nn_mol[i] = nn_use;

        T *atom_base = x_mol + static_cast<long long>(i) * stride_atom;
        for (int k = 0; k < nn_use; ++k) {
            const int j = js[k];
            const T dx = c[j * 3 + 0] - xi;
            const T dy = c[j * 3 + 1] - yi;
            const T dz = c[j * 3 + 2] - zi;
            atom_base[0 * stride_chan + k] = rs[k];
            atom_base[1 * stride_chan + k] = static_cast<T>(zz[j]);
            atom_base[2 * stride_chan + k] = dx;
            atom_base[3 * stride_chan + k] = dy;
            atom_base[4 * stride_chan + k] = dz;
        }
    }
}

template <typename T>
std::tuple<torch::Tensor, torch::Tensor> generate_fchl18_cuda_impl(
    const torch::Tensor &coords,
    const torch::Tensor &z,
    const torch::Tensor &n,
    double cut_distance
) {
    const int nm = static_cast<int>(coords.size(0));
    const int max_size = static_cast<int>(coords.size(1));
    if (nm <= 0) {
        throw std::invalid_argument("generate: empty molecule set");
    }
    if (max_size <= 0 || max_size > kReprMaxSize) {
        throw std::invalid_argument("generate: max_size must be in (0, 128]");
    }
    if (cut_distance <= 0.0) {
        throw std::invalid_argument("generate: cut_distance must be > 0");
    }

    auto x = torch::empty(
        {nm, max_size, 5, max_size},
        coords.options()
    );
    auto nn = torch::zeros({nm, max_size}, z.options().dtype(torch::kInt32));

    const int block = max_size < 128 ? 128 : max_size;
    generate_fchl18_kernel<T><<<nm, block>>>(
        coords.data_ptr<T>(),
        z.data_ptr<int>(),
        n.data_ptr<int>(),
        nm,
        max_size,
        static_cast<T>(cut_distance),
        x.data_ptr<T>(),
        nn.data_ptr<int>()
    );
    CUDA_CHECK(cudaGetLastError());
    return {x, nn};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> generate_fchl18_cuda(
    const torch::Tensor &coords,
    const torch::Tensor &z,
    const torch::Tensor &n,
    double cut_distance
) {
    TORCH_CHECK(coords.is_cuda() && coords.is_contiguous(), "coords must be contiguous CUDA");
    TORCH_CHECK(z.is_cuda() && z.is_contiguous(), "z must be contiguous CUDA");
    TORCH_CHECK(n.is_cuda() && n.is_contiguous(), "n must be contiguous CUDA");
    TORCH_CHECK(
        coords.scalar_type() == torch::kFloat32 || coords.scalar_type() == torch::kFloat64,
        "coords must be float32 or float64"
    );
    TORCH_CHECK(z.scalar_type() == torch::kInt32, "z must be int32");
    TORCH_CHECK(n.scalar_type() == torch::kInt32, "n must be int32");
    TORCH_CHECK(coords.dim() == 3 && coords.size(2) == 3, "coords must be (nm, max_size, 3)");
    TORCH_CHECK(z.dim() == 2, "z must be (nm, max_size)");
    TORCH_CHECK(n.dim() == 1, "n must be (nm,)");
    TORCH_CHECK(z.size(0) == coords.size(0) && z.size(1) == coords.size(1), "z shape mismatch");
    TORCH_CHECK(n.size(0) == coords.size(0), "n size mismatch");

    if (coords.scalar_type() == torch::kFloat32) {
        return generate_fchl18_cuda_impl<float>(coords, z, n, cut_distance);
    }
    return generate_fchl18_cuda_impl<double>(coords, z, n, cut_distance);
}

}  // namespace fchl18
}  // namespace kf
