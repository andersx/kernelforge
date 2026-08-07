// CUDA FCHL18 full combined energy+force kernels.
//
// Assembles EE / EF / FE / FF blocks from the existing scalar, jacobian,
// jacobian_t, and hessian CUDA entry points. Symmetric variants evaluate the
// Hessian lower triangle once and mirror (no 2x FF cost).

#include "cuda_fchl18_kernel.hpp"

#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_fchl18_common.cuh"

namespace kf {
namespace fchl18 {

namespace {

constexpr int kFullScatterBlock = 256;

// Device RFP index — matches kf::rfp_index_upper_N (TRANSR='N', UPLO='U').
__device__ inline long long full_rfp_index_upper_N(int n, int i, int j) {
    const int k = n / 2;
    const int stride = (n % 2 == 0) ? (n + 1) : n;
    if (j >= k) {
        return static_cast<long long>(j - k) * stride + i;
    }
    return static_cast<long long>(i) * stride + (j + k + 1);
}

template <typename T>
__global__ void full_scatter_block_kernel(
    const T *src,
    T *dst,
    int src_rows,
    int src_cols,
    int dst_ld,
    int dst_row0,
    int dst_col0
) {
    const long long n = static_cast<long long>(src_rows) * src_cols;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const int r = static_cast<int>(tid / src_cols);
    const int c = static_cast<int>(tid - static_cast<long long>(r) * src_cols);
    dst[static_cast<long long>(dst_row0 + r) * dst_ld + (dst_col0 + c)] = src[tid];
}

// Mirror FE jacobian into EF: K[j, nm+flat] = K[nm+flat, j] = J[flat, j].
template <typename T>
__global__ void full_scatter_jac_and_mirror_kernel(
    const T *J,
    T *K,
    int D,
    int nm,
    int BIG
) {
    const long long n = static_cast<long long>(D) * nm;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const int flat = static_cast<int>(tid / nm);
    const int j = static_cast<int>(tid - static_cast<long long>(flat) * nm);
    const T v = J[tid];
    K[static_cast<long long>(nm + flat) * BIG + j] = v;
    K[static_cast<long long>(j) * BIG + (nm + flat)] = v;
}

template <typename T>
__global__ void full_pack_scalar_rfp_kernel(
    const T *K_scalar,
    T *rfp,
    int nm,
    int BIG
) {
    const long long n = static_cast<long long>(nm) * (nm + 1) / 2;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    // Decode lower-triangle index tid -> (a, b) with b <= a.
    const double s = sqrt(1.0 + 8.0 * static_cast<double>(tid));
    int a = static_cast<int>((s - 1.0) * 0.5);
    while (a * (a + 1) / 2 > static_cast<int>(tid)) {
        --a;
    }
    while ((a + 1) * (a + 2) / 2 <= static_cast<int>(tid)) {
        ++a;
    }
    const int b = static_cast<int>(tid) - a * (a + 1) / 2;
    rfp[full_rfp_index_upper_N(BIG, b, a)] = K_scalar[a * nm + b];
}

template <typename T>
__global__ void full_pack_jac_rfp_kernel(
    const T *J,
    T *rfp,
    int D,
    int nm,
    int BIG
) {
    const long long n = static_cast<long long>(D) * nm;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const int flat = static_cast<int>(tid / nm);
    const int j = static_cast<int>(tid - static_cast<long long>(flat) * nm);
    const int jac_row = nm + flat;
    // Pack (row=jac_row, col=j) with col <= row into RFP (jac and jac_t share slot).
    if (j <= jac_row) {
        rfp[full_rfp_index_upper_N(BIG, j, jac_row)] = J[tid];
    }
}

template <typename T>
__global__ void full_pack_hess_rfp_kernel(
    const T *H,
    T *rfp,
    int D,
    int nm,
    int BIG
) {
    // Pack upper triangle of FF: BIG indices (nm+row, nm+col) with col <= row.
    const long long n = static_cast<long long>(D) * (D + 1) / 2;
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) {
        return;
    }
    const double s = sqrt(1.0 + 8.0 * static_cast<double>(tid));
    int row = static_cast<int>((s - 1.0) * 0.5);
    while (row * (row + 1) / 2 > static_cast<int>(tid)) {
        --row;
    }
    while ((row + 1) * (row + 2) / 2 <= static_cast<int>(tid)) {
        ++row;
    }
    const int col = static_cast<int>(tid) - row * (row + 1) / 2;
    const int big_row = nm + row;
    const int big_col = nm + col;
    rfp[full_rfp_index_upper_N(BIG, big_col, big_row)] =
        H[static_cast<long long>(row) * D + col];
}

template <typename T>
void launch_scatter(
    const T *src,
    T *dst,
    int src_rows,
    int src_cols,
    int dst_ld,
    int dst_row0,
    int dst_col0
) {
    const long long n = static_cast<long long>(src_rows) * src_cols;
    if (n <= 0) {
        return;
    }
    const int grid = static_cast<int>((n + kFullScatterBlock - 1) / kFullScatterBlock);
    full_scatter_block_kernel<<<grid, kFullScatterBlock>>>(
        src, dst, src_rows, src_cols, dst_ld, dst_row0, dst_col0
    );
    CUDA_CHECK(cudaGetLastError());
}

struct FullParams {
    double sigma;
    double two_body_scaling;
    double two_body_width;
    double two_body_power;
    double three_body_scaling;
    double three_body_width;
    double three_body_power;
    double cut_start;
    double cut_distance;
    int fourier_order;
    bool use_atm;
};

template <typename T>
void kernel_gaussian_full_cu_impl(
    const T *d_x1,
    const T *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const T *d_coords1,
    const int *d_z1,
    const T *d_coords2,
    const int *d_z2,
    T *d_K_out,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    const FullParams &P
) {
    if (nm1 <= 0 || nm2 <= 0 || d_a_rows <= 0 || d_b_cols <= 0) {
        throw std::invalid_argument("kernel_gaussian_full: empty molecule set");
    }

    const int full_rows = nm1 + d_a_rows;
    const int full_cols = nm2 + d_b_cols;
    const size_t full_n = static_cast<size_t>(full_rows) * static_cast<size_t>(full_cols);
    CUDA_CHECK(cudaMemset(d_K_out, 0, full_n * sizeof(T)));

    T *d_EE = nullptr;
    T *d_FE = nullptr;
    T *d_EF = nullptr;
    T *d_FF = nullptr;
    CUDA_CHECK(cudaMalloc(&d_EE, static_cast<size_t>(nm1) * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_FE, static_cast<size_t>(d_a_rows) * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_EF, static_cast<size_t>(nm1) * d_b_cols * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_FF, static_cast<size_t>(d_a_rows) * d_b_cols * sizeof(T)));

    const T sigma = static_cast<T>(P.sigma);
    const T tbs = static_cast<T>(P.two_body_scaling);
    const T tbw = static_cast<T>(P.two_body_width);
    const T tbp = static_cast<T>(P.two_body_power);
    const T ths = static_cast<T>(P.three_body_scaling);
    const T thw = static_cast<T>(P.three_body_width);
    const T thp = static_cast<T>(P.three_body_power);
    const T cs = static_cast<T>(P.cut_start);
    const T cd = static_cast<T>(P.cut_distance);

    kernel_gaussian_rect_cu(
        d_x1, d_x2, d_n1, d_n2, d_nn1, d_nn2, d_EE, sigma, nm1, nm2, max_size1, max_size2, tbs, tbw,
        tbp, ths, thw, thp, cs, cd, P.fourier_order, P.use_atm
    );
    kernel_gaussian_jacobian_cu(
        d_x1, d_x2, d_n1, d_n2, d_nn1, d_nn2, d_coords1, d_z1, d_FE, sigma, nm1, nm2, max_size1,
        max_size2, d_a_rows, tbs, tbw, tbp, ths, thw, thp, cs, cd, P.fourier_order, P.use_atm
    );
    // EF: differentiate B against fixed repr A → jac_t(query=B, fixed=A).
    kernel_gaussian_jacobian_t_cu(
        d_x2, d_x1, d_n2, d_n1, d_nn2, d_nn1, d_coords2, d_z2, d_EF, sigma, nm2, nm1, max_size2,
        max_size1, d_b_cols, tbs, tbw, tbp, ths, thw, thp, cs, cd, P.fourier_order, P.use_atm
    );
    kernel_gaussian_hessian_cu(
        d_x1, d_x2, d_n1, d_n2, d_nn1, d_nn2, d_coords1, d_z1, d_coords2, d_z2, d_FF, sigma, nm1,
        nm2, max_size1, max_size2, d_a_rows, d_b_cols, tbs, tbw, tbp, ths, thw, thp, cs, cd,
        P.fourier_order, P.use_atm
    );

    launch_scatter(d_EE, d_K_out, nm1, nm2, full_cols, 0, 0);
    // EF from jac_t: shape (nm1, d_b_cols) — rows are fixed molecules, cols are B coords.
    launch_scatter(d_EF, d_K_out, nm1, d_b_cols, full_cols, 0, nm2);
    launch_scatter(d_FE, d_K_out, d_a_rows, nm2, full_cols, nm1, 0);
    launch_scatter(d_FF, d_K_out, d_a_rows, d_b_cols, full_cols, nm1, nm2);

    CUDA_CHECK(cudaFree(d_EE));
    CUDA_CHECK(cudaFree(d_FE));
    CUDA_CHECK(cudaFree(d_EF));
    CUDA_CHECK(cudaFree(d_FF));
}

template <typename T>
void kernel_gaussian_full_symm_cu_impl(
    const T *d_x,
    const int *d_n,
    const int *d_nn,
    const T *d_coords,
    const int *d_z,
    T *d_K_out,
    int nm,
    int max_size,
    int D,
    const FullParams &P
) {
    if (nm <= 0 || D <= 0) {
        throw std::invalid_argument("kernel_gaussian_full_symm: empty molecule set");
    }

    const int BIG = nm + D;
    CUDA_CHECK(cudaMemset(d_K_out, 0, static_cast<size_t>(BIG) * BIG * sizeof(T)));

    T *d_EE = nullptr;
    T *d_FE = nullptr;
    T *d_FF = nullptr;
    CUDA_CHECK(cudaMalloc(&d_EE, static_cast<size_t>(nm) * nm * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_FE, static_cast<size_t>(D) * nm * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_FF, static_cast<size_t>(D) * D * sizeof(T)));

    const T sigma = static_cast<T>(P.sigma);
    const T tbs = static_cast<T>(P.two_body_scaling);
    const T tbw = static_cast<T>(P.two_body_width);
    const T tbp = static_cast<T>(P.two_body_power);
    const T ths = static_cast<T>(P.three_body_scaling);
    const T thw = static_cast<T>(P.three_body_width);
    const T thp = static_cast<T>(P.three_body_power);
    const T cs = static_cast<T>(P.cut_start);
    const T cd = static_cast<T>(P.cut_distance);

    kernel_gaussian_symm_cu(
        d_x, d_n, d_nn, d_EE, sigma, nm, max_size, tbs, tbw, tbp, ths, thw, thp, cs, cd,
        P.fourier_order, P.use_atm
    );
    kernel_gaussian_jacobian_cu(
        d_x, d_x, d_n, d_n, d_nn, d_nn, d_coords, d_z, d_FE, sigma, nm, nm, max_size, max_size, D,
        tbs, tbw, tbp, ths, thw, thp, cs, cd, P.fourier_order, P.use_atm
    );
    kernel_gaussian_hessian_symm_cu(
        d_x, d_n, d_nn, d_coords, d_z, d_FF, sigma, nm, max_size, D, tbs, tbw, tbp, ths, thw, thp,
        cs, cd, P.fourier_order, P.use_atm
    );

    launch_scatter(d_EE, d_K_out, nm, nm, BIG, 0, 0);

    {
        const long long n = static_cast<long long>(D) * nm;
        const int grid = static_cast<int>((n + kFullScatterBlock - 1) / kFullScatterBlock);
        full_scatter_jac_and_mirror_kernel<<<grid, kFullScatterBlock>>>(d_FE, d_K_out, D, nm, BIG);
        CUDA_CHECK(cudaGetLastError());
    }

    launch_scatter(d_FF, d_K_out, D, D, BIG, nm, nm);

    CUDA_CHECK(cudaFree(d_EE));
    CUDA_CHECK(cudaFree(d_FE));
    CUDA_CHECK(cudaFree(d_FF));
}

template <typename T>
void kernel_gaussian_full_symm_rfp_cu_impl(
    const T *d_x,
    const int *d_n,
    const int *d_nn,
    const T *d_coords,
    const int *d_z,
    T *d_rfp_out,
    int nm,
    int max_size,
    int D,
    const FullParams &P
) {
    if (nm <= 0 || D <= 0) {
        throw std::invalid_argument("kernel_gaussian_full_symm_rfp: empty molecule set");
    }

    const int BIG = nm + D;
    const size_t rfp_len = static_cast<size_t>(BIG) * (BIG + 1) / 2;
    CUDA_CHECK(cudaMemset(d_rfp_out, 0, rfp_len * sizeof(T)));

    T *d_EE = nullptr;
    T *d_FE = nullptr;
    T *d_FF = nullptr;
    CUDA_CHECK(cudaMalloc(&d_EE, static_cast<size_t>(nm) * nm * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_FE, static_cast<size_t>(D) * nm * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_FF, static_cast<size_t>(D) * D * sizeof(T)));

    const T sigma = static_cast<T>(P.sigma);
    const T tbs = static_cast<T>(P.two_body_scaling);
    const T tbw = static_cast<T>(P.two_body_width);
    const T tbp = static_cast<T>(P.two_body_power);
    const T ths = static_cast<T>(P.three_body_scaling);
    const T thw = static_cast<T>(P.three_body_width);
    const T thp = static_cast<T>(P.three_body_power);
    const T cs = static_cast<T>(P.cut_start);
    const T cd = static_cast<T>(P.cut_distance);

    kernel_gaussian_symm_cu(
        d_x, d_n, d_nn, d_EE, sigma, nm, max_size, tbs, tbw, tbp, ths, thw, thp, cs, cd,
        P.fourier_order, P.use_atm
    );
    kernel_gaussian_jacobian_cu(
        d_x, d_x, d_n, d_n, d_nn, d_nn, d_coords, d_z, d_FE, sigma, nm, nm, max_size, max_size, D,
        tbs, tbw, tbp, ths, thw, thp, cs, cd, P.fourier_order, P.use_atm
    );
    kernel_gaussian_hessian_symm_cu(
        d_x, d_n, d_nn, d_coords, d_z, d_FF, sigma, nm, max_size, D, tbs, tbw, tbp, ths, thw, thp,
        cs, cd, P.fourier_order, P.use_atm
    );

    {
        const long long n = static_cast<long long>(nm) * (nm + 1) / 2;
        const int grid = static_cast<int>((n + kFullScatterBlock - 1) / kFullScatterBlock);
        full_pack_scalar_rfp_kernel<<<grid, kFullScatterBlock>>>(d_EE, d_rfp_out, nm, BIG);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const long long n = static_cast<long long>(D) * nm;
        const int grid = static_cast<int>((n + kFullScatterBlock - 1) / kFullScatterBlock);
        full_pack_jac_rfp_kernel<<<grid, kFullScatterBlock>>>(d_FE, d_rfp_out, D, nm, BIG);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const long long n = static_cast<long long>(D) * (D + 1) / 2;
        const int grid = static_cast<int>((n + kFullScatterBlock - 1) / kFullScatterBlock);
        full_pack_hess_rfp_kernel<<<grid, kFullScatterBlock>>>(d_FF, d_rfp_out, D, nm, BIG);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_EE));
    CUDA_CHECK(cudaFree(d_FE));
    CUDA_CHECK(cudaFree(d_FF));
}

FullParams make_params(
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    return FullParams{
        sigma,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm
    };
}

}  // namespace

void kernel_gaussian_full_cu(
    const float *d_x1,
    const float *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const float *d_coords1,
    const int *d_z1,
    const float *d_coords2,
    const int *d_z2,
    float *d_K_out,
    float sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_full_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_coords2,
        d_z2,
        d_K_out,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
        d_b_cols,
        make_params(
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        )
    );
}

void kernel_gaussian_full_cu(
    const double *d_x1,
    const double *d_x2,
    const int *d_n1,
    const int *d_n2,
    const int *d_nn1,
    const int *d_nn2,
    const double *d_coords1,
    const int *d_z1,
    const double *d_coords2,
    const int *d_z2,
    double *d_K_out,
    double sigma,
    int nm1,
    int nm2,
    int max_size1,
    int max_size2,
    int d_a_rows,
    int d_b_cols,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_full_cu_impl(
        d_x1,
        d_x2,
        d_n1,
        d_n2,
        d_nn1,
        d_nn2,
        d_coords1,
        d_z1,
        d_coords2,
        d_z2,
        d_K_out,
        nm1,
        nm2,
        max_size1,
        max_size2,
        d_a_rows,
        d_b_cols,
        make_params(
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        )
    );
}

void kernel_gaussian_full_symm_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_K_out,
    float sigma,
    int nm,
    int max_size,
    int D,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_full_symm_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_K_out,
        nm,
        max_size,
        D,
        make_params(
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        )
    );
}

void kernel_gaussian_full_symm_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_K_out,
    double sigma,
    int nm,
    int max_size,
    int D,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_full_symm_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_K_out,
        nm,
        max_size,
        D,
        make_params(
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        )
    );
}

void kernel_gaussian_full_symm_rfp_cu(
    const float *d_x,
    const int *d_n,
    const int *d_nn,
    const float *d_coords,
    const int *d_z,
    float *d_rfp_out,
    float sigma,
    int nm,
    int max_size,
    int D,
    float two_body_scaling,
    float two_body_width,
    float two_body_power,
    float three_body_scaling,
    float three_body_width,
    float three_body_power,
    float cut_start,
    float cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_full_symm_rfp_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_rfp_out,
        nm,
        max_size,
        D,
        make_params(
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        )
    );
}

void kernel_gaussian_full_symm_rfp_cu(
    const double *d_x,
    const int *d_n,
    const int *d_nn,
    const double *d_coords,
    const int *d_z,
    double *d_rfp_out,
    double sigma,
    int nm,
    int max_size,
    int D,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    kernel_gaussian_full_symm_rfp_cu_impl(
        d_x,
        d_n,
        d_nn,
        d_coords,
        d_z,
        d_rfp_out,
        nm,
        max_size,
        D,
        make_params(
            sigma,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        )
    );
}

}  // namespace fchl18
}  // namespace kf
