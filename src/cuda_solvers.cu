// cuda_solvers.cu — GPU least-squares solvers via cuSOLVER/cuBLAS (FP32).
//
// cuda_solve_svd:  min-norm least-squares via truncated SVD.
// cuda_solve_qr:   least-squares via QR (cusolverDnSgeqrf + Sormqr + Strsv).
// cuda_solve_gels: least-squares via IRS (cusolverDnSSgels, FP32 working precision).
//                  NOTE: the `iter` argument to cusolverDnSSgels is a HOST pointer;
//                  `d_info` is a DEVICE pointer.  Passing a device pointer for `iter`
//                  causes a NULL-pointer dereference inside IRSInfosGetNiters.
// All accept Z (m x n) and y (m,) as row-major float32 tensors (CPU or GPU).
// All require m >= n (overdetermined system).
// Return w (n,) float32 CPU tensor.
//
// SVD algorithm (overdetermined, m >= n):
//   Z is (m x n) row-major in C/Python.
//   1. Transpose Z to get Zt (n x m) row-major == (m x n) col-major → this is what cuSOLVER sees.
//      cuSOLVER requires m >= n; for overdetermined systems m > n is guaranteed.
//   2. cusolverDnSgesvd on Zt (m x n col-major):
//        jobu='S'  → U  (m x k) col-major, k = min(m,n) = n
//        jobvt='S' → Vt (k x n) col-major
//      Gives: Z = U S Vt  (in col-major / Fortran sense)
//   3. Apply rcond threshold → inv_S (k,)
//   4. tmp = U^T y        (k,)   cublasSgemv op=T on (m x k) col-major
//   5. tmp *= inv_S       (k,)   element-wise
//   6. w   = Vt^T tmp     (n,)   cublasSgemv op=T on (k x n) col-major
//
// QR algorithm (cusolverDnSgeqrf + cusolverDnSormqr + cublasStrsv):
//   1. Transpose Z → QR col-major (m x n) via cublasSgeam.
//   2. cusolverDnSgeqrf: in-place QR factorisation; R stored in upper triangle of QR.
//   3. cusolverDnSormqr: rhs ← Q^T y  (apply Householder reflectors to y in-place).
//   4. cublasStrsv: solve R x = rhs[0:n]  (upper triangular, in-place on rhs).
//   5. Return rhs[0:n] as CPU tensor.
//
// Memory: Zt buffer (m*n floats), U (m*k), Vt (k*n), gesvd workspace.

#include "cuda_solvers.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <string>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t _e = (call);                                                              \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e));               \
    } while (0)

#define CUBLAS_CHECK(call)                                                                    \
    do {                                                                                      \
        cublasStatus_t _s = (call);                                                           \
        TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", static_cast<int>(_s));     \
    } while (0)

#define CUSOLVER_CHECK(call)                                                                  \
    do {                                                                                      \
        cusolverStatus_t _s = (call);                                                         \
        TORCH_CHECK(                                                                          \
            _s == CUSOLVER_STATUS_SUCCESS, "cuSOLVER error: ", static_cast<int>(_s)           \
        );                                                                                    \
    } while (0)

namespace kf {
namespace solvers {

namespace {

cusolverDnHandle_t s_cusolver = nullptr;
cublasHandle_t s_cublas = nullptr;

void ensure_handles() {
    if (s_cusolver == nullptr) {
        CUSOLVER_CHECK(cusolverDnCreate(&s_cusolver));
    }
    if (s_cublas == nullptr) {
        CUBLAS_CHECK(cublasCreate(&s_cublas));
    }
}

// Kernel: inv_S[i] = (S[i] >= threshold) ? 1/S[i] : 0
__global__ void apply_rcond_kernel(
    const float* __restrict__ S, float* __restrict__ inv_S, int k, float threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    float s = S[i];
    inv_S[i] = (s >= threshold) ? (1.0f / s) : 0.0f;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// cuda_solve_svd
// ---------------------------------------------------------------------------
at::Tensor cuda_solve_svd(at::Tensor Z, at::Tensor y, double rcond, bool z_col_major) {
    // When z_col_major=true, Z is (D, m) col-major (i.e. the system has m rows, D cols).
    // When z_col_major=false, Z is the usual (m, n) row-major.
    const int m = z_col_major ? static_cast<int>(Z.size(1)) : static_cast<int>(Z.size(0));
    const int n = z_col_major ? static_cast<int>(Z.size(0)) : static_cast<int>(Z.size(1));
    TORCH_CHECK(Z.dim() == 2, "Z must be 2D");
    TORCH_CHECK(y.dim() == 1, "y must be 1D (m,)");
    TORCH_CHECK(y.size(0) == m, "Z rows must match y length");
    TORCH_CHECK(m >= n, "cuda_solve_svd requires m >= n (overdetermined system)");
    TORCH_CHECK(Z.scalar_type() == at::kFloat, "Z must be float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "y must be float32");

    ensure_handles();

    at::Tensor Z_gpu = Z.to(at::kCUDA).contiguous();
    at::Tensor y_gpu = y.to(at::kCUDA).contiguous();

    const int k = n;  // k = min(m,n) = n for overdetermined

    // If Z is already col-major (D, m) we can use it directly.
    // Otherwise transpose Z (row-major m x n) -> Zt col-major (m x n) via cublasSgeam.
    at::Tensor Zt;
    if (z_col_major) {
        Zt = Z_gpu;  // already (n x m) stored col-major = (m x n) Fortran-order, lda=m
    } else {
        Zt = at::empty({m, n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgeam(
            s_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n,
            &one,
            Z_gpu.data_ptr<float>(), n,
            &zero,
            Zt.data_ptr<float>(), m,
            Zt.data_ptr<float>(), m
        ));
    }

    // SVD: Zt (m x n col-major) = U S Vt
    //   U:  (m x k) col-major, ldu=m
    //   Vt: (k x n) col-major, ldvt=k
    at::Tensor U_t  = at::empty({m, k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    at::Tensor S_t  = at::empty({k},    at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    at::Tensor Vt_t = at::empty({k, n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    at::Tensor d_info_t =
        at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(s_cusolver, m, n, &lwork));
    at::Tensor ws =
        at::empty({lwork + 1}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    CUSOLVER_CHECK(cusolverDnSgesvd(
        s_cusolver,
        'S', 'S',
        m, n,
        Zt.data_ptr<float>(), m,          // lda=m
        S_t.data_ptr<float>(),
        U_t.data_ptr<float>(), m,          // ldu=m
        Vt_t.data_ptr<float>(), k,         // ldvt=k
        ws.data_ptr<float>(), lwork,
        nullptr,                           // rwork (null for real)
        d_info_t.data_ptr<int>()
    ));
    {
        int h = 0;
        CUDA_CHECK(
            cudaMemcpy(&h, d_info_t.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost)
        );
        TORCH_CHECK(h == 0, "cusolverDnSgesvd failed, info=", h);
    }

    // Apply rcond threshold → inv_S
    float s0 = 0.0f;
    CUDA_CHECK(cudaMemcpy(&s0, S_t.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost));
    float threshold;
    if (rcond <= 0.0) {
        threshold = std::numeric_limits<float>::epsilon()
                    * static_cast<float>(std::max(m, n)) * s0;
    } else {
        threshold = static_cast<float>(rcond) * s0;
    }
    at::Tensor inv_S_t =
        at::empty({k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    {
        int threads = 256;
        int blocks  = (k + threads - 1) / threads;
        apply_rcond_kernel<<<blocks, threads>>>(
            S_t.data_ptr<float>(), inv_S_t.data_ptr<float>(), k, threshold
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Back-solve:
    //   tmp = U^T y      U is (m x k) col-major, op=T -> (k x m) * (m,) -> (k,)
    at::Tensor tmp_t = at::empty({k}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    {
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgemv(
            s_cublas, CUBLAS_OP_T,
            m, k,                              // rows, cols of U
            &one,
            U_t.data_ptr<float>(), m,          // U: (m x k) col-major, lda=m
            y_gpu.data_ptr<float>(), 1,
            &zero,
            tmp_t.data_ptr<float>(), 1
        ));
    }

    // tmp *= inv_S  (element-wise, in-place via dgmm)
    CUBLAS_CHECK(cublasSdgmm(
        s_cublas, CUBLAS_SIDE_LEFT,
        k, 1,
        tmp_t.data_ptr<float>(), k,
        inv_S_t.data_ptr<float>(), 1,
        tmp_t.data_ptr<float>(), k
    ));

    // w = Vt^T tmp    Vt is (k x n) col-major, op=T -> (n x k) * (k,) -> (n,)
    at::Tensor w_t = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    {
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgemv(
            s_cublas, CUBLAS_OP_T,
            k, n,                              // rows, cols of Vt
            &one,
            Vt_t.data_ptr<float>(), k,         // Vt: (k x n) col-major, lda=k
            tmp_t.data_ptr<float>(), 1,
            &zero,
            w_t.data_ptr<float>(), 1
        ));
    }

    return w_t.cpu();
}

// ---------------------------------------------------------------------------
// cuda_solve_qr
// ---------------------------------------------------------------------------
at::Tensor cuda_solve_qr(at::Tensor Z, at::Tensor y, bool z_col_major) {
    const int m = z_col_major ? static_cast<int>(Z.size(1)) : static_cast<int>(Z.size(0));
    const int n = z_col_major ? static_cast<int>(Z.size(0)) : static_cast<int>(Z.size(1));
    TORCH_CHECK(Z.dim() == 2, "Z must be 2D");
    TORCH_CHECK(y.dim() == 1, "y must be 1D (m,)");
    TORCH_CHECK(y.size(0) == m, "Z rows must match y length");
    TORCH_CHECK(m >= n, "cuda_solve_qr requires m >= n (overdetermined system)");
    TORCH_CHECK(Z.scalar_type() == at::kFloat, "Z must be float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "y must be float32");

    ensure_handles();

    at::Tensor Z_gpu = Z.to(at::kCUDA).contiguous();
    at::Tensor y_gpu = y.to(at::kCUDA).contiguous();

    // If col-major already, use Z directly as the QR buffer; otherwise transpose.
    at::Tensor QR;
    if (z_col_major) {
        QR = Z_gpu.clone();  // geqrf overwrites in-place; clone to avoid mutating input
    } else {
        QR = at::empty({m, n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgeam(
            s_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n,
            &one,
            Z_gpu.data_ptr<float>(), n,
            &zero,
            QR.data_ptr<float>(), m,
            QR.data_ptr<float>(), m
        ));
    }

    // Householder reflector coefficients tau (n,)
    at::Tensor tau =
        at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    at::Tensor d_info_t =
        at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    // Step 1: QR factorisation  QR ← QR factored form, tau ← Householder scalars
    {
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(s_cusolver, m, n,
            QR.data_ptr<float>(), m, &lwork));
        at::Tensor ws =
            at::empty({lwork}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        CUSOLVER_CHECK(cusolverDnSgeqrf(
            s_cusolver, m, n,
            QR.data_ptr<float>(), m,
            tau.data_ptr<float>(),
            ws.data_ptr<float>(), lwork,
            d_info_t.data_ptr<int>()
        ));
        int h = 0;
        CUDA_CHECK(cudaMemcpy(&h, d_info_t.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
        TORCH_CHECK(h == 0, "cusolverDnSgeqrf failed, info=", h);
    }

    // Step 2: rhs ← Q^T y   (Sormqr applies Q^T from stored Householder vectors)
    at::Tensor rhs = y_gpu.clone();  // (m,) — will be overwritten with Q^T y
    {
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(
            s_cusolver,
            CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
            m, 1, n,
            QR.data_ptr<float>(), m,
            tau.data_ptr<float>(),
            rhs.data_ptr<float>(), m,
            &lwork
        ));
        at::Tensor ws =
            at::empty({lwork}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        CUSOLVER_CHECK(cusolverDnSormqr(
            s_cusolver,
            CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
            m, 1, n,
            QR.data_ptr<float>(), m,
            tau.data_ptr<float>(),
            rhs.data_ptr<float>(), m,
            ws.data_ptr<float>(), lwork,
            d_info_t.data_ptr<int>()
        ));
        int h = 0;
        CUDA_CHECK(cudaMemcpy(&h, d_info_t.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
        TORCH_CHECK(h == 0, "cusolverDnSormqr failed, info=", h);
    }

    // Step 3: solve R x = rhs[0:n]  (upper triangular, stored in upper triangle of QR)
    // cublasSStrsv: x ← R^{-1} rhs[0:n] in-place
    {
        const float one = 1.0f;
        CUBLAS_CHECK(cublasStrsv(
            s_cublas,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n,
            QR.data_ptr<float>(), m,       // R stored col-major, lda=m
            rhs.data_ptr<float>(), 1       // rhs[0:n] in-place
        ));
    }

    // Solution is in rhs[0:n].
    return rhs.narrow(0, 0, n).contiguous().cpu();
}

// ---------------------------------------------------------------------------
// cuda_solve_gels
// ---------------------------------------------------------------------------
at::Tensor cuda_solve_gels(at::Tensor Z, at::Tensor y, bool z_col_major,
                            const std::string& variant) {
    const int m = z_col_major ? static_cast<int>(Z.size(1)) : static_cast<int>(Z.size(0));
    const int n = z_col_major ? static_cast<int>(Z.size(0)) : static_cast<int>(Z.size(1));
    TORCH_CHECK(Z.dim() == 2, "Z must be 2D");
    TORCH_CHECK(y.dim() == 1, "y must be 1D (m,)");
    TORCH_CHECK(y.size(0) == m, "Z rows must match y length");
    TORCH_CHECK(m >= n, "cuda_solve_gels requires m >= n (overdetermined system)");
    TORCH_CHECK(Z.scalar_type() == at::kFloat, "Z must be float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "y must be float32");

    ensure_handles();

    at::Tensor Z_gpu = Z.to(at::kCUDA).contiguous();
    at::Tensor y_gpu = y.to(at::kCUDA).contiguous();

    // A: (m x n) col-major.  SSgels overwrites A in-place → always clone.
    at::Tensor A;
    if (z_col_major) {
        A = Z_gpu.clone();  // already (n x m) stored as col-major (m x n), lda=m
    } else {
        A = at::empty({m, n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        const float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasSgeam(
            s_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            m, n,
            &one,
            Z_gpu.data_ptr<float>(), n,
            &zero,
            A.data_ptr<float>(), m,
            A.data_ptr<float>(), m
        ));
    }

    // B: (m x 1) col-major, lddb=m.  SSgels overwrites B → clone y.
    at::Tensor B = y_gpu.clone();   // contiguous (m,), lddb=m

    // X: output (n x 1) col-major, lddx=n.
    at::Tensor X = at::empty({n}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

    TORCH_CHECK(
        variant == "SS" || variant == "SH" || variant == "SB" || variant == "SX",
        "cuda_solve_gels: variant must be one of SS, SH, SB, SX (got \"", variant, "\")"
    );

    // Query workspace size.
    size_t lwork_bytes = 0;
    if (variant == "SS") {
        CUSOLVER_CHECK(cusolverDnSSgels_bufferSize(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            nullptr, &lwork_bytes));
    } else if (variant == "SH") {
        CUSOLVER_CHECK(cusolverDnSHgels_bufferSize(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            nullptr, &lwork_bytes));
    } else if (variant == "SB") {
        CUSOLVER_CHECK(cusolverDnSBgels_bufferSize(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            nullptr, &lwork_bytes));
    } else {  // SX
        CUSOLVER_CHECK(cusolverDnSXgels_bufferSize(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            nullptr, &lwork_bytes));
    }

    // Allocate workspace (byte buffer).
    at::Tensor ws = at::empty(
        {static_cast<int64_t>(lwork_bytes) + 1},
        at::TensorOptions().dtype(at::kByte).device(at::kCUDA)
    );

    cusolver_int_t h_iter = 0;
    at::Tensor d_info = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    if (variant == "SS") {
        CUSOLVER_CHECK(cusolverDnSSgels(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            static_cast<void*>(ws.data_ptr<uint8_t>()), lwork_bytes,
            &h_iter, d_info.data_ptr<int>()));
    } else if (variant == "SH") {
        CUSOLVER_CHECK(cusolverDnSHgels(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            static_cast<void*>(ws.data_ptr<uint8_t>()), lwork_bytes,
            &h_iter, d_info.data_ptr<int>()));
    } else if (variant == "SB") {
        CUSOLVER_CHECK(cusolverDnSBgels(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            static_cast<void*>(ws.data_ptr<uint8_t>()), lwork_bytes,
            &h_iter, d_info.data_ptr<int>()));
    } else {  // SX
        CUSOLVER_CHECK(cusolverDnSXgels(
            s_cusolver,
            static_cast<cusolver_int_t>(m), static_cast<cusolver_int_t>(n),
            static_cast<cusolver_int_t>(1),
            A.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            B.data_ptr<float>(), static_cast<cusolver_int_t>(m),
            X.data_ptr<float>(), static_cast<cusolver_int_t>(n),
            static_cast<void*>(ws.data_ptr<uint8_t>()), lwork_bytes,
            &h_iter, d_info.data_ptr<int>()));
    }

    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
    TORCH_CHECK(h_info == 0, "cusolverDn", variant, "gels failed, info=", h_info);
    if (h_iter < 0) {
        // iter < 0: IRS fell back to direct factorisation; result is still valid.
        // iter == -1: converged via direct method
        // iter == -2: iterative refinement failed (workspace), fell back
    }

    return X.cpu();
}

}  // namespace solvers
}  // namespace kf
