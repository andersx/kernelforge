// cuda_global_kernels_bindings.cpp — PyTorch-extension bindings for
// kf::kernel_gaussian_full_symm_cu and kf::kernel_gaussian_full_matvec_cu.
//
// Exposed Python functions (module: cuda_global_kernels):
//
//   kernel_gaussian_full_symm(X, dX, sigma)
//       X   : (N, M)    float32 CUDA tensor
//       dX  : (N, D, M) float32 CUDA tensor
//       sigma: float
//       → K_full : (N*(1+D), N*(1+D)) float32 CUDA tensor, fully symmetric
//
//   kernel_gaussian_full_matvec(X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma)
//       X_q          : (N_q, M)    float32 CUDA
//       dX_q         : (N_q, D, M) float32 CUDA
//       X_t          : (N_t, M)    float32 CUDA
//       alpha_E      : (N_t,)      float32 CUDA
//       alpha_desc_F : (N_t, M)    float32 CUDA  [= Σ_d J_t[m,d,:] * alpha_F[m,d]]
//       sigma        : float
//       → (E_pred : (N_q,) float32 CUDA,
//          F_pred : (N_q, D) float32 CUDA)

#include <torch/extension.h>
#include "cuda_global_kernels.hpp"

namespace py = pybind11;


// ---------------------------------------------------------------------------
// Validation helper
// ---------------------------------------------------------------------------

static void check_cuda_float32(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(),
                name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32,
                name, " must be float32");
    TORCH_CHECK(t.is_contiguous(),
                name, " must be contiguous");
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_full_symm(
    torch::Tensor X,   // (N, M)
    torch::Tensor dX,  // (N, D, M)
    double sigma)
{
    check_cuda_float32(X,  "X");
    check_cuda_float32(dX, "dX");
    TORCH_CHECK(X.dim() == 2,  "X must be 2-D (N, M)");
    TORCH_CHECK(dX.dim() == 3, "dX must be 3-D (N, D, M)");
    TORCH_CHECK(sigma > 0.0,   "sigma must be positive");

    int N = (int)X.size(0);
    int M = (int)X.size(1);
    int D = (int)dX.size(1);
    TORCH_CHECK(dX.size(0) == N, "dX.size(0) must equal N");
    TORCH_CHECK(dX.size(2) == M, "dX.size(2) must equal M");

    long long full = (long long)N * (1 + D);

    // Allocate output (square, float32, CUDA).
    // Since K_full is symmetric, the col-major byte layout
    // stored by cuBLAS equals the row-major layout PyTorch uses.
    auto K_full = torch::empty({full, full}, X.options());

    // Reshape dX: (N, D, M) → (N*D, M), same contiguous storage.
    auto dXT = dX.reshape({(long long)N * D, M});

    kf::kernel_gaussian_full_symm_cu(
        X.data_ptr<float>(),
        dXT.data_ptr<float>(),
        K_full.data_ptr<float>(),
        (float)sigma, N, M, D);

    return K_full;
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, torch::Tensor> kernel_gaussian_full_matvec(
    torch::Tensor X_q,           // (N_q, M)
    torch::Tensor dX_q,          // (N_q, D, M)
    torch::Tensor X_t,           // (N_t, M)
    torch::Tensor alpha_E,       // (N_t,)
    torch::Tensor alpha_desc_F,  // (N_t, M)
    double sigma)
{
    check_cuda_float32(X_q,          "X_q");
    check_cuda_float32(dX_q,         "dX_q");
    check_cuda_float32(X_t,          "X_t");
    check_cuda_float32(alpha_E,      "alpha_E");
    check_cuda_float32(alpha_desc_F, "alpha_desc_F");

    TORCH_CHECK(X_q.dim()  == 2, "X_q must be 2-D (N_q, M)");
    TORCH_CHECK(dX_q.dim() == 3, "dX_q must be 3-D (N_q, D, M)");
    TORCH_CHECK(X_t.dim()  == 2, "X_t must be 2-D (N_t, M)");
    TORCH_CHECK(alpha_E.dim()      == 1, "alpha_E must be 1-D (N_t,)");
    TORCH_CHECK(alpha_desc_F.dim() == 2, "alpha_desc_F must be 2-D (N_t, M)");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    int N_q = (int)X_q.size(0);
    int M   = (int)X_q.size(1);
    int D   = (int)dX_q.size(1);
    int N_t = (int)X_t.size(0);

    TORCH_CHECK(dX_q.size(0) == N_q,       "dX_q.size(0) must equal N_q");
    TORCH_CHECK(dX_q.size(2) == M,         "dX_q.size(2) must equal M");
    TORCH_CHECK(X_t.size(1)  == M,         "X_t.size(1) must equal M");
    TORCH_CHECK(alpha_E.size(0) == N_t,    "alpha_E.size(0) must equal N_t");
    TORCH_CHECK(alpha_desc_F.size(0) == N_t, "alpha_desc_F.size(0) must equal N_t");
    TORCH_CHECK(alpha_desc_F.size(1) == M,   "alpha_desc_F.size(1) must equal M");

    // Allocate outputs
    auto opts  = X_q.options();
    auto E_pred = torch::empty({N_q},           opts);
    auto F_pred = torch::empty({N_q, D},        opts);

    // Reshape dX_q: (N_q, D, M) → (N_q*D, M)
    auto dXT_q = dX_q.reshape({(long long)N_q * D, M});

    kf::kernel_gaussian_full_matvec_cu(
        X_q.data_ptr<float>(),
        dXT_q.data_ptr<float>(),
        X_t.data_ptr<float>(),
        alpha_E.data_ptr<float>(),
        alpha_desc_F.data_ptr<float>(),
        E_pred.data_ptr<float>(),
        F_pred.data_ptr<float>(),
        (float)sigma, N_q, N_t, M, D);

    return {E_pred, F_pred};
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm_rfp
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_full_symm_rfp(
    torch::Tensor X,   // (N, M)
    torch::Tensor dX,  // (N, D, M)
    double sigma)
{
    check_cuda_float32(X,  "X");
    check_cuda_float32(dX, "dX");
    TORCH_CHECK(X.dim() == 2,  "X must be 2-D (N, M)");
    TORCH_CHECK(dX.dim() == 3, "dX must be 3-D (N, D, M)");
    TORCH_CHECK(sigma > 0.0,   "sigma must be positive");

    int N = (int)X.size(0);
    int M = (int)X.size(1);
    int D = (int)dX.size(1);
    TORCH_CHECK(dX.size(0) == N, "dX.size(0) must equal N");
    TORCH_CHECK(dX.size(2) == M, "dX.size(2) must equal M");

    long long BIG = (long long)N * (1 + D);
    long long nt  = BIG * (BIG + 1) / 2;

    auto K_rfp = torch::empty({nt}, X.options());

    // Reshape dX: (N, D, M) → (N*D, M), same contiguous storage.
    auto dXT = dX.reshape({(long long)N * D, M});

    kf::kernel_gaussian_full_symm_rfp_cu(
        X.data_ptr<float>(),
        dXT.data_ptr<float>(),
        K_rfp.data_ptr<float>(),
        (float)sigma, N, M, D);

    return K_rfp;
}


// ---------------------------------------------------------------------------
// kernel_gaussian_symm_rfp
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_symm_rfp(
    torch::Tensor X,   // (N, M)
    double sigma)
{
    check_cuda_float32(X, "X");
    TORCH_CHECK(X.dim() == 2, "X must be 2-D (N, M)");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    int N = (int)X.size(0);
    int M = (int)X.size(1);

    long long nt = (long long)N * (N + 1) / 2;
    auto K_rfp = torch::empty({nt}, X.options());

    kf::kernel_gaussian_symm_rfp_cu(
        X.data_ptr<float>(),
        K_rfp.data_ptr<float>(),
        (float)sigma, N, M);

    return K_rfp;
}


// ---------------------------------------------------------------------------
// rfp_potrf
// ---------------------------------------------------------------------------

int rfp_potrf(torch::Tensor K_rfp, int N, double l2)
{
    check_cuda_float32(K_rfp, "K_rfp");
    TORCH_CHECK(K_rfp.dim() == 1, "K_rfp must be 1-D");
    TORCH_CHECK(K_rfp.size(0) == (long long)N * (N + 1) / 2,
                "K_rfp.size(0) must equal N*(N+1)/2");

    int info = 0;
    kf::rfp_potrf_cu(K_rfp.data_ptr<float>(), N, (float)l2, &info);
    return info;
}


// ---------------------------------------------------------------------------
// rfp_potrs
// ---------------------------------------------------------------------------

void rfp_potrs(torch::Tensor L_rfp, torch::Tensor B)
{
    check_cuda_float32(L_rfp, "L_rfp");
    check_cuda_float32(B,     "B");
    TORCH_CHECK(L_rfp.dim() == 1, "L_rfp must be 1-D");
    TORCH_CHECK(B.dim() == 2,     "B must be 2-D (N, nrhs)");

    int N    = (int)B.size(0);
    int nrhs = (int)B.size(1);
    TORCH_CHECK(L_rfp.size(0) == (long long)N * (N + 1) / 2,
                "L_rfp.size(0) must equal N*(N+1)/2");

    kf::rfp_potrs_cu(L_rfp.data_ptr<float>(), B.data_ptr<float>(), N, nrhs);
}


// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(cuda_global_kernels, m)
{
    m.doc() = R"doc(
CUDA-accelerated Gaussian kernel functions for global molecular descriptors.

Mirrors the CPU ``global_kernels`` module but operates on CUDA (GPU) tensors.
All functions accept and return ``torch.Tensor`` objects on the same CUDA device.

Functions
---------
kernel_gaussian_full_symm(X, dX, sigma)
    Build the symmetric (N*(1+D))² energy+force kernel matrix for training.

kernel_gaussian_full_symm_rfp(X, dX, sigma)
    Build the energy+force kernel matrix directly in RFP packed format
    (TRANSR=N, UPLO=L).  No dense BIG×BIG intermediate is allocated.

kernel_gaussian_full_matvec(X_q, dX_q, X_t, alpha_E, alpha_desc_F, sigma)
    Contracted E+F inference using the J^T·alpha trick (no full K_test_train
    materialisation).

kernel_gaussian_symm_rfp(X, sigma)
    Build the energy-only N×N Gaussian kernel matrix in RFP packed format.

rfp_potrf(K_rfp, N, l2)
    Cholesky factorisation of an RFP matrix (with optional diagonal shift l2).
    Returns *info* (0 = success).

rfp_potrs(L_rfp, B)
    Triangular solve with Cholesky factor from rfp_potrf.  B is overwritten
    with the solution X.
)doc";

    m.def("kernel_gaussian_full_symm",
          &kernel_gaussian_full_symm,
          py::arg("X"),
          py::arg("dX"),
          py::arg("sigma"),
          R"doc(
Build the symmetric energy+force kernel matrix for training.

Parameters
----------
X : torch.Tensor, shape (N, M), float32, CUDA
    Training descriptors.
dX : torch.Tensor, shape (N, D, M), float32, CUDA
    Training Jacobians.
sigma : float
    Gaussian length-scale.

Returns
-------
K_full : torch.Tensor, shape (N*(1+D), N*(1+D)), float32, CUDA
    Fully symmetric kernel matrix (both triangles filled).
)doc");

    m.def("kernel_gaussian_full_symm_rfp",
          &kernel_gaussian_full_symm_rfp,
          py::arg("X"),
          py::arg("dX"),
          py::arg("sigma"),
          R"doc(
Build the energy+force kernel matrix directly in RFP packed format.

Like kernel_gaussian_full_symm but stores the result as a 1-D RFP packed
buffer (TRANSR=N, UPLO=L) instead of a dense BIG×BIG matrix.  Each
lower-triangle element is written once; no dense intermediate is allocated
and no mirror step is performed.

Parameters
----------
X : torch.Tensor, shape (N, M), float32, CUDA
    Training descriptors.
dX : torch.Tensor, shape (N, D, M), float32, CUDA
    Training Jacobians.
sigma : float
    Gaussian length-scale.

Returns
-------
K_rfp : torch.Tensor, shape (BIG*(BIG+1)//2,), float32, CUDA
    Lower-triangular RFP packed kernel matrix (TRANSR=N, UPLO=L),
    where BIG = N*(1+D).
)doc");

    m.def("kernel_gaussian_full_matvec",
          &kernel_gaussian_full_matvec,
          py::arg("X_q"),
          py::arg("dX_q"),
          py::arg("X_t"),
          py::arg("alpha_E"),
          py::arg("alpha_desc_F"),
          py::arg("sigma"),
          R"doc(
Contracted energy+force inference (J^T·alpha trick).

Computes E and F predictions without materialising the full test-train kernel
matrix.  Requires alpha_desc_F = einsum('ndm,nd->nm', dX_train, alpha_F),
precomputed once after training.

Parameters
----------
X_q : torch.Tensor, shape (N_q, M), float32, CUDA
    Query descriptors.
dX_q : torch.Tensor, shape (N_q, D, M), float32, CUDA
    Query Jacobians.
X_t : torch.Tensor, shape (N_t, M), float32, CUDA
    Training descriptors.
alpha_E : torch.Tensor, shape (N_t,), float32, CUDA
    Energy dual coefficients.
alpha_desc_F : torch.Tensor, shape (N_t, M), float32, CUDA
    Contracted force weights: Σ_d J_train[m,d,:] * alpha_F[m,d].
sigma : float
    Gaussian length-scale.

Returns
-------
E_pred : torch.Tensor, shape (N_q,), float32, CUDA
F_pred : torch.Tensor, shape (N_q, D), float32, CUDA
    Physical forces F = -dE/dR.
)doc");

    m.def("kernel_gaussian_symm_rfp",
          &kernel_gaussian_symm_rfp,
          py::arg("X"),
          py::arg("sigma"),
          R"doc(
Build the energy-only N×N Gaussian kernel matrix in RFP packed format.

Uses TRANSR=N, UPLO=L convention throughout.  The packed buffer has
N*(N+1)/2 elements and can be passed directly to rfp_potrf / rfp_potrs.

Parameters
----------
X : torch.Tensor, shape (N, M), float32, CUDA
    Training descriptors.
sigma : float
    Gaussian length-scale.

Returns
-------
K_rfp : torch.Tensor, shape (N*(N+1)//2,), float32, CUDA
    Lower-triangular RFP packed kernel matrix (TRANSR=N, UPLO=L).
)doc");

    m.def("rfp_potrf",
          &rfp_potrf,
          py::arg("K_rfp"),
          py::arg("N"),
          py::arg("l2") = 0.0,
          R"doc(
Cholesky factorisation of an RFP-packed symmetric positive definite matrix.

Optionally adds l2 to the diagonal before factorising (Tikhonov
regularisation).  The input buffer is overwritten with the lower Cholesky
factor L.

Convention: TRANSR=N, UPLO=L.

Parameters
----------
K_rfp : torch.Tensor, shape (N*(N+1)//2,), float32, CUDA
    RFP-packed matrix.  Modified in-place.
N : int
    Matrix dimension.
l2 : float, default 0.0
    Diagonal regularisation added before factorisation.

Returns
-------
info : int
    0 on success; positive value k means the k-th leading minor is not
    positive definite.
)doc");

    m.def("rfp_potrs",
          &rfp_potrs,
          py::arg("L_rfp"),
          py::arg("B"),
          R"doc(
Triangular solve using a Cholesky factor produced by rfp_potrf.

Solves (L * L^T) * X = B, overwriting B with the solution X.
Convention: TRANSR=N, UPLO=L.

Parameters
----------
L_rfp : torch.Tensor, shape (N*(N+1)//2,), float32, CUDA
    Lower Cholesky factor in RFP format (from rfp_potrf).
B : torch.Tensor, shape (N, nrhs), float32, CUDA
    Right-hand side matrix.  Modified in-place; on exit holds the solution.
)doc");
}
