// cuda_local_kernels_bindings.cpp — PyTorch-extension bindings for
// kf::fchl19 CUDA local kernel functions.
//
// Exposed Python functions (module: cuda_local_kernels):
//
//   kernel_gaussian_full_symm(X, dX, Q, N, sigma)
//       X   : (nm, max_atoms, rep)                float32 CUDA
//       dX  : (nm, max_atoms, rep, 3*max_atoms)   float32 CUDA
//       Q   : (nm, max_atoms)                     int32   CUDA
//       N   : (nm,)                               int32   CUDA
//       sigma: float
//       → K_full : (nm+naq, nm+naq) float32 CUDA, fully symmetric
//
//   compute_alpha_desc(dX, N, alpha_F)
//       dX      : (nm, max_atoms, rep, 3*max_atoms) float32 CUDA
//       N       : (nm,)                             int32   CUDA
//       alpha_F : (naq,)                            float32 CUDA
//       → alpha_desc : (nm, max_atoms, rep) float32 CUDA
//
//   kernel_gaussian_full_matvec(X_q,dX_q,Q_q,N_q, X_t,Q_t,N_t, alpha_E,alpha_desc, sigma)
//       → (E_pred : (nm_q,) float32 CUDA,
//          F_pred : (naq_q,) float32 CUDA)

#include <torch/extension.h>
#include "cuda_local_kernels.hpp"

namespace py = pybind11;


// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

static void check_cuda_float32(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(),   name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_cuda_int32(const torch::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(),   name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_symm
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_full_symm_local(
    torch::Tensor X,   // (nm, max_atoms, rep)
    torch::Tensor dX,  // (nm, max_atoms, rep, 3*max_atoms)
    torch::Tensor Q,   // (nm, max_atoms)
    torch::Tensor N,   // (nm,)
    double sigma)
{
    check_cuda_float32(X,  "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q,    "Q");
    check_cuda_int32(N,    "N");

    TORCH_CHECK(X.dim()  == 3, "X must be 3-D (nm, max_atoms, rep)");
    TORCH_CHECK(dX.dim() == 4, "dX must be 4-D (nm, max_atoms, rep, 3*max_atoms)");
    TORCH_CHECK(Q.dim()  == 2, "Q must be 2-D (nm, max_atoms)");
    TORCH_CHECK(N.dim()  == 1, "N must be 1-D (nm,)");
    TORCH_CHECK(sigma > 0.0,   "sigma must be positive");

    int nm        = (int)X.size(0);
    int max_atoms = (int)X.size(1);
    int rep_size  = (int)X.size(2);

    TORCH_CHECK(dX.size(0) == nm,        "dX.size(0) must equal nm");
    TORCH_CHECK(dX.size(1) == max_atoms, "dX.size(1) must equal max_atoms");
    TORCH_CHECK(dX.size(2) == rep_size,  "dX.size(2) must equal rep_size");
    TORCH_CHECK(dX.size(3) == 3 * max_atoms, "dX.size(3) must equal 3*max_atoms");
    TORCH_CHECK(Q.size(0) == nm,         "Q.size(0) must equal nm");
    TORCH_CHECK(Q.size(1) == max_atoms,  "Q.size(1) must equal max_atoms");
    TORCH_CHECK(N.size(0) == nm,         "N.size(0) must equal nm");

    // Compute naq = 3 * sum(N) on CPU
    auto N_cpu  = N.cpu();
    const int *N_ptr = N_cpu.data_ptr<int>();
    int naq = 0;
    for (int m = 0; m < nm; m++) {
        int nm_ = N_ptr[m];
        if (nm_ > 0) naq += 3 * nm_;
    }

    long long BIG = (long long)nm + naq;
    auto K_full = torch::zeros({BIG, BIG}, X.options());

    kf::fchl19::kernel_gaussian_full_symm_local_cu(
        X.data_ptr<float>(),
        dX.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        K_full.data_ptr<float>(),
        (float)sigma,
        nm, max_atoms, rep_size, naq);

    return K_full;
}


// ---------------------------------------------------------------------------
// compute_alpha_desc
// ---------------------------------------------------------------------------

torch::Tensor compute_alpha_desc_local(
    torch::Tensor dX,      // (nm, max_atoms, rep, 3*max_atoms)
    torch::Tensor N,       // (nm,)
    torch::Tensor alpha_F) // (naq,)
{
    check_cuda_float32(dX,      "dX");
    check_cuda_int32(N,         "N");
    check_cuda_float32(alpha_F, "alpha_F");

    TORCH_CHECK(dX.dim() == 4, "dX must be 4-D (nm, max_atoms, rep, 3*max_atoms)");
    TORCH_CHECK(N.dim()  == 1, "N must be 1-D (nm,)");
    TORCH_CHECK(alpha_F.dim() == 1, "alpha_F must be 1-D (naq,)");

    int nm        = (int)dX.size(0);
    int max_atoms = (int)dX.size(1);
    int rep_size  = (int)dX.size(2);
    int naq       = (int)alpha_F.size(0);

    TORCH_CHECK(N.size(0) == nm, "N.size(0) must equal nm");
    TORCH_CHECK(dX.size(3) == 3 * max_atoms, "dX.size(3) must equal 3*max_atoms");

    // Output: (nm, max_atoms, rep)
    auto alpha_desc = torch::zeros({nm, max_atoms, rep_size}, dX.options());

    kf::fchl19::compute_alpha_desc_local_cu(
        dX.data_ptr<float>(),
        alpha_F.data_ptr<float>(),
        N.data_ptr<int>(),
        alpha_desc.data_ptr<float>(),
        nm, max_atoms, rep_size, naq);

    return alpha_desc;
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, torch::Tensor> kernel_gaussian_full_matvec_local(
    torch::Tensor X_q,          // (nm_q, max_atoms_q, rep)
    torch::Tensor dX_q,         // (nm_q, max_atoms_q, rep, 3*max_atoms_q)
    torch::Tensor Q_q,          // (nm_q, max_atoms_q)
    torch::Tensor N_q,          // (nm_q,)
    torch::Tensor X_t,          // (nm_t, max_atoms_t, rep)
    torch::Tensor Q_t,          // (nm_t, max_atoms_t)
    torch::Tensor N_t,          // (nm_t,)
    torch::Tensor alpha_E,      // (nm_t,)
    torch::Tensor alpha_desc,   // (nm_t, max_atoms_t, rep)
    double sigma)
{
    check_cuda_float32(X_q,       "X_q");
    check_cuda_float32(dX_q,      "dX_q");
    check_cuda_int32(Q_q,         "Q_q");
    check_cuda_int32(N_q,         "N_q");
    check_cuda_float32(X_t,       "X_t");
    check_cuda_int32(Q_t,         "Q_t");
    check_cuda_int32(N_t,         "N_t");
    check_cuda_float32(alpha_E,   "alpha_E");
    check_cuda_float32(alpha_desc,"alpha_desc");

    TORCH_CHECK(X_q.dim()  == 3, "X_q must be 3-D");
    TORCH_CHECK(dX_q.dim() == 4, "dX_q must be 4-D");
    TORCH_CHECK(Q_q.dim()  == 2, "Q_q must be 2-D");
    TORCH_CHECK(N_q.dim()  == 1, "N_q must be 1-D");
    TORCH_CHECK(X_t.dim()  == 3, "X_t must be 3-D");
    TORCH_CHECK(Q_t.dim()  == 2, "Q_t must be 2-D");
    TORCH_CHECK(N_t.dim()  == 1, "N_t must be 1-D");
    TORCH_CHECK(alpha_E.dim()   == 1, "alpha_E must be 1-D");
    TORCH_CHECK(alpha_desc.dim()== 3, "alpha_desc must be 3-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    int nm_q        = (int)X_q.size(0);
    int max_atoms_q = (int)X_q.size(1);
    int rep_size    = (int)X_q.size(2);
    int nm_t        = (int)X_t.size(0);
    int max_atoms_t = (int)X_t.size(1);

    TORCH_CHECK(X_t.size(2) == rep_size, "rep_size must match between X_q and X_t");

    // Compute naq_q = 3 * sum(N_q)
    auto N_q_cpu   = N_q.cpu();
    const int *Nq_ptr = N_q_cpu.data_ptr<int>();
    int naq_q = 0;
    for (int m = 0; m < nm_q; m++) {
        int n = Nq_ptr[m];
        if (n > 0) naq_q += 3 * n;
    }

    auto opts    = X_q.options();
    auto E_pred  = torch::zeros({nm_q},  opts);
    auto F_pred  = torch::zeros({naq_q}, opts);

    kf::fchl19::kernel_gaussian_full_matvec_local_cu(
        X_q.data_ptr<float>(),
        dX_q.data_ptr<float>(),
        Q_q.data_ptr<int>(),
        N_q.data_ptr<int>(),
        X_t.data_ptr<float>(),
        Q_t.data_ptr<int>(),
        N_t.data_ptr<int>(),
        alpha_E.data_ptr<float>(),
        alpha_desc.data_ptr<float>(),
        E_pred.data_ptr<float>(),
        F_pred.data_ptr<float>(),
        (float)sigma,
        nm_q, nm_t, max_atoms_q, max_atoms_t, rep_size, naq_q);

    return {E_pred, F_pred};
}


// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(cuda_local_kernels, m)
{
    m.doc() = R"doc(
CUDA-accelerated Gaussian kernel functions for local (FCHL19) molecular descriptors.

Mirrors the CPU ``local_kernels`` module but operates on CUDA (GPU) tensors.
All functions accept and return ``torch.Tensor`` objects on the same CUDA device.

Functions
---------
kernel_gaussian_full_symm(X, dX, Q, N, sigma)
    Build the symmetric (nm+naq)² energy+force kernel matrix for training.

compute_alpha_desc(dX, N, alpha_F)
    Precompute descriptor-space force weights for the J^T·alpha trick.

kernel_gaussian_full_matvec(X_q, dX_q, Q_q, N_q, X_t, Q_t, N_t, alpha_E, alpha_desc, sigma)
    Contracted E+F inference using the J^T·alpha trick (no K_test_train materialisation).
)doc";

    m.def("kernel_gaussian_full_symm",
          &kernel_gaussian_full_symm_local,
          py::arg("X"),
          py::arg("dX"),
          py::arg("Q"),
          py::arg("N"),
          py::arg("sigma"),
          R"doc(
Build the symmetric energy+force kernel matrix for local FCHL19 training.

Parameters
----------
X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
    Training atom descriptors (padded).
dX : torch.Tensor, shape (nm, max_atoms, rep, 3*max_atoms), float32, CUDA
    Training Jacobians (padded).
Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
    Atomic labels (nuclear charges).
N : torch.Tensor, shape (nm,), int32, CUDA
    Active atom counts per molecule.
sigma : float
    Gaussian length-scale.

Returns
-------
K_full : torch.Tensor, shape (nm+naq, nm+naq), float32, CUDA
    Fully symmetric kernel matrix.  naq = 3*sum(N).
)doc");

    m.def("compute_alpha_desc",
          &compute_alpha_desc_local,
          py::arg("dX"),
          py::arg("N"),
          py::arg("alpha_F"),
          R"doc(
Precompute descriptor-space force weights (J^T·alpha trick).

alpha_desc[b, i2, k] = sum_{c=0}^{3*N[b]-1} dX[b,i2,k,c] * alpha_F[offs[b]+c]

Call once after the KRR solve.  The result is passed to
``kernel_gaussian_full_matvec`` at inference time.

Parameters
----------
dX : torch.Tensor, shape (nm, max_atoms, rep, 3*max_atoms), float32, CUDA
N : torch.Tensor, shape (nm,), int32, CUDA
alpha_F : torch.Tensor, shape (naq,), float32, CUDA
    Force coefficients from the KRR solve (alpha[nm:]).

Returns
-------
alpha_desc : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
)doc");

    m.def("kernel_gaussian_full_matvec",
          &kernel_gaussian_full_matvec_local,
          py::arg("X_q"),
          py::arg("dX_q"),
          py::arg("Q_q"),
          py::arg("N_q"),
          py::arg("X_t"),
          py::arg("Q_t"),
          py::arg("N_t"),
          py::arg("alpha_E"),
          py::arg("alpha_desc"),
          py::arg("sigma"),
          R"doc(
Contracted energy+force inference using the J^T·alpha trick (local version).

Does not materialise the full test-train kernel matrix.

Parameters
----------
X_q : torch.Tensor, shape (nm_q, max_atoms_q, rep), float32, CUDA
dX_q : torch.Tensor, shape (nm_q, max_atoms_q, rep, 3*max_atoms_q), float32, CUDA
Q_q : torch.Tensor, shape (nm_q, max_atoms_q), int32, CUDA
N_q : torch.Tensor, shape (nm_q,), int32, CUDA
X_t : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
Q_t : torch.Tensor, shape (nm_t, max_atoms_t), int32, CUDA
N_t : torch.Tensor, shape (nm_t,), int32, CUDA
alpha_E : torch.Tensor, shape (nm_t,), float32, CUDA
alpha_desc : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
sigma : float

Returns
-------
E_pred : torch.Tensor, shape (nm_q,), float32, CUDA
F_pred : torch.Tensor, shape (naq_q,), float32, CUDA  (flat Cartesian forces)
)doc");
}
