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
// kernel_gaussian_full_symm_rfp (energy+force in RFP packed format)
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_full_symm_rfp_local(
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

    auto N_cpu  = N.cpu();
    const int *N_ptr = N_cpu.data_ptr<int>();
    int naq = 0;
    for (int m = 0; m < nm; m++) {
        int nm_ = N_ptr[m];
        if (nm_ > 0) naq += 3 * nm_;
    }

    long long BIG = (long long)nm + naq;
    long long nt  = BIG * (BIG + 1) / 2;
    auto K_rfp = torch::zeros({nt}, X.options());

    kf::fchl19::kernel_gaussian_full_symm_rfp_local_cu(
        X.data_ptr<float>(),
        dX.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        K_rfp.data_ptr<float>(),
        (float)sigma,
        nm, max_atoms, rep_size, naq);

    return K_rfp;
}


// ---------------------------------------------------------------------------
// kernel_gaussian_symm (energy-only K_EE)
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_symm_local(
    torch::Tensor X,   // (nm, max_atoms, rep)
    torch::Tensor Q,   // (nm, max_atoms)
    torch::Tensor N,   // (nm,)
    double sigma)
{
    check_cuda_float32(X, "X");
    check_cuda_int32(Q,   "Q");
    check_cuda_int32(N,   "N");

    TORCH_CHECK(X.dim() == 3, "X must be 3-D (nm, max_atoms, rep)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2-D (nm, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1-D (nm,)");
    TORCH_CHECK(sigma > 0.0,  "sigma must be positive");

    int nm        = (int)X.size(0);
    int max_atoms = (int)X.size(1);
    int rep_size  = (int)X.size(2);

    auto KEE = torch::zeros({nm, nm}, X.options());

    kf::fchl19::kernel_gaussian_symm_local_cu(
        X.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        KEE.data_ptr<float>(),
        (float)sigma,
        nm, max_atoms, rep_size);

    return KEE;
}


// ---------------------------------------------------------------------------
// kernel_gaussian_symm_rfp (energy-only K_EE in RFP packed format)
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_symm_rfp_local(
    torch::Tensor X,   // (nm, max_atoms, rep)
    torch::Tensor Q,   // (nm, max_atoms)
    torch::Tensor N,   // (nm,)
    double sigma)
{
    check_cuda_float32(X, "X");
    check_cuda_int32(Q,   "Q");
    check_cuda_int32(N,   "N");

    TORCH_CHECK(X.dim() == 3, "X must be 3-D (nm, max_atoms, rep)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2-D (nm, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1-D (nm,)");
    TORCH_CHECK(sigma > 0.0,  "sigma must be positive");

    int nm        = (int)X.size(0);
    int max_atoms = (int)X.size(1);
    int rep_size  = (int)X.size(2);

    long long nt = (long long)nm * (nm + 1) / 2;
    auto K_rfp = torch::zeros({nt}, X.options());

    kf::fchl19::kernel_gaussian_symm_rfp_local_cu(
        X.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        K_rfp.data_ptr<float>(),
        (float)sigma,
        nm, max_atoms, rep_size);

    return K_rfp;
}


// ---------------------------------------------------------------------------
// kernel_gaussian_rect (energy-only rectangular K_EE)
// ---------------------------------------------------------------------------

torch::Tensor kernel_gaussian_rect_local(
    torch::Tensor X_q,  // (nm_q, max_atoms_q, rep)
    torch::Tensor Q_q,  // (nm_q, max_atoms_q)
    torch::Tensor N_q,  // (nm_q,)
    torch::Tensor X_t,  // (nm_t, max_atoms_t, rep)
    torch::Tensor Q_t,  // (nm_t, max_atoms_t)
    torch::Tensor N_t,  // (nm_t,)
    double sigma)
{
    check_cuda_float32(X_q, "X_q");
    check_cuda_int32(Q_q,   "Q_q");
    check_cuda_int32(N_q,   "N_q");
    check_cuda_float32(X_t, "X_t");
    check_cuda_int32(Q_t,   "Q_t");
    check_cuda_int32(N_t,   "N_t");

    int nm_q        = (int)X_q.size(0);
    int max_atoms_q = (int)X_q.size(1);
    int rep_size    = (int)X_q.size(2);
    int nm_t        = (int)X_t.size(0);
    int max_atoms_t = (int)X_t.size(1);

    TORCH_CHECK(X_t.size(2) == rep_size, "rep_size must match between X_q and X_t");

    auto KEE = torch::zeros({nm_q, nm_t}, X_q.options());

    kf::fchl19::kernel_gaussian_rect_local_cu(
        X_q.data_ptr<float>(),
        Q_q.data_ptr<int>(),
        N_q.data_ptr<int>(),
        X_t.data_ptr<float>(),
        Q_t.data_ptr<int>(),
        N_t.data_ptr<int>(),
        KEE.data_ptr<float>(),
        (float)sigma,
        nm_q, nm_t, max_atoms_q, max_atoms_t, rep_size);

    return KEE;
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
// precompute_train
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> precompute_train_local(
    torch::Tensor X_t,         // (nm_t, max_atoms_t, rep)
    torch::Tensor Q_t,         // (nm_t, max_atoms_t)  — unused here but kept for API symmetry
    torch::Tensor N_t,         // (nm_t,)
    torch::Tensor alpha_E,     // (nm_t,)
    torch::Tensor alpha_desc)  // (nm_t, max_atoms_t, rep)
{
    check_cuda_float32(X_t,        "X_t");
    check_cuda_int32(N_t,          "N_t");
    check_cuda_float32(alpha_E,    "alpha_E");
    check_cuda_float32(alpha_desc, "alpha_desc");

    TORCH_CHECK(X_t.dim()        == 3, "X_t must be 3-D (nm_t, max_atoms_t, rep)");
    TORCH_CHECK(N_t.dim()        == 1, "N_t must be 1-D (nm_t,)");
    TORCH_CHECK(alpha_E.dim()    == 1, "alpha_E must be 1-D (nm_t,)");
    TORCH_CHECK(alpha_desc.dim() == 3, "alpha_desc must be 3-D (nm_t, max_atoms_t, rep)");

    int nm_t        = (int)X_t.size(0);
    int max_atoms_t = (int)X_t.size(1);
    int rep_size    = (int)X_t.size(2);
    int N_t_flat    = nm_t * max_atoms_t;

    auto opts = X_t.options();
    auto norms_t    = torch::empty({N_t_flat},           opts);
    auto S_adF      = torch::empty({N_t_flat},           opts);
    auto alpha_E_t  = torch::empty({N_t_flat},           opts);
    auto combined_t = torch::empty({N_t_flat, rep_size}, opts);

    kf::fchl19::kernel_gaussian_precompute_train_local_cu(
        X_t.data_ptr<float>(),
        alpha_desc.data_ptr<float>(),
        alpha_E.data_ptr<float>(),
        N_t.data_ptr<int>(),
        norms_t.data_ptr<float>(),
        S_adF.data_ptr<float>(),
        alpha_E_t.data_ptr<float>(),
        combined_t.data_ptr<float>(),
        nm_t, max_atoms_t, rep_size);

    return {norms_t, S_adF, alpha_E_t, combined_t};
}


// ---------------------------------------------------------------------------
// kernel_gaussian_full_matvec_cached
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, torch::Tensor> kernel_gaussian_full_matvec_cached_local(
    torch::Tensor X_q,          // (nm_q, max_atoms_q, rep)
    torch::Tensor dX_q,         // (nm_q, max_atoms_q, rep, 3*max_atoms_q)
    torch::Tensor Q_q,          // (nm_q, max_atoms_q)
    torch::Tensor N_q,          // (nm_q,)
    torch::Tensor X_t,          // (nm_t, max_atoms_t, rep)
    torch::Tensor Q_t,          // (nm_t, max_atoms_t)
    torch::Tensor N_t,          // (nm_t,)
    torch::Tensor alpha_E,      // (nm_t,)
    torch::Tensor alpha_desc,   // (nm_t, max_atoms_t, rep)
    torch::Tensor norms_t,      // (nm_t * max_atoms_t,)   precomputed
    torch::Tensor S_adF,        // (nm_t * max_atoms_t,)   precomputed
    torch::Tensor alpha_E_t,    // (nm_t * max_atoms_t,)   precomputed
    torch::Tensor combined_t,   // (nm_t * max_atoms_t, rep) precomputed
    double sigma)
{
    check_cuda_float32(X_q,         "X_q");
    check_cuda_float32(dX_q,        "dX_q");
    check_cuda_int32(Q_q,           "Q_q");
    check_cuda_int32(N_q,           "N_q");
    check_cuda_float32(X_t,         "X_t");
    check_cuda_int32(Q_t,           "Q_t");
    check_cuda_int32(N_t,           "N_t");
    check_cuda_float32(alpha_E,     "alpha_E");
    check_cuda_float32(alpha_desc,  "alpha_desc");
    check_cuda_float32(norms_t,     "norms_t");
    check_cuda_float32(S_adF,       "S_adF");
    check_cuda_float32(alpha_E_t,   "alpha_E_t");
    check_cuda_float32(combined_t,  "combined_t");

    TORCH_CHECK(X_q.dim()        == 3, "X_q must be 3-D");
    TORCH_CHECK(dX_q.dim()       == 4, "dX_q must be 4-D");
    TORCH_CHECK(Q_q.dim()        == 2, "Q_q must be 2-D");
    TORCH_CHECK(N_q.dim()        == 1, "N_q must be 1-D");
    TORCH_CHECK(X_t.dim()        == 3, "X_t must be 3-D");
    TORCH_CHECK(Q_t.dim()        == 2, "Q_t must be 2-D");
    TORCH_CHECK(N_t.dim()        == 1, "N_t must be 1-D");
    TORCH_CHECK(alpha_E.dim()    == 1, "alpha_E must be 1-D");
    TORCH_CHECK(alpha_desc.dim() == 3, "alpha_desc must be 3-D");
    TORCH_CHECK(norms_t.dim()    == 1, "norms_t must be 1-D");
    TORCH_CHECK(S_adF.dim()      == 1, "S_adF must be 1-D");
    TORCH_CHECK(alpha_E_t.dim()  == 1, "alpha_E_t must be 1-D");
    TORCH_CHECK(combined_t.dim() == 2, "combined_t must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    int nm_q        = (int)X_q.size(0);
    int max_atoms_q = (int)X_q.size(1);
    int rep_size    = (int)X_q.size(2);
    int nm_t        = (int)X_t.size(0);
    int max_atoms_t = (int)X_t.size(1);

    TORCH_CHECK(X_t.size(2) == rep_size, "rep_size must match between X_q and X_t");

    // Compute naq_q = 3 * sum(N_q)
    auto N_q_cpu = N_q.cpu();
    const int *Nq_ptr = N_q_cpu.data_ptr<int>();
    int naq_q = 0;
    for (int m = 0; m < nm_q; m++) {
        int n = Nq_ptr[m];
        if (n > 0) naq_q += 3 * n;
    }

    auto opts   = X_q.options();
    auto E_pred = torch::zeros({nm_q},  opts);
    auto F_pred = torch::zeros({naq_q}, opts);

    kf::fchl19::kernel_gaussian_full_matvec_cached_local_cu(
        X_q.data_ptr<float>(),
        dX_q.data_ptr<float>(),
        Q_q.data_ptr<int>(),
        N_q.data_ptr<int>(),
        X_t.data_ptr<float>(),
        Q_t.data_ptr<int>(),
        N_t.data_ptr<int>(),
        alpha_E.data_ptr<float>(),
        alpha_desc.data_ptr<float>(),
        norms_t.data_ptr<float>(),
        S_adF.data_ptr<float>(),
        alpha_E_t.data_ptr<float>(),
        combined_t.data_ptr<float>(),
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

    m.def("kernel_gaussian_full_symm_rfp",
          &kernel_gaussian_full_symm_rfp_local,
          py::arg("X"),
          py::arg("dX"),
          py::arg("Q"),
          py::arg("N"),
          py::arg("sigma"),
          R"doc(
Build the symmetric energy+force kernel matrix in RFP packed format.

Convention: TRANSR=N, UPLO=L.  BIG = nm + naq, naq = 3*sum(N).
Unpack with: kernelmath.rfp_to_full(K_rfp, BIG, uplo='U', transr='N')

Parameters
----------
X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
dX : torch.Tensor, shape (nm, max_atoms, rep, 3*max_atoms), float32, CUDA
Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
N : torch.Tensor, shape (nm,), int32, CUDA
sigma : float

Returns
-------
K_rfp : torch.Tensor, shape (BIG*(BIG+1)//2,), float32, CUDA
)doc");

    m.def("kernel_gaussian_symm",
          &kernel_gaussian_symm_local,
          py::arg("X"),
          py::arg("Q"),
          py::arg("N"),
          py::arg("sigma"),
          R"doc(
Build the symmetric energy-only kernel matrix K_EE (nm × nm).

Parameters
----------
X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
N : torch.Tensor, shape (nm,), int32, CUDA
sigma : float

Returns
-------
K_EE : torch.Tensor, shape (nm, nm), float32, CUDA
)doc");

    m.def("kernel_gaussian_symm_rfp",
          &kernel_gaussian_symm_rfp_local,
          py::arg("X"),
          py::arg("Q"),
          py::arg("N"),
          py::arg("sigma"),
          R"doc(
Build the symmetric energy-only kernel matrix K_EE in RFP packed format.

Convention: TRANSR=N, UPLO=L.
Unpack with: kernelmath.rfp_to_full(K_rfp, nm, uplo='U', transr='N')

Parameters
----------
X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
N : torch.Tensor, shape (nm,), int32, CUDA
sigma : float

Returns
-------
K_rfp : torch.Tensor, shape (nm*(nm+1)//2,), float32, CUDA
)doc");

    m.def("kernel_gaussian_rect",
          &kernel_gaussian_rect_local,
          py::arg("X_q"), py::arg("Q_q"), py::arg("N_q"),
          py::arg("X_t"), py::arg("Q_t"), py::arg("N_t"),
          py::arg("sigma"),
          "Build rectangular energy-only kernel K_EE (nm_q × nm_t).");

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

    m.def("precompute_train",
          &precompute_train_local,
          py::arg("X_t"),
          py::arg("Q_t"),
          py::arg("N_t"),
          py::arg("alpha_E"),
          py::arg("alpha_desc"),
          R"doc(
Precompute training-side constants for repeated inference (e.g. MD simulation).

These three tensors are fixed as long as X_t, alpha_E, and alpha_desc do not
change.  Pass the returned tensors to ``kernel_gaussian_full_matvec_cached``
at every inference step to avoid recomputing them.

Parameters
----------
X_t : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
Q_t : torch.Tensor, shape (nm_t, max_atoms_t), int32, CUDA
N_t : torch.Tensor, shape (nm_t,), int32, CUDA
alpha_E : torch.Tensor, shape (nm_t,), float32, CUDA
alpha_desc : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA

Returns
-------
norms_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
    Precomputed squared norms ||X_t[t]||^2.
S_adF : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
    Precomputed dot products X_t[t] · alpha_desc[t].
alpha_E_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
    Atom-expanded energy dual coefficients alpha_E[mol(t)].
combined_t : torch.Tensor, shape (nm_t * max_atoms_t, rep), float32, CUDA
    Precomputed combined matrix alpha_desc + alpha_E_t[t] * X_t.
)doc");

    m.def("kernel_gaussian_full_matvec_cached",
          &kernel_gaussian_full_matvec_cached_local,
          py::arg("X_q"),
          py::arg("dX_q"),
          py::arg("Q_q"),
          py::arg("N_q"),
          py::arg("X_t"),
          py::arg("Q_t"),
          py::arg("N_t"),
          py::arg("alpha_E"),
          py::arg("alpha_desc"),
          py::arg("norms_t"),
          py::arg("S_adF"),
          py::arg("alpha_E_t"),
          py::arg("combined_t"),
          py::arg("sigma"),
          R"doc(
Cached variant of kernel_gaussian_full_matvec for repeated inference.

Compared to ``kernel_gaussian_full_matvec``, this version accepts four
precomputed training-side tensors (from ``precompute_train``) and avoids
recomputing them on every call.  Significantly faster for MD simulations where
the training set is fixed across all steps.

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
norms_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
S_adF : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
combined_t : torch.Tensor, shape (nm_t * max_atoms_t, rep), float32, CUDA
sigma : float

Returns
-------
E_pred : torch.Tensor, shape (nm_q,), float32, CUDA
F_pred : torch.Tensor, shape (naq_q,), float32, CUDA  (flat Cartesian forces)
)doc");
}
