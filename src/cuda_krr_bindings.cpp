/*
 * cuda_krr_bindings.cpp — Pybind11 Python bindings for cuda_krr.
 *
 * Exposes a single class `CudaKRRState` which wraps a KrrInferenceState
 * and provides:
 *   train_ef(X, dXT, E, F, sigma, lambda) -> alpha ndarray
 *   predict(X_te, dXT_te)                 -> (E_pred, F_pred)
 *   get_state()                           -> tuple of 6 ndarrays (for save)
 *   load_state(arrays..., sigma, N, M, D) -> (for load)
 *
 * Memory layout convention (see cuda_krr.h):
 *   X    : numpy (N, M) float32 C-contiguous  <-> cuBLAS (M, N) col-major
 *   dXT  : numpy (N*D, M) float32 C-contiguous <-> cuBLAS (M, N*D) col-major
 *          The Python layer passes dX (N, D, M) reshaped to (N*D, M).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>
#include <cstring>

#include "cuda_krr.h"

namespace py = pybind11;

/* Convenience alias: accept any float32 array, force C-contiguous copy if needed */
using f32arr = py::array_t<float, py::array::c_style | py::array::forcecast>;


/* ============================================================
 * CudaKRRState — Python-visible class
 * ============================================================ */

class CudaKRRState {
    KrrInferenceState st_;
    bool  initialized_ = false;
    int   N_train_  = 0;
    int   M_        = 0;
    int   D_        = 0;
    float sigma_    = 0.0f;

    void free_state() {
        if (initialized_) {
            cuda_krr_free(&st_);
            initialized_ = false;
        }
    }

public:
    CudaKRRState() {
        std::memset(&st_, 0, sizeof(st_));
    }

    ~CudaKRRState() {
        free_state();
    }

    /* Disable copy — GPU memory is not copyable */
    CudaKRRState(const CudaKRRState &) = delete;
    CudaKRRState &operator=(const CudaKRRState &) = delete;

    /* ------------------------------------------------------------------
     * train_ef
     *
     * Parameters
     * ----------
     * X     : (N, M)   float32 C-contiguous
     * dXT   : (N*D, M) float32 C-contiguous  [dX (N,D,M) reshaped to (N*D,M)]
     * E     : (N,)     float32
     * F     : (N*D,)   float32   physical forces F = -dE/dR
     * sigma : float
     * lam   : float   L2 regularisation
     *
     * Returns
     * -------
     * alpha : (N*(1+D),) float32 ndarray
     * ------------------------------------------------------------------ */
    py::array_t<float> train_ef(
        f32arr X_in, f32arr dXT_in, f32arr E_in, f32arr F_in,
        float sigma, float lam)
    {
        auto X_buf   = X_in.request();
        auto dXT_buf = dXT_in.request();
        auto E_buf   = E_in.request();
        auto F_buf   = F_in.request();

        if (X_buf.ndim != 2)
            throw std::invalid_argument("X must be 2-D (N, M)");
        if (dXT_buf.ndim != 2)
            throw std::invalid_argument("dXT must be 2-D (N*D, M)");
        if (E_buf.ndim != 1)
            throw std::invalid_argument("E must be 1-D (N,)");
        if (F_buf.ndim != 1)
            throw std::invalid_argument("F must be 1-D (N*D,)");

        int N = static_cast<int>(X_buf.shape[0]);
        int M = static_cast<int>(X_buf.shape[1]);
        int ND = static_cast<int>(dXT_buf.shape[0]);
        if (ND % N != 0)
            throw std::invalid_argument("dXT.shape[0] must be divisible by N");
        int D = ND / N;

        if (static_cast<int>(E_buf.shape[0]) != N)
            throw std::invalid_argument("E length must match X.shape[0]");
        if (static_cast<int>(F_buf.shape[0]) != ND)
            throw std::invalid_argument("F length must be N*D");

        /* Allocate output buffer for alpha */
        long long full = (long long)N * (1 + D);
        py::array_t<float> alpha({full});

        free_state();
        std::memset(&st_, 0, sizeof(st_));

        cuda_krr_train_ef(
            &st_,
            static_cast<const float*>(X_buf.ptr),
            static_cast<const float*>(dXT_buf.ptr),
            static_cast<const float*>(E_buf.ptr),
            static_cast<const float*>(F_buf.ptr),
            static_cast<float*>(alpha.request().ptr),
            sigma, lam, N, M, D);

        N_train_     = N;
        M_           = M;
        D_           = D;
        sigma_       = sigma;
        initialized_ = true;
        return alpha;
    }

    /* ------------------------------------------------------------------
     * predict
     *
     * Parameters
     * ----------
     * X_te   : (N_test, M)   float32 C-contiguous
     * dXT_te : (N_test*D, M) float32 C-contiguous
     *
     * Returns
     * -------
     * (E_pred, F_pred) : tuple of float32 ndarrays
     *   E_pred : (N_test,)
     *   F_pred : (N_test*D,)  — physical forces F = -dE/dR
     * ------------------------------------------------------------------ */
    std::tuple<py::array_t<float>, py::array_t<float>>
    predict(f32arr X_te_in, f32arr dXT_te_in)
    {
        if (!initialized_)
            throw std::runtime_error(
                "CudaKRRState.predict(): model not trained. Call train_ef() first.");

        auto X_buf   = X_te_in.request();
        auto dXT_buf = dXT_te_in.request();

        if (X_buf.ndim != 2)
            throw std::invalid_argument("X_te must be 2-D (N_test, M)");
        if (dXT_buf.ndim != 2)
            throw std::invalid_argument("dXT_te must be 2-D (N_test*D, M)");

        int N_test = static_cast<int>(X_buf.shape[0]);
        int M_te   = static_cast<int>(X_buf.shape[1]);
        if (M_te != M_)
            throw std::invalid_argument(
                "X_te.shape[1] does not match training M");

        int ND_te = static_cast<int>(dXT_buf.shape[0]);
        if (ND_te != N_test * D_)
            throw std::invalid_argument(
                "dXT_te.shape[0] must be N_test * D (= N_test * " +
                std::to_string(D_) + ")");

        py::array_t<float> E_pred({N_test});
        py::array_t<float> F_pred({(long long)N_test * D_});

        cuda_krr_predict(
            &st_,
            static_cast<const float*>(X_buf.ptr),
            static_cast<const float*>(dXT_buf.ptr),
            static_cast<float*>(E_pred.request().ptr),
            static_cast<float*>(F_pred.request().ptr),
            N_test);

        return {E_pred, F_pred};
    }

    /* ------------------------------------------------------------------
     * get_state
     *
     * Returns six float32 arrays needed to reconstruct the model without
     * re-training (used in model.save()).
     *
     * Returns
     * -------
     * (X_train, W_F_bar, W_combined, W_F_self, alpha_E, norms_tr)
     *   X_train    : (N, M)
     *   W_F_bar    : (N, M)
     *   W_combined : (N, M)
     *   W_F_self   : (N,)
     *   alpha_E    : (N,)
     *   norms_tr   : (N,)
     * ------------------------------------------------------------------ */
    py::tuple get_state()
    {
        if (!initialized_)
            throw std::runtime_error(
                "CudaKRRState.get_state(): model not trained.");

        long long NM = (long long)N_train_ * M_;
        py::array_t<float> X_train({N_train_, M_});
        py::array_t<float> W_F_bar({N_train_, M_});
        py::array_t<float> W_combined({N_train_, M_});
        py::array_t<float> W_F_self({N_train_});
        py::array_t<float> alpha_E({N_train_});
        py::array_t<float> norms_tr({N_train_});

        cuda_krr_get_state(
            &st_,
            static_cast<float*>(X_train.request().ptr),
            static_cast<float*>(W_F_bar.request().ptr),
            static_cast<float*>(W_combined.request().ptr),
            static_cast<float*>(W_F_self.request().ptr),
            static_cast<float*>(alpha_E.request().ptr),
            static_cast<float*>(norms_tr.request().ptr));

        return py::make_tuple(
            X_train, W_F_bar, W_combined, W_F_self, alpha_E, norms_tr);
    }

    /* ------------------------------------------------------------------
     * load_state
     *
     * Reconstruct the inference state from saved arrays (no dXT needed).
     *
     * Parameters
     * ----------
     * X_train    : (N, M)  float32
     * W_F_bar    : (N, M)  float32
     * W_combined : (N, M)  float32
     * W_F_self   : (N,)    float32
     * alpha_E    : (N,)    float32
     * norms_tr   : (N,)    float32
     * sigma      : float
     * N_train    : int
     * M          : int
     * D          : int  (= 3 * n_atoms)
     * ------------------------------------------------------------------ */
    void load_state(
        f32arr X_train_in, f32arr W_F_bar_in, f32arr W_combined_in,
        f32arr W_F_self_in, f32arr alpha_E_in, f32arr norms_tr_in,
        float sigma, int N_train, int M, int D)
    {
        auto xt = X_train_in.request();
        auto wf = W_F_bar_in.request();
        auto wc = W_combined_in.request();
        auto ws = W_F_self_in.request();
        auto ae = alpha_E_in.request();
        auto nt = norms_tr_in.request();

        free_state();
        std::memset(&st_, 0, sizeof(st_));

        cuda_krr_load_state(
            &st_,
            static_cast<const float*>(xt.ptr),
            static_cast<const float*>(wf.ptr),
            static_cast<const float*>(wc.ptr),
            static_cast<const float*>(ws.ptr),
            static_cast<const float*>(ae.ptr),
            static_cast<const float*>(nt.ptr),
            sigma, N_train, M, D);

        N_train_     = N_train;
        M_           = M;
        D_           = D;
        sigma_       = sigma;
        initialized_ = true;
    }

    /* Read-only properties */
    int   n_train()  const { return N_train_; }
    int   m()        const { return M_; }
    int   d()        const { return D_; }
    float sigma()    const { return sigma_; }
    bool  is_ready() const { return initialized_; }
};


/* ============================================================
 * build_kernel_matrix  [diagnostic binding]
 *
 * Exposes cuda_krr_build_kernel_matrix to Python.
 * Returns the full symmetric K_full as a float32 numpy array of
 * shape (full_rows, full_rows) where full_rows = N*(1+D).
 *
 * Note on layout: cuBLAS stores K_full column-major internally.
 * After mirroring the lower triangle to upper, the matrix is
 * fully symmetric (K == K^T), so reshaping the flat col-major
 * buffer as a C-order (row-major) numpy array also gives K.
 * ============================================================ */
static py::array_t<float> build_kernel_matrix_py(
    f32arr X_in, f32arr dXT_in, float sigma)
{
    auto X_buf   = X_in.request();
    auto dXT_buf = dXT_in.request();

    if (X_buf.ndim != 2)
        throw std::invalid_argument("X must be 2-D (N, M)");
    if (dXT_buf.ndim != 2)
        throw std::invalid_argument("dXT must be 2-D (N*D, M)");

    int N  = static_cast<int>(X_buf.shape[0]);
    int M  = static_cast<int>(X_buf.shape[1]);
    int ND = static_cast<int>(dXT_buf.shape[0]);
    if (ND % N != 0)
        throw std::invalid_argument("dXT.shape[0] must be divisible by N");
    int D = ND / N;

    long long full_rows = (long long)N * (1 + D);

    py::array_t<float> K_full({full_rows, full_rows});

    cuda_krr_build_kernel_matrix(
        static_cast<const float*>(X_buf.ptr),
        static_cast<const float*>(dXT_buf.ptr),
        static_cast<float*>(K_full.request().ptr),
        sigma, N, M, D);

    return K_full;
}


/* ============================================================
 * Module
 * ============================================================ */

PYBIND11_MODULE(cuda_krr_ext, mod)
{
    mod.doc() = R"doc(
CUDA-accelerated KRR with joint energy + force training.

Exposes a single class ``CudaKRRState`` that wraps the persistent GPU
inference state.  Train once with ``train_ef()``, then call ``predict()``
repeatedly without re-uploading training data.
)doc";

    py::class_<CudaKRRState>(mod, "CudaKRRState",
        R"doc(
Persistent GPU state for Gaussian KRR with E+F training.

After ``train_ef()`` the training Jacobians are contracted into
``W_F_bar`` and freed; only (3 * N_train * M + 3 * N_train) floats
remain on GPU.  Repeated ``predict()`` calls reuse this persistent state.
)doc")
        .def(py::init<>())

        .def("train_ef", &CudaKRRState::train_ef,
             py::arg("X"), py::arg("dXT"), py::arg("E"), py::arg("F"),
             py::arg("sigma"), py::arg("lam"),
             R"doc(
Train the KRR model and initialise the GPU inference state.

Parameters
----------
X : ndarray, shape (N, M), float32, C-contiguous
    Training descriptors.
dXT : ndarray, shape (N*D, M), float32, C-contiguous
    Flattened Jacobians — dX (N, D, M) reshaped to (N*D, M).
E : ndarray, shape (N,), float32
    Training energies.
F : ndarray, shape (N*D,), float32
    Training physical forces F = -dE/dR, flattened.
sigma : float
    Gaussian kernel length-scale.
lam : float
    L2 regularisation strength (added to kernel diagonal).

Returns
-------
alpha : ndarray, shape (N*(1+D),), float32
    Trained dual coefficients.  alpha[:N] are the energy weights;
    alpha[N:] are the force weights.
)doc")

        .def("predict", &CudaKRRState::predict,
             py::arg("X_te"), py::arg("dXT_te"),
             R"doc(
Predict energies and forces for a batch of test molecules.

Parameters
----------
X_te : ndarray, shape (N_test, M), float32, C-contiguous
dXT_te : ndarray, shape (N_test*D, M), float32, C-contiguous
    Flattened test Jacobians.

Returns
-------
E_pred : ndarray, shape (N_test,), float32
F_pred : ndarray, shape (N_test*D,), float32  — physical forces F = -dE/dR
)doc")

        .def("get_state", &CudaKRRState::get_state,
             R"doc(
Extract the six precomputed GPU arrays needed for serialisation.

Returns
-------
(X_train, W_F_bar, W_combined, W_F_self, alpha_E, norms_tr)
All are float32 ndarrays on host.
)doc")

        .def("load_state", &CudaKRRState::load_state,
             py::arg("X_train"), py::arg("W_F_bar"), py::arg("W_combined"),
             py::arg("W_F_self"), py::arg("alpha_E"), py::arg("norms_tr"),
             py::arg("sigma"), py::arg("N_train"), py::arg("M"), py::arg("D"),
             R"doc(
Reconstruct the GPU inference state from saved arrays.

Called by model.load() after reading an .npz file.  No dXT data needed.
)doc")

        .def_property_readonly("n_train",  &CudaKRRState::n_train,
                               "Number of training molecules.")
        .def_property_readonly("m",        &CudaKRRState::m,
                               "Descriptor dimension M = N*(N-1)/2.")
        .def_property_readonly("d",        &CudaKRRState::d,
                               "Degrees of freedom per molecule D = 3*n_atoms.")
        .def_property_readonly("sigma",    &CudaKRRState::sigma,
                               "Gaussian kernel length-scale.")
        .def_property_readonly("is_ready", &CudaKRRState::is_ready,
                               "True after train_ef() or load_state().");

    mod.def("build_kernel_matrix", &build_kernel_matrix_py,
            py::arg("X"), py::arg("dXT"), py::arg("sigma"),
            R"doc(
[DIAGNOSTIC] Build the full symmetric K_full training kernel matrix on GPU.

Returns the matrix as a (N*(1+D), N*(1+D)) float32 numpy array.  The lower
triangle is filled by the standard GPU assembly kernel; the upper triangle is
mirrored from it before returning, making the result fully symmetric.

Intended for element-by-element comparison against the CPU reference
``global_kernels.kernel_gaussian_full_symm``.

Parameters
----------
X   : ndarray (N, M) float32
dXT : ndarray (N*D, M) float32
sigma : float

Returns
-------
K_full : ndarray (N*(1+D), N*(1+D)) float32
)doc");
}
