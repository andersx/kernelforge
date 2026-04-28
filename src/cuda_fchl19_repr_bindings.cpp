// cuda_fchl19_repr_bindings.cpp — PyTorch-extension bindings for the GPU FCHL19
// forward representation.
//
// Exposed Python function (module: cuda_fchl19_repr):
//
//   generate_fchl_acsf(coords, Q, N, nelements, nRs2, nRs3, nFourier,
//                      eta2, eta3, zeta, rcut, acut,
//                      two_body_decay, three_body_decay, three_body_weight)
//     coords : (nm, max_atoms, 3)  float32 CUDA
//     Q      : (nm, max_atoms)     int32   CUDA   element indices [0..nelements)
//     N      : (nm,)               int32   CUDA   active atom counts
//     → rep  : (nm, max_atoms, rep_size) float32 CUDA
//
// The three_body_weight parameter is normalised internally (identical to the
// CPU fchl19_repr module convention):
//   norm = sqrt(eta3 / pi) * three_body_weight

#include <cmath>

#include <torch/extension.h>

#include "cuda_fchl19_repr.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Python wrapper — normalises three_body_weight before calling the CUDA driver
// ---------------------------------------------------------------------------

static torch::Tensor generate_fchl_acsf_py(
    torch::Tensor coords,        // (nm, max_atoms, 3) float32 CUDA
    torch::Tensor Q,             // (nm, max_atoms)    int32   CUDA
    torch::Tensor N,             // (nm,)              int32   CUDA
    int nelements,
    int nRs2,
    int nRs3,
    int nFourier,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight
) {
    // Same normalisation applied by the CPU fchl19_repr bindings
    const float w3_norm = (float)(std::sqrt(eta3 / M_PI) * three_body_weight);

    return kf::fchl19::generate_fchl_acsf_cuda(
        coords,
        Q,
        N,
        nelements,
        nRs2,
        nRs3,
        nFourier,
        (float)eta2,
        (float)eta3,
        (float)zeta,
        (float)rcut,
        (float)acut,
        (float)two_body_decay,
        (float)three_body_decay,
        w3_norm
    );
}

static py::tuple generate_fchl_acsf_and_gradients_py(
    torch::Tensor coords,
    torch::Tensor Q,
    torch::Tensor N,
    int nelements,
    int nRs2,
    int nRs3,
    int nFourier,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight
) {
    const float w3_norm = (float)(std::sqrt(eta3 / M_PI) * three_body_weight);

    auto [rep, grad] = kf::fchl19::generate_fchl_acsf_and_gradients_cuda(
        coords,
        Q,
        N,
        nelements,
        nRs2,
        nRs3,
        nFourier,
        (float)eta2,
        (float)eta3,
        (float)zeta,
        (float)rcut,
        (float)acut,
        (float)two_body_decay,
        (float)three_body_decay,
        w3_norm
    );
    return py::make_tuple(rep, grad);
}


// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(cuda_fchl19_repr, m)
{
    m.doc() = R"doc(
GPU-accelerated FCHL19 ACSF representation generator (FP32).

Computes the FCHL19 representation for a batch of molecules in parallel on
the GPU.  The input tensors must already reside on a CUDA device.

The convention for Q (element indices), basis construction (Rs2/Rs3/Ts), and
three_body_weight normalisation is identical to the CPU ``fchl19_repr`` module.
)doc";

    m.def(
        "generate_fchl_acsf",
        &generate_fchl_acsf_py,
        py::arg("coords"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("nelements"),
        py::arg("nRs2") = 24,
        py::arg("nRs3") = 20,
        py::arg("nFourier") = 1,
        py::arg("eta2") = 0.32,
        py::arg("eta3") = 2.7,
        py::arg("zeta") = M_PI,
        py::arg("rcut") = 8.0,
        py::arg("acut") = 8.0,
        py::arg("two_body_decay") = 1.8,
        py::arg("three_body_decay") = 0.57,
        py::arg("three_body_weight") = 13.4,
        R"doc(
Generate FCHL19 ACSF representations on the GPU (FP32).

Parameters
----------
coords : torch.Tensor, shape (nm, max_atoms, 3), float32, CUDA
    Batched padded Cartesian coordinates.
Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
    Element indices in [0, nelements).  Padded slots may hold any value.
N : torch.Tensor, shape (nm,), int32, CUDA
    Number of active atoms per molecule.
nelements : int
    Number of distinct element types.
nRs2 : int, default 24
    Number of two-body radial basis functions.
nRs3 : int, default 20
    Number of three-body radial basis functions.
nFourier : int, default 1
    Number of Fourier (angular) harmonics.  nabasis = 2*nFourier.
eta2 : float, default 0.32
    Two-body lognormal Gaussian width parameter.
eta3 : float, default 2.7
    Three-body Gaussian width parameter.
zeta : float, default pi
    Angular decay parameter.
rcut : float, default 8.0
    Two-body cutoff radius (Angstrom).
acut : float, default 8.0
    Three-body cutoff radius (Angstrom).
two_body_decay : float, default 1.8
    Power-law decay exponent for two-body prefactor.
three_body_decay : float, default 0.57
    Power-law decay exponent for three-body ATM denominator.
three_body_weight : float, default 13.4
    Three-body weight; rescaled internally by sqrt(eta3/pi).

Returns
-------
rep : torch.Tensor, shape (nm, max_atoms, rep_size), float32, CUDA
    Per-atom FCHL19 representations.  Padded slots (i >= N[m]) are zeroed.
    rep_size = nelements*nRs2 + nelements*(nelements+1)//2 * nRs3 * 2*nFourier
)doc"
    );

    m.def(
        "generate_fchl_acsf_and_gradients",
        &generate_fchl_acsf_and_gradients_py,
        py::arg("coords"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("nelements"),
        py::arg("nRs2") = 24,
        py::arg("nRs3") = 20,
        py::arg("nFourier") = 1,
        py::arg("eta2") = 0.32,
        py::arg("eta3") = 2.7,
        py::arg("zeta") = M_PI,
        py::arg("rcut") = 8.0,
        py::arg("acut") = 8.0,
        py::arg("two_body_decay") = 1.8,
        py::arg("three_body_decay") = 0.57,
        py::arg("three_body_weight") = 13.4,
        R"doc(
Generate FCHL19 ACSF representations and coordinate Jacobians on the GPU (FP32).

Returns
-------
(rep, grad)
    rep : torch.Tensor, shape (nm, max_atoms, rep_size), float32, CUDA
    grad : torch.Tensor, shape (nm, max_atoms, rep_size, max_atoms, 3), float32, CUDA
)doc"
    );
}
