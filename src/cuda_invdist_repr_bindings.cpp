// cuda_invdist_repr_bindings.cpp — pybind11 bindings for GPU invdist representation

#include <pybind11/pybind11.h>

#include "cuda_invdist_repr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_invdist_repr, m) {
    m.doc() = "GPU batched inverse-distance representation and Jacobian (FP32).";

    m.def(
        "inverse_distance_upper",
        &kf::invdist_cuda::inverse_distance_upper_cuda,
        py::arg("coords"),
        py::arg("n_atoms"),
        py::arg("eps") = 1e-6f,
        "Compute batched inverse-distance representation on GPU.\n\n"
        "Parameters\n"
        "----------\n"
        "coords : torch.Tensor, shape (nm, n_atoms, 3), float32, CUDA\n"
        "    Batched Cartesian coordinates.  All molecules must have the same n_atoms.\n"
        "n_atoms : int\n"
        "    Number of atoms per molecule.\n"
        "eps : float, default 1e-6\n"
        "    Distance floor: r is clamped to max(r, eps) before inversion.\n\n"
        "Returns\n"
        "-------\n"
        "X : torch.Tensor, shape (nm, M), float32, CUDA\n"
        "    M = n_atoms*(n_atoms-1)//2.  Entry [m, p] = 1/r_{ij} for pair p=(i<j)."
    );

    m.def(
        "inverse_distance_upper_and_jacobian",
        &kf::invdist_cuda::inverse_distance_upper_and_jacobian_cuda,
        py::arg("coords"),
        py::arg("n_atoms"),
        py::arg("eps") = 1e-6f,
        "Compute batched inverse-distance representation and Jacobian on GPU.\n\n"
        "Parameters\n"
        "----------\n"
        "coords : torch.Tensor, shape (nm, n_atoms, 3), float32, CUDA\n"
        "n_atoms : int\n"
        "eps : float, default 1e-6\n\n"
        "Returns\n"
        "-------\n"
        "(X, dX) : tuple of torch.Tensor\n"
        "    X  : (nm, M),    float32, CUDA\n"
        "    dX : (nm, D, M), float32, CUDA    D = 3*n_atoms\n"
        "    dX[m, 3*a+d, p] = d(1/r_ij)/d(R_{a,d}) for pair p=(i<j), atom a, dim d."
    );
}
