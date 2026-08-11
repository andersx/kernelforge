#include <pybind11/pybind11.h>

#include "cuda_fchl18_repr.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_fchl18_repr, m) {
    m.doc() = "GPU FCHL18 representation generation (fp32 / fp64).";

    m.def(
        "generate",
        &kf::fchl18::generate_fchl18_cuda,
        py::arg("coords"),
        py::arg("Z"),
        py::arg("N"),
        py::arg("cut_distance") = 5.0,
        R"doc(
Generate FCHL18 representations on the GPU.

Parameters
----------
coords : torch.Tensor, shape (nm, max_size, 3), float32 or float64, CUDA
    Padded Cartesian coordinates.
Z : torch.Tensor, shape (nm, max_size), int32, CUDA
    Padded nuclear charges.
N : torch.Tensor, shape (nm,), int32, CUDA
    Active atom counts per molecule.
cut_distance : float, default 5.0
    Neighbour cutoff radius in Angstrom.

Returns
-------
x : torch.Tensor, shape (nm, max_size, 5, max_size), same dtype as coords, CUDA
    Representation. Layout per atom: [distances, Z-neighbours, dx, dy, dz].
    Unused slots are filled with 1e100 (fp64) or 1e30 (fp32).
nn : torch.Tensor, shape (nm, max_size), int32, CUDA
    Number of neighbours within cut_distance per atom (self at index 0).
)doc"
    );
}
