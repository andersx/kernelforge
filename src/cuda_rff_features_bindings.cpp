// cuda_rff_features_bindings.cpp — PyTorch-extension bindings for CUDA RFF features

#include <pybind11/pybind11.h>

#include "cuda_rff_features.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuda_rff_features, m) {
    m.doc() = "CUDA Random Fourier Features for global descriptors (FP32).";

    m.def(
        "rff_features",
        &kf::rff_cuda::rff_features_cuda,
        py::arg("X"),
        py::arg("W"),
        py::arg("b"),
        "Compute global RFF features Z = sqrt(2/D) * cos(X @ W + b) on GPU."
    );

    m.def(
        "rff_gramian_symm_rfp",
        &kf::rff_cuda::rff_gramian_symm_rfp_cuda,
        py::arg("X"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("chunk_size") = 256,
        "Compute RFP-packed Z.T @ Z and Z.T @ Y for energy-only RFF training."
    );

    m.def(
        "rff_full_gramian_symm_rfp",
        &kf::rff_cuda::rff_full_gramian_symm_rfp_cuda,
        py::arg("X"),
        py::arg("dX"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("F"),
        py::arg("energy_chunk") = 256,
        py::arg("force_chunk") = 64,
        "Compute RFP-packed full energy+force normal equations for global RFF training."
    );

    m.def(
        "rff_predict_energy",
        &kf::rff_cuda::rff_predict_energy_cuda,
        py::arg("X"),
        py::arg("W"),
        py::arg("b"),
        py::arg("weights"),
        py::arg("chunk_size") = 256,
        "Compute energy predictions Z @ weights in chunks on GPU."
    );

    m.def(
        "rff_predict_force",
        &kf::rff_cuda::rff_predict_force_cuda,
        py::arg("X"),
        py::arg("dX"),
        py::arg("W"),
        py::arg("b"),
        py::arg("weights"),
        py::arg("chunk_size") = 64,
        "Compute force predictions G @ weights in chunks on GPU."
    );

    m.def(
        "rff_features_elemental",
        &kf::rff_cuda::rff_features_elemental_cuda,
        py::arg("X"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        "Compute local elemental RFF features on GPU. Returns (nmol, D) row-major."
    );

    m.def(
        "rff_features_elemental_col_major",
        &kf::rff_cuda::rff_features_elemental_col_major_cuda,
        py::arg("X"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        "Compute local elemental RFF features on GPU. Returns (D, nmol) col-major."
    );

    m.def(
        "rff_gradient_elemental",
        &kf::rff_cuda::rff_gradient_elemental_cuda,
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        py::arg("chunk_size") = 64,
        "Compute local elemental RFF gradient feature matrix G (total_naq, D) on GPU."
    );

    m.def(
        "rff_gradient_elemental_col_major",
        &kf::rff_cuda::rff_gradient_elemental_col_major_cuda,
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        py::arg("chunk_size") = 64,
        "Compute local elemental RFF gradient feature matrix G (D, total_naq) col-major on GPU."
    );

    m.def(
        "rff_gramian_elemental_rfp",
        &kf::rff_cuda::rff_gramian_elemental_rfp_cuda,
        py::arg("X"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("chunk_size") = 256,
        "Compute RFP-packed local elemental RFF energy-only normal equations."
    );

    m.def(
        "rff_predict_energy_elemental",
        &kf::rff_cuda::rff_predict_energy_elemental_cuda,
        py::arg("X"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        py::arg("weights"),
        py::arg("chunk_size") = 256,
        "Compute local elemental RFF energy predictions on GPU."
    );

    m.def(
        "rff_full_gramian_elemental_rfp",
        &kf::rff_cuda::rff_full_gramian_elemental_rfp_cuda,
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        py::arg("Y"),
        py::arg("F"),
        py::arg("energy_chunk") = 256,
        py::arg("force_chunk") = 64,
        "Compute RFP-packed local elemental RFF energy+force normal equations."
    );

    m.def(
        "rff_predict_force_elemental",
        &kf::rff_cuda::rff_predict_force_elemental_cuda,
        py::arg("X"),
        py::arg("dX"),
        py::arg("Q"),
        py::arg("N"),
        py::arg("W"),
        py::arg("b"),
        py::arg("weights"),
        py::arg("chunk_size") = 64,
        "Compute local elemental RFF force predictions on GPU."
    );
}
