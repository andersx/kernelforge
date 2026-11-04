#include <torch/extension.h>

#include "cuda_fchl18_direct.hpp"

namespace py = pybind11;

namespace {

void check_cuda_f32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_i32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor kernel_gaussian(
    torch::Tensor coords1,
    torch::Tensor Z1,
    torch::Tensor N1,
    torch::Tensor coords2,
    torch::Tensor Z2,
    torch::Tensor N2,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    int fourier_order,
    bool use_atm,
    bool use_alchemy,
    double alchemy_period_width,
    double alchemy_group_width
) {
    check_cuda_f32(coords1, "coords1");
    check_cuda_f32(coords2, "coords2");
    check_cuda_i32(Z1, "Z1");
    check_cuda_i32(Z2, "Z2");
    check_cuda_i32(N1, "N1");
    check_cuda_i32(N2, "N2");

    TORCH_CHECK(coords1.dim() == 3 && coords1.size(2) == 3, "coords1 must be (nm, max_atoms, 3)");
    TORCH_CHECK(coords2.dim() == 3 && coords2.size(2) == 3, "coords2 must be (nm, max_atoms, 3)");
    TORCH_CHECK(Z1.dim() == 2 && Z2.dim() == 2, "Z must be (nm, max_atoms)");
    TORCH_CHECK(N1.dim() == 1 && N2.dim() == 1, "N must be (nm,)");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(fourier_order >= 1 && fourier_order <= 16, "fourier_order out of range");

    const int nm1 = static_cast<int>(coords1.size(0));
    const int max1 = static_cast<int>(coords1.size(1));
    const int nm2 = static_cast<int>(coords2.size(0));
    const int max2 = static_cast<int>(coords2.size(1));

    TORCH_CHECK(Z1.size(0) == nm1 && Z1.size(1) == max1, "Z1 shape mismatch");
    TORCH_CHECK(Z2.size(0) == nm2 && Z2.size(1) == max2, "Z2 shape mismatch");
    TORCH_CHECK(N1.size(0) == nm1 && N2.size(0) == nm2, "N shape mismatch");

    auto K = torch::empty({nm1, nm2}, coords1.options());
    kf::fchl18::direct::kernel_gaussian_rect_fp32(
        coords1.data_ptr<float>(),
        Z1.data_ptr<int>(),
        N1.data_ptr<int>(),
        coords2.data_ptr<float>(),
        Z2.data_ptr<int>(),
        N2.data_ptr<int>(),
        K.data_ptr<float>(),
        static_cast<float>(sigma),
        nm1,
        nm2,
        max1,
        max2,
        static_cast<float>(two_body_scaling),
        static_cast<float>(two_body_width),
        static_cast<float>(two_body_power),
        static_cast<float>(three_body_scaling),
        static_cast<float>(three_body_width),
        static_cast<float>(three_body_power),
        fourier_order,
        use_atm,
        use_alchemy,
        static_cast<float>(alchemy_period_width),
        static_cast<float>(alchemy_group_width)
    );
    return K;
}

}  // namespace

PYBIND11_MODULE(cuda_fchl18_direct, m) {
    m.doc() = "Coords-only FCHL18 Gaussian kernel (FP32, no cutoff, molecules)";
    m.def(
        "kernel_gaussian",
        &kernel_gaussian,
        py::arg("coords1"),
        py::arg("Z1"),
        py::arg("N1"),
        py::arg("coords2"),
        py::arg("Z2"),
        py::arg("N2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = true,
        py::arg("use_alchemy") = false,
        py::arg("alchemy_period_width") = 1.6,
        py::arg("alchemy_group_width") = 1.6,
        "Rectangular FCHL18 Gaussian from coords/Z/N (FP32, no cutoff)."
    );
}
