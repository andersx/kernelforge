#include <torch/extension.h>

#include "cuda_fchl18_kernel.hpp"

namespace py = pybind11;

namespace {

void check_cuda_int32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_floating(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kFloat64,
        name,
        " must be float32 or float64"
    );
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor kernel_gaussian(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor n1,
    torch::Tensor n2,
    torch::Tensor nn1,
    torch::Tensor nn2,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x1, "X1");
    check_cuda_floating(x2, "X2");
    check_cuda_int32(n1, "N1");
    check_cuda_int32(n2, "N2");
    check_cuda_int32(nn1, "NN1");
    check_cuda_int32(nn2, "NN2");

    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "X1 and X2 dtype must match");
    TORCH_CHECK(x1.dim() == 4, "X1 must be 4-D");
    TORCH_CHECK(x2.dim() == 4, "X2 must be 4-D");
    TORCH_CHECK(x1.size(2) == 5, "X1.size(2) must equal 5");
    TORCH_CHECK(x2.size(2) == 5, "X2.size(2) must equal 5");
    TORCH_CHECK(n1.dim() == 1, "N1 must be 1-D");
    TORCH_CHECK(n2.dim() == 1, "N2 must be 1-D");
    TORCH_CHECK(nn1.dim() == 2, "NN1 must be 2-D");
    TORCH_CHECK(nn2.dim() == 2, "NN2 must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(cut_distance > 0.0, "cut_distance must be positive");

    const int nm1 = static_cast<int>(x1.size(0));
    const int max_size1 = static_cast<int>(x1.size(1));
    const int nm2 = static_cast<int>(x2.size(0));
    const int max_size2 = static_cast<int>(x2.size(1));

    TORCH_CHECK(x1.size(3) == max_size1, "X1 shape must be (nm1, max_size1, 5, max_size1)");
    TORCH_CHECK(x2.size(3) == max_size2, "X2 shape must be (nm2, max_size2, 5, max_size2)");
    TORCH_CHECK(n1.size(0) == nm1, "N1.size(0) must equal X1.size(0)");
    TORCH_CHECK(n2.size(0) == nm2, "N2.size(0) must equal X2.size(0)");
    TORCH_CHECK(nn1.size(0) == nm1 && nn1.size(1) == max_size1, "NN1 shape mismatch");
    TORCH_CHECK(nn2.size(0) == nm2 && nn2.size(1) == max_size2, "NN2 shape mismatch");

    auto K = torch::empty({nm1, nm2}, x1.options());
    if (x1.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_rect_cu(
            x1.data_ptr<float>(),
            x2.data_ptr<float>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            K.data_ptr<float>(),
            static_cast<float>(sigma),
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_rect_cu(
            x1.data_ptr<double>(),
            x2.data_ptr<double>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            K.data_ptr<double>(),
            sigma,
            nm1,
            nm2,
            max_size1,
            max_size2,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return K;
}

torch::Tensor kernel_gaussian_symm(
    torch::Tensor x,
    torch::Tensor n,
    torch::Tensor nn,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x, "X");
    check_cuda_int32(n, "N");
    check_cuda_int32(nn, "NN");

    TORCH_CHECK(x.dim() == 4, "X must be 4-D");
    TORCH_CHECK(x.size(2) == 5, "X.size(2) must equal 5");
    TORCH_CHECK(n.dim() == 1, "N must be 1-D");
    TORCH_CHECK(nn.dim() == 2, "NN must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(cut_distance > 0.0, "cut_distance must be positive");

    const int nm = static_cast<int>(x.size(0));
    const int max_size = static_cast<int>(x.size(1));

    TORCH_CHECK(x.size(3) == max_size, "X shape must be (nm, max_size, 5, max_size)");
    TORCH_CHECK(n.size(0) == nm, "N.size(0) must equal X.size(0)");
    TORCH_CHECK(nn.size(0) == nm && nn.size(1) == max_size, "NN shape mismatch");

    auto K = torch::empty({nm, nm}, x.options());
    if (x.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_symm_cu(
            x.data_ptr<float>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            K.data_ptr<float>(),
            static_cast<float>(sigma),
            nm,
            max_size,
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_symm_cu(
            x.data_ptr<double>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            K.data_ptr<double>(),
            sigma,
            nm,
            max_size,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return K;
}

torch::Tensor kernel_gaussian_symm_rfp(
    torch::Tensor x,
    torch::Tensor n,
    torch::Tensor nn,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x, "X");
    check_cuda_int32(n, "N");
    check_cuda_int32(nn, "NN");

    TORCH_CHECK(x.dim() == 4, "X must be 4-D");
    TORCH_CHECK(x.size(2) == 5, "X.size(2) must equal 5");
    TORCH_CHECK(n.dim() == 1, "N must be 1-D");
    TORCH_CHECK(nn.dim() == 2, "NN must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(cut_distance > 0.0, "cut_distance must be positive");

    const int nm = static_cast<int>(x.size(0));
    const int max_size = static_cast<int>(x.size(1));
    const long long n_rfp = static_cast<long long>(nm) * (nm + 1) / 2;

    TORCH_CHECK(x.size(3) == max_size, "X shape must be (nm, max_size, 5, max_size)");
    TORCH_CHECK(n.size(0) == nm, "N.size(0) must equal X.size(0)");
    TORCH_CHECK(nn.size(0) == nm && nn.size(1) == max_size, "NN shape mismatch");

    auto K_rfp = torch::empty({n_rfp}, x.options());
    if (x.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_symm_rfp_cu(
            x.data_ptr<float>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            K_rfp.data_ptr<float>(),
            static_cast<float>(sigma),
            nm,
            max_size,
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_symm_rfp_cu(
            x.data_ptr<double>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            K_rfp.data_ptr<double>(),
            sigma,
            nm,
            max_size,
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return K_rfp;
}

torch::Tensor kernel_gaussian_jacobian(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor n1,
    torch::Tensor n2,
    torch::Tensor nn1,
    torch::Tensor nn2,
    torch::Tensor coords1,
    torch::Tensor z1,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x1, "X1");
    check_cuda_floating(x2, "X2");
    check_cuda_floating(coords1, "coords1");
    check_cuda_int32(n1, "N1");
    check_cuda_int32(n2, "N2");
    check_cuda_int32(nn1, "NN1");
    check_cuda_int32(nn2, "NN2");
    check_cuda_int32(z1, "Z1");

    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "X1 and X2 dtype must match");
    TORCH_CHECK(x1.scalar_type() == coords1.scalar_type(), "X1 and coords1 dtype must match");
    TORCH_CHECK(x1.dim() == 4, "X1 must be 4-D");
    TORCH_CHECK(x2.dim() == 4, "X2 must be 4-D");
    TORCH_CHECK(x1.size(2) == 5, "X1.size(2) must equal 5");
    TORCH_CHECK(x2.size(2) == 5, "X2.size(2) must equal 5");
    TORCH_CHECK(n1.dim() == 1, "N1 must be 1-D");
    TORCH_CHECK(n2.dim() == 1, "N2 must be 1-D");
    TORCH_CHECK(nn1.dim() == 2, "NN1 must be 2-D");
    TORCH_CHECK(nn2.dim() == 2, "NN2 must be 2-D");
    TORCH_CHECK(coords1.dim() == 3, "coords1 must be 3-D");
    TORCH_CHECK(z1.dim() == 2, "Z1 must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(cut_distance > 0.0, "cut_distance must be positive");

    const int nm1 = static_cast<int>(x1.size(0));
    const int max_size1 = static_cast<int>(x1.size(1));
    const int nm2 = static_cast<int>(x2.size(0));
    const int max_size2 = static_cast<int>(x2.size(1));

    TORCH_CHECK(x1.size(3) == max_size1, "X1 shape must be (nm1, max_size1, 5, max_size1)");
    TORCH_CHECK(x2.size(3) == max_size2, "X2 shape must be (nm2, max_size2, 5, max_size2)");
    TORCH_CHECK(n1.size(0) == nm1, "N1.size(0) must equal X1.size(0)");
    TORCH_CHECK(n2.size(0) == nm2, "N2.size(0) must equal X2.size(0)");
    TORCH_CHECK(nn1.size(0) == nm1 && nn1.size(1) == max_size1, "NN1 shape mismatch");
    TORCH_CHECK(nn2.size(0) == nm2 && nn2.size(1) == max_size2, "NN2 shape mismatch");
    TORCH_CHECK(
        coords1.size(0) == nm1 && coords1.size(1) == max_size1 && coords1.size(2) == 3,
        "coords1 shape must be (nm1, max_size1, 3)"
    );
    TORCH_CHECK(z1.size(0) == nm1 && z1.size(1) == max_size1, "Z1 shape must be (nm1, max_size1)");

    const auto n1_host = n1.to(torch::kCPU);
    const int *n1_ptr = n1_host.data_ptr<int>();
    long long d_a_rows = 0;
    for (int a = 0; a < nm1; ++a) {
        TORCH_CHECK(
            n1_ptr[a] >= 0 && n1_ptr[a] <= max_size1, "N1 entries must be in [0, max_size1]"
        );
        d_a_rows += 3LL * n1_ptr[a];
    }
    TORCH_CHECK(d_a_rows > 0, "kernel_gaussian_jacobian: query set has no atoms");

    auto J = torch::empty({d_a_rows, nm2}, x1.options());
    if (x1.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_jacobian_cu(
            x1.data_ptr<float>(),
            x2.data_ptr<float>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<float>(),
            z1.data_ptr<int>(),
            J.data_ptr<float>(),
            static_cast<float>(sigma),
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_jacobian_cu(
            x1.data_ptr<double>(),
            x2.data_ptr<double>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<double>(),
            z1.data_ptr<int>(),
            J.data_ptr<double>(),
            sigma,
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return J;
}

torch::Tensor kernel_gaussian_jacobian_t(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor n1,
    torch::Tensor n2,
    torch::Tensor nn1,
    torch::Tensor nn2,
    torch::Tensor coords1,
    torch::Tensor z1,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x1, "X1");
    check_cuda_floating(x2, "X2");
    check_cuda_floating(coords1, "coords1");
    check_cuda_int32(n1, "N1");
    check_cuda_int32(n2, "N2");
    check_cuda_int32(nn1, "NN1");
    check_cuda_int32(nn2, "NN2");
    check_cuda_int32(z1, "Z1");

    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "X1 and X2 dtype must match");
    TORCH_CHECK(x1.scalar_type() == coords1.scalar_type(), "X1 and coords1 dtype must match");
    TORCH_CHECK(x1.dim() == 4, "X1 must be 4-D");
    TORCH_CHECK(x2.dim() == 4, "X2 must be 4-D");
    TORCH_CHECK(x1.size(2) == 5, "X1.size(2) must equal 5");
    TORCH_CHECK(x2.size(2) == 5, "X2.size(2) must equal 5");
    TORCH_CHECK(n1.dim() == 1, "N1 must be 1-D");
    TORCH_CHECK(n2.dim() == 1, "N2 must be 1-D");
    TORCH_CHECK(nn1.dim() == 2, "NN1 must be 2-D");
    TORCH_CHECK(nn2.dim() == 2, "NN2 must be 2-D");
    TORCH_CHECK(coords1.dim() == 3, "coords1 must be 3-D");
    TORCH_CHECK(z1.dim() == 2, "Z1 must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(cut_distance > 0.0, "cut_distance must be positive");

    const int nm1 = static_cast<int>(x1.size(0));
    const int max_size1 = static_cast<int>(x1.size(1));
    const int nm2 = static_cast<int>(x2.size(0));
    const int max_size2 = static_cast<int>(x2.size(1));

    TORCH_CHECK(x1.size(3) == max_size1, "X1 shape must be (nm1, max_size1, 5, max_size1)");
    TORCH_CHECK(x2.size(3) == max_size2, "X2 shape must be (nm2, max_size2, 5, max_size2)");
    TORCH_CHECK(n1.size(0) == nm1, "N1.size(0) must equal X1.size(0)");
    TORCH_CHECK(n2.size(0) == nm2, "N2.size(0) must equal X2.size(0)");
    TORCH_CHECK(nn1.size(0) == nm1 && nn1.size(1) == max_size1, "NN1 shape mismatch");
    TORCH_CHECK(nn2.size(0) == nm2 && nn2.size(1) == max_size2, "NN2 shape mismatch");
    TORCH_CHECK(
        coords1.size(0) == nm1 && coords1.size(1) == max_size1 && coords1.size(2) == 3,
        "coords1 shape must be (nm1, max_size1, 3)"
    );
    TORCH_CHECK(z1.size(0) == nm1 && z1.size(1) == max_size1, "Z1 shape must be (nm1, max_size1)");

    const auto n1_host = n1.to(torch::kCPU);
    const int *n1_ptr = n1_host.data_ptr<int>();
    long long d_a_rows = 0;
    for (int a = 0; a < nm1; ++a) {
        TORCH_CHECK(
            n1_ptr[a] >= 0 && n1_ptr[a] <= max_size1, "N1 entries must be in [0, max_size1]"
        );
        d_a_rows += 3LL * n1_ptr[a];
    }
    TORCH_CHECK(d_a_rows > 0, "kernel_gaussian_jacobian_t: query set has no atoms");

    auto Jt = torch::empty({nm2, d_a_rows}, x1.options());
    if (x1.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_jacobian_t_cu(
            x1.data_ptr<float>(),
            x2.data_ptr<float>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<float>(),
            z1.data_ptr<int>(),
            Jt.data_ptr<float>(),
            static_cast<float>(sigma),
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_jacobian_t_cu(
            x1.data_ptr<double>(),
            x2.data_ptr<double>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<double>(),
            z1.data_ptr<int>(),
            Jt.data_ptr<double>(),
            sigma,
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return Jt;
}

torch::Tensor kernel_gaussian_hessian(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor n1,
    torch::Tensor n2,
    torch::Tensor nn1,
    torch::Tensor nn2,
    torch::Tensor coords1,
    torch::Tensor z1,
    torch::Tensor coords2,
    torch::Tensor z2,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x1, "X1");
    check_cuda_floating(x2, "X2");
    check_cuda_floating(coords1, "coords1");
    check_cuda_floating(coords2, "coords2");
    check_cuda_int32(n1, "N1");
    check_cuda_int32(n2, "N2");
    check_cuda_int32(nn1, "NN1");
    check_cuda_int32(nn2, "NN2");
    check_cuda_int32(z1, "Z1");
    check_cuda_int32(z2, "Z2");

    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "X1 and X2 dtype must match");
    TORCH_CHECK(x1.scalar_type() == coords1.scalar_type(), "X1 and coords1 dtype must match");
    TORCH_CHECK(x1.scalar_type() == coords2.scalar_type(), "X1 and coords2 dtype must match");
    TORCH_CHECK(x1.dim() == 4, "X1 must be 4-D");
    TORCH_CHECK(x2.dim() == 4, "X2 must be 4-D");
    TORCH_CHECK(x1.size(2) == 5, "X1.size(2) must equal 5");
    TORCH_CHECK(x2.size(2) == 5, "X2.size(2) must equal 5");
    TORCH_CHECK(n1.dim() == 1, "N1 must be 1-D");
    TORCH_CHECK(n2.dim() == 1, "N2 must be 1-D");
    TORCH_CHECK(nn1.dim() == 2, "NN1 must be 2-D");
    TORCH_CHECK(nn2.dim() == 2, "NN2 must be 2-D");
    TORCH_CHECK(coords1.dim() == 3, "coords1 must be 3-D");
    TORCH_CHECK(coords2.dim() == 3, "coords2 must be 3-D");
    TORCH_CHECK(z1.dim() == 2, "Z1 must be 2-D");
    TORCH_CHECK(z2.dim() == 2, "Z2 must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");
    TORCH_CHECK(cut_distance > 0.0, "cut_distance must be positive");

    const int nm1 = static_cast<int>(x1.size(0));
    const int max_size1 = static_cast<int>(x1.size(1));
    const int nm2 = static_cast<int>(x2.size(0));
    const int max_size2 = static_cast<int>(x2.size(1));

    TORCH_CHECK(x1.size(3) == max_size1, "X1 shape must be (nm1, max_size1, 5, max_size1)");
    TORCH_CHECK(x2.size(3) == max_size2, "X2 shape must be (nm2, max_size2, 5, max_size2)");
    TORCH_CHECK(n1.size(0) == nm1, "N1.size(0) must equal X1.size(0)");
    TORCH_CHECK(n2.size(0) == nm2, "N2.size(0) must equal X2.size(0)");
    TORCH_CHECK(nn1.size(0) == nm1 && nn1.size(1) == max_size1, "NN1 shape mismatch");
    TORCH_CHECK(nn2.size(0) == nm2 && nn2.size(1) == max_size2, "NN2 shape mismatch");
    TORCH_CHECK(
        coords1.size(0) == nm1 && coords1.size(1) == max_size1 && coords1.size(2) == 3,
        "coords1 shape must be (nm1, max_size1, 3)"
    );
    TORCH_CHECK(
        coords2.size(0) == nm2 && coords2.size(1) == max_size2 && coords2.size(2) == 3,
        "coords2 shape must be (nm2, max_size2, 3)"
    );
    TORCH_CHECK(z1.size(0) == nm1 && z1.size(1) == max_size1, "Z1 shape must be (nm1, max_size1)");
    TORCH_CHECK(z2.size(0) == nm2 && z2.size(1) == max_size2, "Z2 shape must be (nm2, max_size2)");

    const auto n1_host = n1.to(torch::kCPU);
    const int *n1_ptr = n1_host.data_ptr<int>();
    long long d_a_rows = 0;
    for (int a = 0; a < nm1; ++a) {
        TORCH_CHECK(
            n1_ptr[a] >= 0 && n1_ptr[a] <= max_size1, "N1 entries must be in [0, max_size1]"
        );
        d_a_rows += 3LL * n1_ptr[a];
    }
    const auto n2_host = n2.to(torch::kCPU);
    const int *n2_ptr = n2_host.data_ptr<int>();
    long long d_b_cols = 0;
    for (int b = 0; b < nm2; ++b) {
        TORCH_CHECK(
            n2_ptr[b] >= 0 && n2_ptr[b] <= max_size2, "N2 entries must be in [0, max_size2]"
        );
        d_b_cols += 3LL * n2_ptr[b];
    }
    TORCH_CHECK(d_a_rows > 0 && d_b_cols > 0, "kernel_gaussian_hessian: empty molecule set");

    auto H = torch::empty({d_a_rows, d_b_cols}, x1.options());
    if (x1.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_hessian_cu(
            x1.data_ptr<float>(),
            x2.data_ptr<float>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<float>(),
            z1.data_ptr<int>(),
            coords2.data_ptr<float>(),
            z2.data_ptr<int>(),
            H.data_ptr<float>(),
            static_cast<float>(sigma),
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            static_cast<int>(d_b_cols),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_hessian_cu(
            x1.data_ptr<double>(),
            x2.data_ptr<double>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<double>(),
            z1.data_ptr<int>(),
            coords2.data_ptr<double>(),
            z2.data_ptr<int>(),
            H.data_ptr<double>(),
            sigma,
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            static_cast<int>(d_b_cols),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return H;
}

torch::Tensor kernel_gaussian_full(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor n1,
    torch::Tensor n2,
    torch::Tensor nn1,
    torch::Tensor nn2,
    torch::Tensor coords1,
    torch::Tensor z1,
    torch::Tensor coords2,
    torch::Tensor z2,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x1, "X1");
    check_cuda_floating(x2, "X2");
    check_cuda_floating(coords1, "coords1");
    check_cuda_floating(coords2, "coords2");
    check_cuda_int32(n1, "N1");
    check_cuda_int32(n2, "N2");
    check_cuda_int32(nn1, "NN1");
    check_cuda_int32(nn2, "NN2");
    check_cuda_int32(z1, "Z1");
    check_cuda_int32(z2, "Z2");

    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "X1 and X2 dtype must match");
    TORCH_CHECK(x1.scalar_type() == coords1.scalar_type(), "X1 and coords1 dtype must match");
    TORCH_CHECK(x1.scalar_type() == coords2.scalar_type(), "X1 and coords2 dtype must match");
    TORCH_CHECK(x1.dim() == 4 && x2.dim() == 4, "X1/x2 must be 4-D");
    TORCH_CHECK(x1.size(2) == 5 && x2.size(2) == 5, "X channel dim must be 5");
    TORCH_CHECK(n1.dim() == 1 && n2.dim() == 1, "N must be 1-D");
    TORCH_CHECK(nn1.dim() == 2 && nn2.dim() == 2, "NN must be 2-D");
    TORCH_CHECK(coords1.dim() == 3 && coords2.dim() == 3, "coords must be 3-D");
    TORCH_CHECK(z1.dim() == 2 && z2.dim() == 2, "Z must be 2-D");
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    const int nm1 = static_cast<int>(x1.size(0));
    const int max_size1 = static_cast<int>(x1.size(1));
    const int nm2 = static_cast<int>(x2.size(0));
    const int max_size2 = static_cast<int>(x2.size(1));
    TORCH_CHECK(x1.size(3) == max_size1 && x2.size(3) == max_size2, "X trailing size mismatch");
    TORCH_CHECK(n1.size(0) == nm1 && n2.size(0) == nm2, "N size mismatch");
    TORCH_CHECK(nn1.size(0) == nm1 && nn1.size(1) == max_size1, "NN1 shape mismatch");
    TORCH_CHECK(nn2.size(0) == nm2 && nn2.size(1) == max_size2, "NN2 shape mismatch");
    TORCH_CHECK(
        coords1.size(0) == nm1 && coords1.size(1) == max_size1 && coords1.size(2) == 3,
        "coords1 shape mismatch"
    );
    TORCH_CHECK(
        coords2.size(0) == nm2 && coords2.size(1) == max_size2 && coords2.size(2) == 3,
        "coords2 shape mismatch"
    );
    TORCH_CHECK(z1.size(0) == nm1 && z1.size(1) == max_size1, "Z1 shape mismatch");
    TORCH_CHECK(z2.size(0) == nm2 && z2.size(1) == max_size2, "Z2 shape mismatch");

    const auto n1_host = n1.to(torch::kCPU);
    const auto n2_host = n2.to(torch::kCPU);
    const int *n1_ptr = n1_host.data_ptr<int>();
    const int *n2_ptr = n2_host.data_ptr<int>();
    long long d_a_rows = 0;
    long long d_b_cols = 0;
    for (int a = 0; a < nm1; ++a) {
        d_a_rows += 3LL * n1_ptr[a];
    }
    for (int b = 0; b < nm2; ++b) {
        d_b_cols += 3LL * n2_ptr[b];
    }
    TORCH_CHECK(d_a_rows > 0 && d_b_cols > 0, "kernel_gaussian_full: empty molecule set");

    auto K = torch::empty({nm1 + d_a_rows, nm2 + d_b_cols}, x1.options());
    if (x1.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_full_cu(
            x1.data_ptr<float>(),
            x2.data_ptr<float>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<float>(),
            z1.data_ptr<int>(),
            coords2.data_ptr<float>(),
            z2.data_ptr<int>(),
            K.data_ptr<float>(),
            static_cast<float>(sigma),
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            static_cast<int>(d_b_cols),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_full_cu(
            x1.data_ptr<double>(),
            x2.data_ptr<double>(),
            n1.data_ptr<int>(),
            n2.data_ptr<int>(),
            nn1.data_ptr<int>(),
            nn2.data_ptr<int>(),
            coords1.data_ptr<double>(),
            z1.data_ptr<int>(),
            coords2.data_ptr<double>(),
            z2.data_ptr<int>(),
            K.data_ptr<double>(),
            sigma,
            nm1,
            nm2,
            max_size1,
            max_size2,
            static_cast<int>(d_a_rows),
            static_cast<int>(d_b_cols),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return K;
}

torch::Tensor kernel_gaussian_full_symm(
    torch::Tensor x,
    torch::Tensor n,
    torch::Tensor nn,
    torch::Tensor coords,
    torch::Tensor z,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x, "X");
    check_cuda_floating(coords, "coords");
    check_cuda_int32(n, "N");
    check_cuda_int32(nn, "NN");
    check_cuda_int32(z, "Z");
    TORCH_CHECK(x.scalar_type() == coords.scalar_type(), "X and coords dtype must match");
    TORCH_CHECK(x.dim() == 4 && x.size(2) == 5, "X must be (nm, max_size, 5, max_size)");
    TORCH_CHECK(n.dim() == 1 && nn.dim() == 2 && coords.dim() == 3 && z.dim() == 2);
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    const int nm = static_cast<int>(x.size(0));
    const int max_size = static_cast<int>(x.size(1));
    TORCH_CHECK(x.size(3) == max_size, "X trailing size mismatch");
    TORCH_CHECK(n.size(0) == nm, "N size mismatch");
    TORCH_CHECK(nn.size(0) == nm && nn.size(1) == max_size, "NN shape mismatch");
    TORCH_CHECK(
        coords.size(0) == nm && coords.size(1) == max_size && coords.size(2) == 3,
        "coords shape mismatch"
    );
    TORCH_CHECK(z.size(0) == nm && z.size(1) == max_size, "Z shape mismatch");

    const auto n_host = n.to(torch::kCPU);
    const int *n_ptr = n_host.data_ptr<int>();
    long long D = 0;
    for (int a = 0; a < nm; ++a) {
        D += 3LL * n_ptr[a];
    }
    TORCH_CHECK(D > 0, "kernel_gaussian_full_symm: empty molecule set");

    auto K = torch::empty({nm + D, nm + D}, x.options());
    if (x.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_full_symm_cu(
            x.data_ptr<float>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<float>(),
            z.data_ptr<int>(),
            K.data_ptr<float>(),
            static_cast<float>(sigma),
            nm,
            max_size,
            static_cast<int>(D),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_full_symm_cu(
            x.data_ptr<double>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<double>(),
            z.data_ptr<int>(),
            K.data_ptr<double>(),
            sigma,
            nm,
            max_size,
            static_cast<int>(D),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return K;
}

torch::Tensor kernel_gaussian_full_symm_rfp(
    torch::Tensor x,
    torch::Tensor n,
    torch::Tensor nn,
    torch::Tensor coords,
    torch::Tensor z,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x, "X");
    check_cuda_floating(coords, "coords");
    check_cuda_int32(n, "N");
    check_cuda_int32(nn, "NN");
    check_cuda_int32(z, "Z");
    TORCH_CHECK(x.scalar_type() == coords.scalar_type(), "X and coords dtype must match");
    TORCH_CHECK(x.dim() == 4 && x.size(2) == 5, "X must be (nm, max_size, 5, max_size)");
    TORCH_CHECK(n.dim() == 1 && nn.dim() == 2 && coords.dim() == 3 && z.dim() == 2);
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    const int nm = static_cast<int>(x.size(0));
    const int max_size = static_cast<int>(x.size(1));
    TORCH_CHECK(x.size(3) == max_size, "X trailing size mismatch");
    TORCH_CHECK(n.size(0) == nm, "N size mismatch");
    TORCH_CHECK(nn.size(0) == nm && nn.size(1) == max_size, "NN shape mismatch");
    TORCH_CHECK(
        coords.size(0) == nm && coords.size(1) == max_size && coords.size(2) == 3,
        "coords shape mismatch"
    );
    TORCH_CHECK(z.size(0) == nm && z.size(1) == max_size, "Z shape mismatch");

    const auto n_host = n.to(torch::kCPU);
    const int *n_ptr = n_host.data_ptr<int>();
    long long D = 0;
    for (int a = 0; a < nm; ++a) {
        D += 3LL * n_ptr[a];
    }
    TORCH_CHECK(D > 0, "kernel_gaussian_full_symm_rfp: empty molecule set");

    const long long BIG = nm + D;
    const long long rfp_len = BIG * (BIG + 1) / 2;
    auto K_rfp = torch::empty({rfp_len}, x.options());
    if (x.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_full_symm_rfp_cu(
            x.data_ptr<float>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<float>(),
            z.data_ptr<int>(),
            K_rfp.data_ptr<float>(),
            static_cast<float>(sigma),
            nm,
            max_size,
            static_cast<int>(D),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_full_symm_rfp_cu(
            x.data_ptr<double>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<double>(),
            z.data_ptr<int>(),
            K_rfp.data_ptr<double>(),
            sigma,
            nm,
            max_size,
            static_cast<int>(D),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return K_rfp;
}

torch::Tensor kernel_gaussian_gradient(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor n1,
    torch::Tensor n2,
    torch::Tensor nn1,
    torch::Tensor nn2,
    torch::Tensor coords1,
    torch::Tensor z1,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    // Single-query convenience wrapper around jacobian with CPU gradient layout
    // G[alpha, mu, b] = dK[A,b]/dR_A[alpha, mu].
    TORCH_CHECK(x1.dim() == 4 && x1.size(0) == 1, "kernel_gaussian_gradient requires nm1 == 1");
    check_cuda_int32(n1, "N1");
    const auto n1_host = n1.to(torch::kCPU);
    TORCH_CHECK(n1_host.numel() == 1, "N1 must have length 1");
    const int n_atoms = n1_host.data_ptr<int>()[0];
    TORCH_CHECK(n_atoms > 0, "kernel_gaussian_gradient: empty query molecule");

    auto J = kernel_gaussian_jacobian(
        x1,
        x2,
        n1,
        n2,
        nn1,
        nn2,
        coords1,
        z1,
        sigma,
        two_body_scaling,
        two_body_width,
        two_body_power,
        three_body_scaling,
        three_body_width,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm
    );
    // J is (3*n_atoms, nm2) with row = alpha*3 + mu → (n_atoms, 3, nm2).
    return J.view({n_atoms, 3, J.size(1)});
}

torch::Tensor kernel_gaussian_hessian_symm(
    torch::Tensor x,
    torch::Tensor n,
    torch::Tensor nn,
    torch::Tensor coords,
    torch::Tensor z,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x, "X");
    check_cuda_floating(coords, "coords");
    check_cuda_int32(n, "N");
    check_cuda_int32(nn, "NN");
    check_cuda_int32(z, "Z");
    TORCH_CHECK(x.scalar_type() == coords.scalar_type(), "X and coords dtype must match");
    TORCH_CHECK(x.dim() == 4 && x.size(2) == 5, "X must be (nm, max_size, 5, max_size)");
    TORCH_CHECK(n.dim() == 1 && nn.dim() == 2 && coords.dim() == 3 && z.dim() == 2);
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    const int nm = static_cast<int>(x.size(0));
    const int max_size = static_cast<int>(x.size(1));
    TORCH_CHECK(x.size(3) == max_size, "X trailing size mismatch");
    TORCH_CHECK(n.size(0) == nm, "N size mismatch");
    TORCH_CHECK(nn.size(0) == nm && nn.size(1) == max_size, "NN shape mismatch");
    TORCH_CHECK(
        coords.size(0) == nm && coords.size(1) == max_size && coords.size(2) == 3,
        "coords shape mismatch"
    );
    TORCH_CHECK(z.size(0) == nm && z.size(1) == max_size, "Z shape mismatch");

    const auto n_host = n.to(torch::kCPU);
    const int *n_ptr = n_host.data_ptr<int>();
    long long D = 0;
    for (int a = 0; a < nm; ++a) {
        D += 3LL * n_ptr[a];
    }
    TORCH_CHECK(D > 0, "kernel_gaussian_hessian_symm: empty molecule set");

    auto H = torch::empty({D, D}, x.options());
    if (x.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_hessian_symm_cu(
            x.data_ptr<float>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<float>(),
            z.data_ptr<int>(),
            H.data_ptr<float>(),
            static_cast<float>(sigma),
            nm,
            max_size,
            static_cast<int>(D),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_hessian_symm_cu(
            x.data_ptr<double>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<double>(),
            z.data_ptr<int>(),
            H.data_ptr<double>(),
            sigma,
            nm,
            max_size,
            static_cast<int>(D),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return H;
}

torch::Tensor kernel_gaussian_hessian_symm_rfp(
    torch::Tensor x,
    torch::Tensor n,
    torch::Tensor nn,
    torch::Tensor coords,
    torch::Tensor z,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int fourier_order,
    bool use_atm
) {
    check_cuda_floating(x, "X");
    check_cuda_floating(coords, "coords");
    check_cuda_int32(n, "N");
    check_cuda_int32(nn, "NN");
    check_cuda_int32(z, "Z");
    TORCH_CHECK(x.scalar_type() == coords.scalar_type(), "X and coords dtype must match");
    TORCH_CHECK(x.dim() == 4 && x.size(2) == 5, "X must be (nm, max_size, 5, max_size)");
    TORCH_CHECK(n.dim() == 1 && nn.dim() == 2 && coords.dim() == 3 && z.dim() == 2);
    TORCH_CHECK(sigma > 0.0, "sigma must be positive");

    const int nm = static_cast<int>(x.size(0));
    const int max_size = static_cast<int>(x.size(1));
    TORCH_CHECK(x.size(3) == max_size, "X trailing size mismatch");
    TORCH_CHECK(n.size(0) == nm, "N size mismatch");
    TORCH_CHECK(nn.size(0) == nm && nn.size(1) == max_size, "NN shape mismatch");
    TORCH_CHECK(
        coords.size(0) == nm && coords.size(1) == max_size && coords.size(2) == 3,
        "coords shape mismatch"
    );
    TORCH_CHECK(z.size(0) == nm && z.size(1) == max_size, "Z shape mismatch");

    const auto n_host = n.to(torch::kCPU);
    const int *n_ptr = n_host.data_ptr<int>();
    long long D = 0;
    for (int a = 0; a < nm; ++a) {
        D += 3LL * n_ptr[a];
    }
    TORCH_CHECK(D > 0, "kernel_gaussian_hessian_symm_rfp: empty molecule set");

    const long long rfp_len = D * (D + 1) / 2;
    auto H_rfp = torch::empty({rfp_len}, x.options());
    if (x.scalar_type() == torch::kFloat32) {
        kf::fchl18::kernel_gaussian_hessian_symm_rfp_cu(
            x.data_ptr<float>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<float>(),
            z.data_ptr<int>(),
            H_rfp.data_ptr<float>(),
            static_cast<float>(sigma),
            nm,
            max_size,
            static_cast<int>(D),
            static_cast<float>(two_body_scaling),
            static_cast<float>(two_body_width),
            static_cast<float>(two_body_power),
            static_cast<float>(three_body_scaling),
            static_cast<float>(three_body_width),
            static_cast<float>(three_body_power),
            static_cast<float>(cut_start),
            static_cast<float>(cut_distance),
            fourier_order,
            use_atm
        );
    } else {
        kf::fchl18::kernel_gaussian_hessian_symm_rfp_cu(
            x.data_ptr<double>(),
            n.data_ptr<int>(),
            nn.data_ptr<int>(),
            coords.data_ptr<double>(),
            z.data_ptr<int>(),
            H_rfp.data_ptr<double>(),
            sigma,
            nm,
            max_size,
            static_cast<int>(D),
            two_body_scaling,
            two_body_width,
            two_body_power,
            three_body_scaling,
            three_body_width,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm
        );
    }
    return H_rfp;
}

}  // namespace

PYBIND11_MODULE(cuda_fchl18_kernel, m) {
    m.doc() = R"doc(
CUDA-accelerated FCHL18 scalar kernels.

Current scope:
- fp32 and fp64
- rectangular energy-only kernel_gaussian
- symmetric energy-only kernel_gaussian_symm (upper triangle + mirror)
- symmetric RFP kernel_gaussian_symm_rfp (TRANSR='N', UPLO='U')
- Jacobian kernel_gaussian_jacobian (dK/dR of the query molecules)
- Jacobian-T kernel_gaussian_jacobian_t (transposed store layout)
- Gradient kernel_gaussian_gradient (single-query, CPU G layout)
- Hessian kernel_gaussian_hessian (d2K/dR_A dR_B)
- Symmetric Hessian kernel_gaussian_hessian_symm / _symm_rfp
- Full combined kernel_gaussian_full / _symm / _symm_rfp
- operates on precomputed FCHL18 representations already resident on GPU
)doc";

    m.def(
        "kernel_gaussian",
        &kernel_gaussian,
        py::arg("X1"),
        py::arg("X2"),
        py::arg("N1"),
        py::arg("N2"),
        py::arg("NN1"),
        py::arg("NN2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"doc(
Compute the rectangular FCHL18 Gaussian kernel matrix on GPU.

Parameters
----------
X1, X2 : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
    Precomputed FCHL18 representations. Dtypes must match.
N1, N2 : torch.Tensor, shape (nm,), int32, CUDA
    Number of real atoms per molecule.
NN1, NN2 : torch.Tensor, shape (nm, max_size), int32, CUDA
    Number of neighbors per atom.
sigma : float
    Gaussian kernel length-scale.

Returns
-------
torch.Tensor, shape (nm1, nm2), same dtype as X1, CUDA
    Rectangular energy-only kernel matrix.
)doc"
    );

    m.def(
        "kernel_gaussian_symm",
        &kernel_gaussian_symm,
        py::arg("X"),
        py::arg("N"),
        py::arg("NN"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"doc(
Compute the symmetric FCHL18 Gaussian kernel matrix on GPU.

Only upper-triangle molecule pairs (a <= b) are evaluated; K[b,a] is mirrored.

Parameters
----------
X : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
    Precomputed FCHL18 representations.
N : torch.Tensor, shape (nm,), int32, CUDA
    Number of real atoms per molecule.
NN : torch.Tensor, shape (nm, max_size), int32, CUDA
    Number of neighbors per atom.
sigma : float
    Gaussian kernel length-scale.

Returns
-------
torch.Tensor, shape (nm, nm), same dtype as X, CUDA
    Symmetric energy-only kernel matrix.
)doc"
    );

    m.def(
        "kernel_gaussian_symm_rfp",
        &kernel_gaussian_symm_rfp,
        py::arg("X"),
        py::arg("N"),
        py::arg("NN"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"doc(
Compute the symmetric FCHL18 Gaussian kernel in RFP packed format on GPU.

Only upper-triangle molecule pairs (a <= b) are evaluated. Output uses
TRANSR='N', UPLO='U' (same layout as CPU fchl18_kernel.kernel_gaussian_symm_rfp
and kf::rfp_index_upper_N). Compatible with kernelmath.cho_solve_rfp.

Unpack with: kernelmath.rfp_to_full(K_rfp, nm, uplo='L', transr='N')
(C-order UPLO swap in kernelmath; LAPACK storage is UPLO='U').

Parameters
----------
X : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
    Precomputed FCHL18 representations.
N : torch.Tensor, shape (nm,), int32, CUDA
    Number of real atoms per molecule.
NN : torch.Tensor, shape (nm, max_size), int32, CUDA
    Number of neighbors per atom.
sigma : float
    Gaussian kernel length-scale.

Returns
-------
torch.Tensor, shape (nm*(nm+1)//2,), same dtype as X, CUDA
    Upper-triangle RFP-packed energy-only kernel.
)doc"
    );

    m.def(
        "kernel_gaussian_jacobian",
        &kernel_gaussian_jacobian,
        py::arg("X1"),
        py::arg("X2"),
        py::arg("N1"),
        py::arg("N2"),
        py::arg("NN1"),
        py::arg("NN2"),
        py::arg("coords1"),
        py::arg("Z1"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"doc(
Compute the FCHL18 Jacobian kernel dK/dR_A on GPU.

J[row, b] = dK(A_a, B_b) / dR_{A_a}[alpha, mu]
with row = row_offset[a] + alpha*3 + mu and row_offset[a] = 3 * sum_{a' < a} N1[a'].
Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_jacobian.

Unlike the CPU function, which takes raw coordinate lists and builds the query
representation internally, this takes a precomputed X1 (as kernel_gaussian does)
plus coords1/Z1. The coordinates are needed to map each neighbour slot of the
representation back to an atom index for the chain rule; they must be the same
padded coordinates the representation was generated from.

Parameters
----------
X1, X2 : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
    Precomputed FCHL18 representations. Dtypes must match.
N1, N2 : torch.Tensor, shape (nm,), int32, CUDA
    Number of real atoms per molecule.
NN1, NN2 : torch.Tensor, shape (nm, max_size), int32, CUDA
    Number of neighbors per atom.
coords1 : torch.Tensor, shape (nm1, max_size1, 3), same dtype as X1, CUDA
    Padded Cartesian coordinates of the query molecules.
Z1 : torch.Tensor, shape (nm1, max_size1), int32, CUDA
    Padded nuclear charges of the query molecules.
sigma : float
    Gaussian kernel length-scale.

Returns
-------
torch.Tensor, shape (3 * sum(N1), nm2), same dtype as X1, CUDA
    Jacobian of the energy kernel with respect to the query coordinates.
)doc"
    );

    m.def(
        "kernel_gaussian_jacobian_t",
        &kernel_gaussian_jacobian_t,
        py::arg("X1"),
        py::arg("X2"),
        py::arg("N1"),
        py::arg("N2"),
        py::arg("NN1"),
        py::arg("NN2"),
        py::arg("coords1"),
        py::arg("Z1"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"doc(
Compute the FCHL18 Jacobian-T kernel on GPU.

Same computation as kernel_gaussian_jacobian, but stores
Jt[b, row] = dK(A_a, B_b) / dR_{A_a}[...] with shape (nm2, 3*sum(N1)).

Numerically comparable to CPU fchl18_kernel.kernel_gaussian_jacobian_t after
argument remapping; the CUDA keyword surface uses X1/X2/N1/N2/NN1/NN2/coords1/Z1
rather than the CPU train/test list API.
)doc"
    );

    m.def(
        "kernel_gaussian_gradient",
        &kernel_gaussian_gradient,
        py::arg("X1"),
        py::arg("X2"),
        py::arg("N1"),
        py::arg("N2"),
        py::arg("NN1"),
        py::arg("NN2"),
        py::arg("coords1"),
        py::arg("Z1"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 0.5,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 2,
        py::arg("use_atm") = true,
        R"doc(
Compute the FCHL18 gradient kernel for a single query molecule on GPU.

Requires nm1 == 1. Returns G with shape (n_atoms_A, 3, nm2) matching CPU
kernelforge.fchl18_kernel.kernel_gaussian_gradient. Implemented as a reshape of
kernel_gaussian_jacobian.
)doc"
    );

    m.def(
        "kernel_gaussian_hessian",
        &kernel_gaussian_hessian,
        py::arg("X1"),
        py::arg("X2"),
        py::arg("N1"),
        py::arg("N2"),
        py::arg("NN1"),
        py::arg("NN2"),
        py::arg("coords1"),
        py::arg("Z1"),
        py::arg("coords2"),
        py::arg("Z2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"doc(
Compute the FCHL18 Hessian kernel d2K/dR_A dR_B on GPU.

H[row, col] = d2K(A_a, B_b) / dR_{A_a}[alpha, mu] dR_{B_b}[beta, nu]
with row = row_offset[a] + alpha*3 + mu, col = col_offset[b] + beta*3 + nu,
row_offset[a] = 3 * sum_{a' < a} N1[a'] and col_offset[b] = 3 * sum_{b' < b} N2[b'].
Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_hessian.

Like kernel_gaussian_jacobian, both sides take a precomputed representation plus
the padded coordinates and nuclear charges the representation was generated from;
those are needed to map neighbour slots back to atom indices for the chain rule.

The underlying cross-Hessian is only exact when the cutoff is inactive
(cut_start >= 1 or cut_distance far beyond every interatomic distance) and
use_atm is False, matching the CPU implementation.

Parameters
----------
X1, X2 : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
    Precomputed FCHL18 representations. Dtypes must match.
N1, N2 : torch.Tensor, shape (nm,), int32, CUDA
    Number of real atoms per molecule.
NN1, NN2 : torch.Tensor, shape (nm, max_size), int32, CUDA
    Number of neighbors per atom.
coords1, coords2 : torch.Tensor, shape (nm, max_size, 3), same dtype as X1, CUDA
    Padded Cartesian coordinates of each side's molecules.
Z1, Z2 : torch.Tensor, shape (nm, max_size), int32, CUDA
    Padded nuclear charges of each side's molecules.
sigma : float
    Gaussian kernel length-scale.

Returns
-------
torch.Tensor, shape (3 * sum(N1), 3 * sum(N2)), same dtype as X1, CUDA
    Force-force block matrix, one dense (3*N1[a], 3*N2[b]) block per molecule pair.
)doc"
    );

    m.def(
        "kernel_gaussian_hessian_symm",
        &kernel_gaussian_hessian_symm,
        py::arg("X"),
        py::arg("N"),
        py::arg("NN"),
        py::arg("coords"),
        py::arg("Z"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"doc(
Compute the symmetric FCHL18 Hessian kernel on GPU.

Evaluates lower-triangle molecule pairs once, mirrors off-diagonal blocks, and
symmetrises diagonal blocks. Shape (D, D) with D = 3*sum(N).
Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_hessian_symm.
)doc"
    );

    m.def(
        "kernel_gaussian_hessian_symm_rfp",
        &kernel_gaussian_hessian_symm_rfp,
        py::arg("X"),
        py::arg("N"),
        py::arg("NN"),
        py::arg("coords"),
        py::arg("Z"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"doc(
Compute the symmetric FCHL18 Hessian kernel in RFP format on GPU.

TRANSR='N', UPLO='U'. Length D*(D+1)/2 with D = 3*sum(N).
Unpack with: kernelmath.rfp_to_full(H_rfp, D, uplo='L', transr='N')
Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_hessian_symm_rfp.
)doc"
    );

    m.def(
        "kernel_gaussian_full",
        &kernel_gaussian_full,
        py::arg("X1"),
        py::arg("X2"),
        py::arg("N1"),
        py::arg("N2"),
        py::arg("NN1"),
        py::arg("NN2"),
        py::arg("coords1"),
        py::arg("Z1"),
        py::arg("coords2"),
        py::arg("Z2"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"doc(
Compute the full FCHL18 energy+force kernel matrix on GPU.

Block layout (N_A=nm1, N_B=nm2, D_A=3*sum(N1), D_B=3*sum(N2)):
  K[0:N_A, 0:N_B]     scalar
  K[0:N_A, N_B:]      jacobian_t (dK/dR_B)
  K[N_A:,  0:N_B]     jacobian   (dK/dR_A)
  K[N_A:,  N_B:]      hessian
Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_full.
)doc"
    );

    m.def(
        "kernel_gaussian_full_symm",
        &kernel_gaussian_full_symm,
        py::arg("X"),
        py::arg("N"),
        py::arg("NN"),
        py::arg("coords"),
        py::arg("Z"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"doc(
Compute the symmetric full FCHL18 energy+force kernel on GPU.

Shape (N+D, N+D) with N=nm and D=3*sum(N). Hessian uses lower-triangle pairs
only. Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_full_symm.
)doc"
    );

    m.def(
        "kernel_gaussian_full_symm_rfp",
        &kernel_gaussian_full_symm_rfp,
        py::arg("X"),
        py::arg("N"),
        py::arg("NN"),
        py::arg("coords"),
        py::arg("Z"),
        py::arg("sigma"),
        py::arg("two_body_scaling") = 2.0,
        py::arg("two_body_width") = 0.1,
        py::arg("two_body_power") = 6.0,
        py::arg("three_body_scaling") = 2.0,
        py::arg("three_body_width") = 3.0,
        py::arg("three_body_power") = 3.0,
        py::arg("cut_start") = 1.0,
        py::arg("cut_distance") = 1e6,
        py::arg("fourier_order") = 1,
        py::arg("use_atm") = false,
        R"doc(
Compute the symmetric full FCHL18 kernel in RFP packed format on GPU.

TRANSR='N', UPLO='U'. Unpack with kernelmath.rfp_to_full(..., uplo='L', transr='N').
Matches CPU kernelforge.fchl18_kernel.kernel_gaussian_full_symm_rfp.
)doc"
    );
}
