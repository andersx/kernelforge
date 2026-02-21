#pragma once
#include <cstddef>
#include <vector>

namespace kf {
namespace fchl19 {

// Compute the expected representation size per atom
std::size_t compute_rep_size(std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
                             std::size_t nabasis);

// FCHL19 Representation generation functions

void generate_fchl_acsf(const std::vector<double> &coords, const std::vector<int> &nuclear_z,
                        const std::vector<int> &elements, const std::vector<double> &Rs2,
                        const std::vector<double> &Rs3, const std::vector<double> &Ts, double eta2,
                        double eta3, double zeta, double rcut, double acut, double two_body_decay,
                        double three_body_decay, double three_body_weight,
                        std::vector<double> &rep);

void generate_fchl_acsf_and_gradients(
    const std::vector<double> &coords, const std::vector<int> &nuclear_z,
    const std::vector<int> &elements, const std::vector<double> &Rs2,
    const std::vector<double> &Rs3, const std::vector<double> &Ts, double eta2, double eta3,
    double zeta, double rcut, double acut, double two_body_decay, double three_body_decay,
    double three_body_weight, std::vector<double> &rep, std::vector<double> &grad);

}  // namespace fchl19
}  // namespace kf
