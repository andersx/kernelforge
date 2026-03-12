#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace kf {
namespace fchl19v2 {

// ==================== Enum types for basis selection ====================

enum class TwoBodyType {
    LogNormal,       // T1: current FCHL19 baseline
    GaussianR,       // T2: fixed-width Gaussian in r
    GaussianLogR,    // T3: fixed-width Gaussian in ln(r)
    GaussianRNoPow,  // T4: fixed-width Gaussian in r, no 1/r^p
    Bessel,          // T5: radial Bessel basis sin(k*pi*r/rcut)/r
};

enum class ThreeBodyType {
    OddFourier_Rbar,               // A1: odd cos+sin harmonics, r_bar radial
    CosineSeries_Rbar,             // A2: full cosine series, r_bar radial
    OddFourier_SplitR,             // A3: odd cos+sin harmonics, r_plus/r_minus radial
    CosineSeries_SplitR,           // A4: full cosine series, r_plus/r_minus radial
    CosineSeries_SplitR_NoATM,     // A5: full cosine series, r_plus/r_minus, no ATM factor
    OddFourier_ElementResolved,    // A6: odd harmonics; B!=C: (r_ij,r_ik) ordered; B==C: SplitR
    CosineSeries_ElementResolved,  // A7: cosine series;  B!=C: (r_ij,r_ik) ordered; B==C: SplitR
    Legendre_BesselJoint           // A8: Legendre P_l angular + Bessel radial + diagonal joint coupling
};

// ==================== Utility ====================

// Convert string to enum (for Python bindings)
TwoBodyType two_body_type_from_string(const std::string &s);
ThreeBodyType three_body_type_from_string(const std::string &s);

// Compute the expected representation size per atom
std::size_t compute_rep_size(
    std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3, std::size_t nabasis,
    ThreeBodyType three_body_type, std::size_t nbasis3_minus = 0
);

// ==================== Generation functions ====================

// Forward pass only: compute representation
void generate(
    const std::vector<double> &coords, const std::vector<int> &nuclear_z,
    const std::vector<int> &elements, const std::vector<double> &Rs2,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta2,
    double eta3, double eta3_minus, double zeta, double rcut, double acut,
    double two_body_decay, double three_body_decay, double three_body_weight,
    TwoBodyType two_body_type, ThreeBodyType three_body_type, int nabasis,
    bool use_two_body, bool use_three_body, bool use_atm, std::vector<double> &rep
);

// Forward + Jacobian: compute representation and its gradient wrt coordinates
void generate_and_gradients(
    const std::vector<double> &coords, const std::vector<int> &nuclear_z,
    const std::vector<int> &elements, const std::vector<double> &Rs2,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta2,
    double eta3, double eta3_minus, double zeta, double rcut, double acut,
    double two_body_decay, double three_body_decay, double three_body_weight,
    TwoBodyType two_body_type, ThreeBodyType three_body_type, int nabasis,
    bool use_two_body, bool use_three_body, bool use_atm, std::vector<double> &rep,
    std::vector<double> &grad
);

}  // namespace fchl19v2
}  // namespace kf
