#pragma once
#include <cstddef>
#include <vector>

namespace kf {
namespace fchl19 {

// Local (atom-pair-wise) Gaussian kernel functions

void kernel_gaussian(const std::vector<double> &x1, const std::vector<double> &x2,
                     const std::vector<int> &q1, const std::vector<int> &q2,
                     const std::vector<int> &n1, const std::vector<int> &n2, int nm1, int nm2,
                     int max_atoms1, int max_atoms2, int rep_size, double sigma, double *kernel_out);

void kernel_gaussian_symm_rfp(const std::vector<double> &x, const std::vector<int> &q,
                              const std::vector<int> &n, int nm, int max_atoms, int rep_size,
                              double sigma, double *arf);

void kernel_gaussian_symm(const std::vector<double> &x, const std::vector<int> &q,
                          const std::vector<int> &n, int nm, int max_atoms, int rep_size,
                          double sigma, double *kernel_out);

void kernel_gaussian_jacobian(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep)
    const std::vector<double> &x2,   // (nm2, max_atoms2, rep)
    const std::vector<double> &dX2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    const std::vector<int> &q1,      // (nm1, max_atoms1)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n1,      // (nm1)
    const std::vector<int> &n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq2,  // should equal 3 * sum(n2)
    double sigma,
    double *kernel_out  // (nm1, naq2) row-major
);

void kernel_gaussian_hessian(const std::vector<double> &x1,   // (nm1, max_atoms1, rep_size)
                             const std::vector<double> &x2,   // (nm2, max_atoms2, rep_size)
                             const std::vector<double> &dx1,  // (nm1, max_atoms1, rep_size, 3*max_atoms1)
                             const std::vector<double> &dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
                             const std::vector<int> &q1,      // (nm1, max_atoms1)
                             const std::vector<int> &q2,      // (nm2, max_atoms2)
                             const std::vector<int> &n1,      // (nm1)
                             const std::vector<int> &n2,      // (nm2)
                             int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
                             int naq1,  // must equal 3 * sum_a n1[a]
                             int naq2,  // must equal 3 * sum_b n2[b]
                             double sigma,
                             double *kernel_out  // (naq2, naq1) row-major
);

void kernel_gaussian_hessian_symm(
    const std::vector<double> &x,   // (nm, max_atoms, rep_size)
    const std::vector<double> &dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int> &q,      // (nm, max_atoms)
    const std::vector<int> &n,      // (nm)
    int nm, int max_atoms, int rep_size,
    int naq,  // must be 3 * sum_m n[m]
    double sigma,
    double *kernel_out  // (naq, naq) row-major
);

}  // namespace fchl19
}  // namespace kf
