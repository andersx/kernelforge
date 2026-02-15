#pragma once
#include <cstddef>
#include <vector>

namespace fchl19 {

// ---- constants ----
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double SQRT_2PI = 2.5066282746310005;  // sqrt(2*pi)
constexpr double EPS = 1e-12;

// ---- core API ----

// Compute the expected representation size per atom
// size_t compute_rep_size(size_t nelements,
//                         size_t nbasis2,
//                         size_t nbasis3,
//                         size_t nabasis);

std::size_t compute_rep_size(size_t nelements, size_t nbasis2, size_t nbasis3, size_t nabasis);
// Main ACSF generator
// - coords: length natoms*3 (x,y,z per atom)
// - nuclear_z: length natoms
// - elements: list of unique elements present
// - Rs2, Rs3, Ts: basis arrays
// Output: rep will be resized to natoms * rep_size
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

void flocal_kernel_fast(const double *x1, const int *q1, const int *n1, int nm1, int max_n1,
                        const double *x2, const int *q2, const int *n2, int nm2, int max_n2,
                        int rep, double sigma,
                        double *kernel /* out: [nm2 * nm1], row-major (b, a) */);

// Compute row-wise squared L2 norms for an (n × d) row-major matrix X.
void rowwise_self_norms(const double *X, std::size_t n, std::size_t d, double *out);

// FCHL19 local kernel (charge-matched Gaussian) between two *sets* of molecules.
// Data layout (row-major):
//   X1: concat of all atoms' representations for molecules in set 1, shape (N1_total × d)
//   Q1: concat of all atoms' integer charges, length N1_total
//   offsets1: size (nm1 + 1). For molecule a: atoms are in [offsets1[a], offsets1[a+1])
// Likewise for X2/Q2/offsets2 (set 2).
//
// alpha should be  -1 / (2*sigma^2).  (We keep alpha to mirror your kernel style.)
// Output K has shape (nm1 × nm2), row-major with ldc = nm2.

void flocal_kernel(const std::vector<double> &x1, const std::vector<double> &x2,
                   const std::vector<int> &q1, const std::vector<int> &q2,
                   const std::vector<int> &n1, const std::vector<int> &n2, int nm1, int nm2,
                   int max_atoms1, int max_atoms2, int rep_size, double sigma, double *kernel_out
                   // std::vector<double>& kernel
);

void flocal_kernel_symmetric_rfp(const std::vector<double> &x, const std::vector<int> &q,
                                 const std::vector<int> &n, int nm, int max_atoms, int rep_size,
                                 double sigma, double *arf);

void flocal_kernel_symmetric(const std::vector<double> &x, const std::vector<int> &q,
                             const std::vector<int> &n, int nm, int max_atoms, int rep_size,
                             double sigma, double *kernel_out);

void fatomic_local_gradient_kernel(
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

void fgdml_kernel(const std::vector<double> &x1,   // (nm1, max_atoms1, rep_size)
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

void fgdml_kernel_symmetric_lower(
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
