#pragma once
#include <cstddef>
#include <vector>

namespace kf {
namespace fchl19 {

// Local (atom-pair-wise) Gaussian kernel functions

void kernel_gaussian(
    const std::vector<double> &x1, const std::vector<double> &x2, const std::vector<int> &q1,
    const std::vector<int> &q2, const std::vector<int> &n1, const std::vector<int> &n2, int nm1,
    int nm2, int max_atoms1, int max_atoms2, int rep_size, double sigma, double *kernel_out
);

void kernel_gaussian_symm_rfp(
    const std::vector<double> &x, const std::vector<int> &q, const std::vector<int> &n, int nm,
    int max_atoms, int rep_size, double sigma, double *arf
);

void kernel_gaussian_symm(
    const std::vector<double> &x, const std::vector<int> &q, const std::vector<int> &n, int nm,
    int max_atoms, int rep_size, double sigma, double *kernel_out
);

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

// Transposed Jacobian kernel: Jacobians on set-1 side (dX1).
// Output shape: (naq1, nm2), where naq1 = 3 * sum(n1).
// Property: kernel_gaussian_jacobian_t(x1, dX1, x2, ...) ==
//           kernel_gaussian_jacobian(x2, x1, dX1, ...).T
void kernel_gaussian_jacobian_t(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep)
    const std::vector<double> &x2,   // (nm2, max_atoms2, rep)
    const std::vector<double> &dX1,  // (nm1, max_atoms1, rep, 3*max_atoms1)
    const std::vector<int> &q1,      // (nm1, max_atoms1)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n1,      // (nm1)
    const std::vector<int> &n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // should equal 3 * sum(n1)
    double sigma,
    double *kernel_out  // (naq1, nm2) row-major
);

void kernel_gaussian_hessian(
    const std::vector<double> &x1,   // (nm1, max_atoms1, rep_size)
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

void kernel_gaussian_hessian_symm_rfp(
    const std::vector<double> &x,   // (nm, max_atoms, rep_size)
    const std::vector<double> &dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int> &q,      // (nm, max_atoms)
    const std::vector<int> &n,      // (nm)
    int nm, int max_atoms, int rep_size,
    int naq,  // must be 3 * sum_m n[m]
    double sigma,
    double *arf  // length naq*(naq+1)/2, RFP TRANSR='N', UPLO='U'
);

// Full combined energy+force kernel (asymmetric).
// Output: ((nm1+naq1) x (nm2+naq2)) row-major matrix.
// Blocks: [0:nm1,0:nm2]=scalar, [0:nm1,nm2:]=jac_t, [nm1:,0:nm2]=jac, [nm1:,nm2:]=hessian.
void kernel_gaussian_full(
    const std::vector<double> &x1, const std::vector<double> &x2, const std::vector<double> &dx1,
    const std::vector<double> &dx2, const std::vector<int> &q1, const std::vector<int> &q2,
    const std::vector<int> &n1, const std::vector<int> &n2, int nm1, int nm2, int max_atoms1,
    int max_atoms2, int rep_size, int naq1, int naq2, double sigma,
    double *kernel_out  // ((nm1+naq1) x (nm2+naq2)), row-major
);

// Full combined energy+force kernel (symmetric, full square output).
// Output: ((nm+naq) x (nm+naq)) row-major, fully filled symmetric matrix.
void kernel_gaussian_full_symm(
    const std::vector<double> &x, const std::vector<double> &dx, const std::vector<int> &q,
    const std::vector<int> &n, int nm, int max_atoms, int rep_size, int naq, double sigma,
    double *kernel_out  // ((nm+naq) x (nm+naq)), row-major
);

// Full combined energy+force kernel (symmetric, RFP output).
// Output: 1-D array of length BIG*(BIG+1)/2, BIG=nm+naq, TRANSR='N', UPLO='U'.
void kernel_gaussian_full_symm_rfp(
    const std::vector<double> &x, const std::vector<double> &dx, const std::vector<int> &q,
    const std::vector<int> &n, int nm, int max_atoms, int rep_size, int naq, double sigma,
    double *arf  // length (nm+naq)*(nm+naq+1)/2
);

// ============================================================================
// J^T·α Trick for Local (FCHL19) Hessian Kernel
// ============================================================================
//
// Pre-compute descriptor-space force coefficients for efficient inference.
// Reduces Hessian-matvec cost from O(naq2·naq2·rep) per training atom to O(rep).
//
// Input:
//   dx2:      (nm2, max_atoms2, rep_size, 3*max_atoms2), training Jacobians
//   q2:       (nm2, max_atoms2), atomic labels
//   n2:       (nm2), active atom counts per molecule
//   alpha:    (naq2,) where naq2 = 3*sum(n2), force coefficients in Cartesian space
//
// Output:
//   alpha_desc: (nm2, max_atoms2, rep_size), descriptor-space coefficients
//              alpha_desc[b,i2,:] = dx2[b,i2,:,:3*n2[b]]^T @ alpha[offs2[b]:offs2[b]+3*n2[b]]
//
// Call once after training, then use with kernel_gaussian_local_hessian_matvec.
void kernel_gaussian_local_compute_alpha_desc(
    const std::vector<double> &dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
    const std::vector<int> &q2,      // (nm2, max_atoms2)
    const std::vector<int> &n2,      // (nm2)
    int nm2, int max_atoms2, int rep_size,
    const double *alpha,  // (naq2,) where naq2 = 3*sum(n2)
    double *alpha_desc    // (nm2, max_atoms2, rep_size) output, row-major
);

// Efficient Hessian-matvec using J^T·α descriptor-space trick.
//
// Computes F = H @ alpha where H is the local Hessian kernel, without forming H.
// Cost: O(nm1·naq2·rep + naq1) vs O(naq1·naq2·rep + naq1) for full matrix.
//
// Algorithm (per query molecule a, per query atom i1):
//   1. For each matching training atom (b, i2) with same label:
//      - Compute d = x1[a,i1] - x2[b,i2]
//      - Accumulate in descriptor space: G[i1,:] += kernel_scalars(||d||²) * α̃[b,i2]
//   2. Back-project: F[a,i1,:] = dx1[a,i1]^T @ G[i1,:]
//
// Inputs:
//   x1, x2:      (nm1/nm2, max_atoms1/2, rep_size), atomic descriptor vectors
//   dx1:         (nm1, max_atoms1, rep_size, 3*max_atoms1), query Jacobians
//   q1, q2:      (nm1/nm2, max_atoms1/2), atomic labels (for matching)
//   n1, n2:      (nm1/nm2), active atom counts
//   alpha_desc:  (nm2, max_atoms2, rep_size), pre-computed via compute_alpha_desc
//   sigma:       Gaussian width parameter
//
// Output:
//   F: (naq1,) where naq1 = 3*sum(n1), forces in Cartesian coordinates
void kernel_gaussian_local_hessian_matvec(
    const std::vector<double> &x1,    // (nm1, max_atoms1, rep_size)
    const std::vector<double> &dx1,   // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    const std::vector<double> &x2,    // (nm2, max_atoms2, rep_size)
    const std::vector<double> &alpha_desc,  // (nm2, max_atoms2, rep_size)
    const std::vector<int> &q1,       // (nm1, max_atoms1)
    const std::vector<int> &q2,       // (nm2, max_atoms2)
    const std::vector<int> &n1,       // (nm1)
    const std::vector<int> &n2,       // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must equal 3 * sum(n1)
    double sigma,
    double *F  // (naq1,) output, row-major
);

}  // namespace fchl19
}  // namespace kf
