#pragma once
#include "array_view.hpp"

#include <cstddef>
#include <vector>

namespace kf {
namespace fchl19 {

using ::kf::ConstArray;

// Local (atom-pair-wise) Gaussian kernel functions

void kernel_gaussian(
    ConstArray<double> x1, ConstArray<double> x2, ConstArray<int> q1, ConstArray<int> q2,
    ConstArray<int> n1, ConstArray<int> n2, int nm1, int nm2, int max_atoms1, int max_atoms2,
    int rep_size, double sigma, double *kernel_out
);

void kernel_gaussian_symm_rfp(
    ConstArray<double> x, ConstArray<int> q, ConstArray<int> n, int nm, int max_atoms, int rep_size,
    double sigma, double *arf
);

void kernel_gaussian_symm(
    ConstArray<double> x, ConstArray<int> q, ConstArray<int> n, int nm, int max_atoms, int rep_size,
    double sigma, double *kernel_out
);

void kernel_gaussian_jacobian(
    ConstArray<double> x1,   // (nm1, max_atoms1, rep)
    ConstArray<double> x2,   // (nm2, max_atoms2, rep)
    ConstArray<double> dX2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    ConstArray<int> q1,      // (nm1, max_atoms1)
    ConstArray<int> q2,      // (nm2, max_atoms2)
    ConstArray<int> n1,      // (nm1)
    ConstArray<int> n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq2,  // should equal 3 * sum(n2)
    double sigma,
    double *kernel_out  // (nm1, naq2) row-major
);

// Transposed Jacobian kernel: Jacobians on set-1 side (dX1).
// Output shape: (naq1, nm2), where naq1 = 3 * sum(n1).
// Property: kernel_gaussian_jacobian_t(X1, dX1, X2, ...) ==
//           -kernel_gaussian_jacobian(X2, X1, dX1, ...).T
void kernel_gaussian_jacobian_t(
    ConstArray<double> x1,   // (nm1, max_atoms1, rep)
    ConstArray<double> x2,   // (nm2, max_atoms2, rep)
    ConstArray<double> dX1,  // (nm1, max_atoms1, rep, 3*max_atoms1)
    ConstArray<int> q1,      // (nm1, max_atoms1)
    ConstArray<int> q2,      // (nm2, max_atoms2)
    ConstArray<int> n1,      // (nm1)
    ConstArray<int> n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // should equal 3 * sum(n1)
    double sigma,
    double *kernel_out  // (naq1, nm2) row-major
);

void kernel_gaussian_hessian(
    ConstArray<double> x1,   // (nm1, max_atoms1, rep_size)
    ConstArray<double> x2,   // (nm2, max_atoms2, rep_size)
    ConstArray<double> dx1,  // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    ConstArray<double> dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
    ConstArray<int> q1,      // (nm1, max_atoms1)
    ConstArray<int> q2,      // (nm2, max_atoms2)
    ConstArray<int> n1,      // (nm1)
    ConstArray<int> n2,      // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must equal 3 * sum_a n1[a]
    int naq2,  // must equal 3 * sum_b n2[b]
    double sigma,
    double *kernel_out  // (naq2, naq1) row-major
);

void kernel_gaussian_hessian_symm(
    ConstArray<double> x,   // (nm, max_atoms, rep_size)
    ConstArray<double> dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    ConstArray<int> q,      // (nm, max_atoms)
    ConstArray<int> n,      // (nm)
    int nm, int max_atoms, int rep_size,
    int naq,  // must be 3 * sum_m n[m]
    double sigma,
    double *kernel_out  // (naq, naq) row-major
);

void kernel_gaussian_hessian_symm_rfp(
    ConstArray<double> x,   // (nm, max_atoms, rep_size)
    ConstArray<double> dx,  // (nm, max_atoms, rep_size, 3*max_atoms)
    ConstArray<int> q,      // (nm, max_atoms)
    ConstArray<int> n,      // (nm)
    int nm, int max_atoms, int rep_size,
    int naq,  // must be 3 * sum_m n[m]
    double sigma,
    double *arf  // length naq*(naq+1)/2, RFP TRANSR='N', UPLO='U'
);

// Full combined energy+force kernel (asymmetric).
// Output: ((nm1+naq1) x (nm2+naq2)) row-major matrix.
// Blocks: [0:nm1,0:nm2]=scalar, [0:nm1,nm2:]=jac_t, [nm1:,0:nm2]=jac, [nm1:,nm2:]=hessian.
void kernel_gaussian_full(
    ConstArray<double> x1, ConstArray<double> x2, ConstArray<double> dx1, ConstArray<double> dx2,
    ConstArray<int> q1, ConstArray<int> q2, ConstArray<int> n1, ConstArray<int> n2, int nm1,
    int nm2, int max_atoms1, int max_atoms2, int rep_size, int naq1, int naq2, double sigma,
    double *kernel_out  // ((nm1+naq1) x (nm2+naq2)), row-major
);

// Full combined energy+force kernel (symmetric, full square output).
// Output: ((nm+naq) x (nm+naq)) row-major, fully filled symmetric matrix.
void kernel_gaussian_full_symm(
    ConstArray<double> x, ConstArray<double> dx, ConstArray<int> q, ConstArray<int> n, int nm,
    int max_atoms, int rep_size, int naq, double sigma,
    double *kernel_out  // ((nm+naq) x (nm+naq)), row-major
);

// Full combined energy+force kernel (symmetric, RFP output).
// Output: 1-D array of length BIG*(BIG+1)/2, BIG=nm+naq, TRANSR='N', UPLO='U'.
void kernel_gaussian_full_symm_rfp(
    ConstArray<double> x, ConstArray<double> dx, ConstArray<int> q, ConstArray<int> n, int nm,
    int max_atoms, int rep_size, int naq, double sigma,
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
    ConstArray<double> dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
    ConstArray<int> q2,      // (nm2, max_atoms2)
    ConstArray<int> n2,      // (nm2)
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
    ConstArray<double> x1,          // (nm1, max_atoms1, rep_size)
    ConstArray<double> dx1,         // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    ConstArray<double> x2,          // (nm2, max_atoms2, rep_size)
    ConstArray<double> alpha_desc,  // (nm2, max_atoms2, rep_size)
    ConstArray<int> q1,             // (nm1, max_atoms1)
    ConstArray<int> q2,             // (nm2, max_atoms2)
    ConstArray<int> n1,             // (nm1)
    ConstArray<int> n2,             // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must equal 3 * sum(n1)
    double sigma,
    double *F  // (naq1,) output, row-major
);

// Efficient energy prediction via local scalar kernel matvec (Kw).
//
// Computes E[a] = Σ_b K[a,b] · alpha_E[b] without forming K.
// K[a,b] is the local (atom-pairwise, matching-label) Gaussian kernel.
//
// Shapes: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep), alpha_E(nm2,) -> E_out(nm1,)
void kernel_gaussian_local_matvec(
    ConstArray<double> x1,  // (nm1, max_atoms1, rep_size)
    ConstArray<double> x2,  // (nm2, max_atoms2, rep_size)
    const double *alpha_E,  // (nm2,)
    ConstArray<int> q1,     // (nm1, max_atoms1)
    ConstArray<int> q2,     // (nm2, max_atoms2)
    ConstArray<int> n1,     // (nm1)
    ConstArray<int> n2,     // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size, double sigma,
    double *E_out  // (nm1,)
);

// Efficient energy prediction via local Jacobian kernel matvec using J^T·α trick.
//
// Computes E[a] = Σ_{b,r} K_jac[a, offs2[b]+r] · alpha_F[offs2[b]+r] without forming K_jac.
// Requires alpha_desc = kernel_gaussian_local_compute_alpha_desc(dX2, q2, n2, ..., alpha_F).
//
// Shapes: x1(nm1,max_atoms1,rep), x2(nm2,max_atoms2,rep),
//         alpha_desc(nm2,max_atoms2,rep) -> E_out(nm1,)
void kernel_gaussian_local_jacobian_t_matvec(
    ConstArray<double> x1,          // (nm1, max_atoms1, rep_size)
    ConstArray<double> x2,          // (nm2, max_atoms2, rep_size)
    ConstArray<double> alpha_desc,  // (nm2, max_atoms2, rep_size) precomputed
    ConstArray<int> q1,             // (nm1, max_atoms1)
    ConstArray<int> q2,             // (nm2, max_atoms2)
    ConstArray<int> n1,             // (nm1)
    ConstArray<int> n2,             // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size, double sigma,
    double *E_out  // (nm1,) output energies
);

// Efficient combined energy+force prediction via full local kernel matvec using J^T·α trick.
//
// Given alpha_E(nm2,) and alpha_desc_F = compute_alpha_desc(dX2, alpha_F),
// computes E(nm1,) and F(naq1,) predictions simultaneously.
//
// Shapes: x1(nm1,max_atoms1,rep), dx1(nm1,max_atoms1,rep,3*max_atoms1),
//         x2(nm2,max_atoms2,rep), alpha_E(nm2,), alpha_desc_F(nm2,max_atoms2,rep)
//         -> E_out(nm1,), F_out(naq1,)
void kernel_gaussian_local_full_matvec(
    ConstArray<double> x1,            // (nm1, max_atoms1, rep_size)
    ConstArray<double> dx1,           // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    ConstArray<double> x2,            // (nm2, max_atoms2, rep_size)
    ConstArray<double> alpha_desc_F,  // (nm2, max_atoms2, rep_size) precomputed
    const double *alpha_E,            // (nm2,) energy coefficients
    ConstArray<int> q1,               // (nm1, max_atoms1)
    ConstArray<int> q2,               // (nm2, max_atoms2)
    ConstArray<int> n1,               // (nm1)
    ConstArray<int> n2,               // (nm2)
    int nm1, int nm2, int max_atoms1, int max_atoms2, int rep_size,
    int naq1,  // must equal 3 * sum(n1)
    double sigma,
    double *E_out,  // (nm1,) output energies
    double *F_out   // (naq1,) output forces
);

}  // namespace fchl19
}  // namespace kf
