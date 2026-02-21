// Own header
#include "fchl19_repr.hpp"

// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Third-party libraries
#include <omp.h>

// Project headers
#include "constants.hpp"

namespace kf {
namespace fchl19 {

// ==================== Helper Functions (file-local) ====================

// Compute the expected representation size per atom
std::size_t compute_rep_size(std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
                             std::size_t nabasis) {
    const std::size_t two_body = nelements * nbasis2;
    const std::size_t n_pairs_symmetric = nelements * (nelements + 1) / 2;  // unordered pairs
    const std::size_t three_body = n_pairs_symmetric * nbasis3 * nabasis;
    return two_body + three_body;
}

// Flat 2D indexing: (i, j) -> i*ncols + j
static inline std::size_t idx2(std::size_t i, std::size_t j, std::size_t ncols) {
    return i * ncols + j;
}

// Gradient indexing: (i, feat, a, d) -> flat index for shape (natoms, rep_size, natoms, 3)
static inline std::size_t gidx(std::size_t i, std::size_t feat, std::size_t a, std::size_t d,
                               std::size_t rep_size, std::size_t natoms) {
    return (((i * rep_size + feat) * natoms + a) * 3 + d);
}

// Symmetric pair index for elements p, q
static inline std::size_t pair_index(std::size_t nelements, int p, int q) {
    if (p > q)
        std::swap(p, q);
    long long llp = p, llq = q, llN = static_cast<long long>(nelements);
    long long idx = -llp * (llp + 1) / 2 + llq + llN * llp;
    return static_cast<std::size_t>(idx);
}

// Half-cosine cutoff decay: 0.5*(cos(pi*r*invrc) + 1)
static void decay_matrix(const std::vector<double> &rmat, double invrc, std::size_t natoms,
                         std::vector<double> &out) {
    out.resize(natoms * natoms);
    const double f = M_PI * invrc;
    for (std::size_t i = 0; i < natoms * natoms; ++i) {
        out[i] = 0.5 * (std::cos(f * rmat[i]) + 1.0);
    }
}

// Compute full pairwise distance matrix (natoms x natoms, row-major)
static std::vector<double> pairwise_distances(const std::vector<double> &coords,
                                               std::size_t natoms) {
    if (coords.size() != natoms * 3)
        throw std::invalid_argument("coords.size() must equal natoms*3");
    std::vector<double> D(natoms * natoms, 0.0);
    for (std::size_t i = 0; i < natoms; ++i) {
        const double *ri = &coords[3 * i];
        for (std::size_t j = i + 1; j < natoms; ++j) {
            const double *rj = &coords[3 * j];
            double dx = rj[0] - ri[0];
            double dy = rj[1] - ri[1];
            double dz = rj[2] - ri[2];
            double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            D[idx2(i, j, natoms)] = d;
            D[idx2(j, i, natoms)] = d;
        }
    }
    return D;
}

// ==================== FCHL19 Representation Functions ====================

void generate_fchl_acsf(const std::vector<double> &coords, const std::vector<int> &nuclear_z,
                        const std::vector<int> &elements, const std::vector<double> &Rs2,
                        const std::vector<double> &Rs3, const std::vector<double> &Ts, double eta2,
                        double eta3, double zeta, double rcut, double acut, double two_body_decay,
                        double three_body_decay, double three_body_weight,
                        std::vector<double> &rep) {
    const size_t natoms = nuclear_z.size();
    if (coords.size() != natoms * 3)
        throw std::invalid_argument("coords size must be natoms*3");

    const size_t nelements = elements.size();
    const size_t nbasis2 = Rs2.size();
    const size_t nbasis3 = Rs3.size();
    const size_t nabasis = Ts.size();
    if (nabasis % 2 != 0)
        throw std::invalid_argument("Ts.size() (nabasis) must be even");

    const size_t rep_size = compute_rep_size(nelements, nbasis2, nbasis3, nabasis);
    rep.assign(natoms * rep_size, 0.0);

    // Map Z -> element channel index [0..nelements)
    std::unordered_map<int, int> z2idx;
    z2idx.reserve(nelements * 2);
    for (size_t j = 0; j < nelements; ++j)
        z2idx[elements[j]] = static_cast<int>(j);

    std::vector<int> elem_of_atom(natoms, -1);
    for (size_t i = 0; i < natoms; ++i) {
        auto it = z2idx.find(nuclear_z[i]);
        if (it == z2idx.end())
            throw std::runtime_error("nuclear_z contains an element not present in elements");
        elem_of_atom[i] = it->second;  // 0-based
    }

    // Distances and decays
    const std::vector<double> D = pairwise_distances(coords, natoms);

    std::vector<double> rdecay2, rdecay3;
    decay_matrix(D, (rcut > 0 ? 1.0 / rcut : 0.0), natoms, rdecay2);
    decay_matrix(D, (acut > 0 ? 1.0 / acut : 0.0), natoms, rdecay3);

    // Precompute angular weights for harmonics o = 1,3,5,... (length nabasis/2)
    const size_t n_harm = nabasis / 2;
    std::vector<double> ang_w(n_harm);  // 2*exp(-0.5*(zeta*o)^2)
    std::vector<int> ang_o(n_harm);     // o values
    for (size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2 * l + 1);  // 1,3,5,...
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }

    // Precompute log(Rs2) and 1/Rs2
    std::vector<double> log_Rs2(nbasis2, 0.0);
    std::vector<double> inv_Rs2(nbasis2, 0.0);
    for (size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0)
            throw std::invalid_argument("All Rs2 must be > 0");
        log_Rs2[k] = std::log(Rs2[k]);
        inv_Rs2[k] = 1.0 / Rs2[k];
    }

// ---------------- Two-body term ----------------
// Thread-local buffers + reduction (no locks needed)
#pragma omp parallel
    {
        std::vector<double> rep_local(natoms * rep_size, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (long long ii = 0; ii < static_cast<long long>(natoms); ++ii) {
            const size_t i = static_cast<size_t>(ii);
            const int elem_i = elem_of_atom[i];
            for (size_t j = i + 1; j < natoms; ++j) {
                const int elem_j = elem_of_atom[j];
                const double rij = D[idx2(i, j, natoms)];
                if (rij > rcut)
                    continue;

                const double rij2 = rij * rij;
                const double t = eta2 / std::max(rij2, kf::EPS);
                const double log1pt = std::log1p(t);
                const double sigma = std::sqrt(std::max(log1pt, 0.0));
                if (sigma < kf::EPS)
                    continue;
                const double mu = std::log(rij) - 0.5 * log1pt;
                const double decay_ij = rdecay2[idx2(i, j, natoms)];
                const double inv_pref =
                    decay_ij / (sigma * kf::SQRT_2PI * std::pow(rij, two_body_decay));
                const double inv_sigma_sq = 1.0 / (sigma * sigma);

                const size_t ch_j = static_cast<size_t>(elem_j) * nbasis2;
                const size_t ch_i = static_cast<size_t>(elem_i) * nbasis2;

                for (size_t k = 0; k < nbasis2; ++k) {
                    const double dlog = log_Rs2[k] - mu;
                    const double g = std::exp(-0.5 * dlog * dlog * inv_sigma_sq);
                    const double val = inv_pref * g * inv_Rs2[k];
                    rep_local[idx2(i, ch_j + k, rep_size)] += val;
                    rep_local[idx2(j, ch_i + k, rep_size)] += val;
                }
            }
        }

// Reduce thread-local buffers into global rep
#pragma omp critical
        {
            for (size_t idx = 0; idx < natoms * rep_size; ++idx) {
                rep[idx] += rep_local[idx];
            }
        }
    }

    // ---------------- Three-body term ----------------
    const size_t three_offset = nelements * nbasis2;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < natoms; ++i) {
        // Thread-local angular buffer (avoid heap alloc per triplet)
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i)
                continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut)
                continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            // Precompute normalized edge vector eij = (A-B)/rij
            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (size_t k = j + 1; k < natoms; ++k) {
                if (k == i)
                    continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut)
                    continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                // Normalized edge vector eik = (C-B)/rik
                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                // Normalized edge vector ejk = (C-A)/rjk
                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                // Cosines from dot products of normalized vectors (no acos needed)
                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);  // dot(-eij, ejk)
                const double cos_k =
                    eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;  // dot(eik, ejk) = dot(-eik, -ejk)

                // ATM factor
                const double denom = std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // Angular basis via Chebyshev recurrence (no acos/cos/sin calls)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                // o=1: cos(1*angle) = cos_i, sin(1*angle) = sin_i
                angular[0] = ang_w[0] * cos_i;
                if (nabasis > 1)
                    angular[1] = ang_w[0] * sin_i;

                // Higher harmonics o=3,5,... via Chebyshev recurrence
                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0;      // cos(0), sin(0)
                    double cn_1 = cos_i, sn_1 = sin_i;  // cos(θ), sin(θ)
                    size_t harm_stored = 1;
                    const int max_o = ang_o[n_harm - 1];
                    for (int n = 2; n <= max_o; ++n) {
                        const double cn = two_cos * cn_1 - cn_2;
                        const double sn = two_cos * sn_1 - sn_2;
                        cn_2 = cn_1;
                        sn_2 = sn_1;
                        cn_1 = cn;
                        sn_1 = sn;
                        if (n == ang_o[harm_stored]) {
                            angular[2 * harm_stored] = ang_w[harm_stored] * cn;
                            angular[2 * harm_stored + 1] = ang_w[harm_stored] * sn;
                            if (++harm_stored >= n_harm)
                                break;
                        }
                    }
                }

                // Element pair index and accumulation
                const size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const size_t base = three_offset + pair_idx * (nbasis3 * nabasis);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double rbar = 0.5 * (rij + rik);

                for (size_t l = 0; l < nbasis3; ++l) {
                    const double dr = rbar - Rs3[l];
                    const double radial_l = std::exp(-eta3 * dr * dr) * decay_ij * decay_ik;
                    const double scale = radial_l * ksi3;
                    const size_t z = base + l * nabasis;
                    double *dst = &rep[idx2(i, z, rep_size)];
                    for (size_t t = 0; t < nabasis; ++t)
                        dst[t] += angular[t] * scale;
                }
            }
        }
    }
}

// ##################################
// # FCHL19 REPRESENTATION JACOBIAN #
// ##################################

// Computes representation and its Jacobian wrt coordinates.
// rep:  shape (natoms, rep_size) in row-major (flattened to size natoms*rep_size)
// grad: shape (natoms, rep_size, natoms, 3) flattened to size natoms*rep_size*natoms*3
void generate_fchl_acsf_and_gradients(
    const std::vector<double> &coords, const std::vector<int> &nuclear_z,
    const std::vector<int> &elements, const std::vector<double> &Rs2,
    const std::vector<double> &Rs3, const std::vector<double> &Ts, double eta2, double eta3,
    double zeta, double rcut, double acut, double two_body_decay, double three_body_decay,
    double three_body_weight, std::vector<double> &rep, std::vector<double> &grad) {
    const size_t natoms = nuclear_z.size();
    if (coords.size() != natoms * 3)
        throw std::invalid_argument("coords size must be natoms*3");
    const size_t nelements = elements.size();
    const size_t nbasis2 = Rs2.size();
    const size_t nbasis3 = Rs3.size();
    const size_t nabasis = Ts.size();
    const size_t rep_size = compute_rep_size(nelements, nbasis2, nbasis3, nabasis);

    rep.assign(natoms * rep_size, 0.0);
    grad.assign(natoms * rep_size * natoms * 3, 0.0);

    // Map Z->element index
    std::unordered_map<int, int> z2idx;
    z2idx.reserve(nelements * 2);
    for (size_t j = 0; j < nelements; ++j)
        z2idx[elements[j]] = (int)j;
    std::vector<int> elem_of_atom(natoms, -1);
    for (size_t i = 0; i < natoms; ++i) {
        auto it = z2idx.find(nuclear_z[i]);
        if (it == z2idx.end())
            throw std::runtime_error("Unknown element in nuclear_z");
        elem_of_atom[i] = it->second;
    }

    // Distances and powers
    const std::vector<double> D = pairwise_distances(coords, natoms);
    std::vector<double> D2(natoms * natoms, 0.0), invD(natoms * natoms, 0.0),
        invD2(natoms * natoms, 0.0);
    for (size_t i = 0; i < natoms; ++i) {
        for (size_t j = i + 1; j < natoms; ++j) {
            double rij = D[idx2(i, j, natoms)];
            double rij2 = std::max(rij * rij, kf::EPS);
            double invr = 1.0 / rij;
            double invr2 = 1.0 / rij2;
            D2[idx2(i, j, natoms)] = D2[idx2(j, i, natoms)] = rij2;
            invD[idx2(i, j, natoms)] = invD[idx2(j, i, natoms)] = invr;
            invD2[idx2(i, j, natoms)] = invD2[idx2(j, i, natoms)] = invr2;
        }
    }

    // Decays
    std::vector<double> rdecay2, rdecay3;
    if (rcut > 0)
        decay_matrix(D, 1.0 / rcut, natoms, rdecay2);
    else
        rdecay2.assign(natoms * natoms, 1.0);
    if (acut > 0)
        decay_matrix(D, 1.0 / acut, natoms, rdecay3);
    else
        rdecay3.assign(natoms * natoms, 1.0);

    // Precompute log(Rs2)
    std::vector<double> log_Rs2(nbasis2);
    for (size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0)
            throw std::invalid_argument("Rs2 must be >0");
        log_Rs2[k] = std::log(Rs2[k]);
    }

    // ---------------- Two-body: values + gradients ----------------
    for (size_t i = 0; i < natoms; ++i) {
        const int elem_i = elem_of_atom[i];
        for (size_t j = i + 1; j < natoms; ++j) {
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i, j, natoms)];
            if (rij > rcut)
                continue;
            const double invr = invD[idx2(i, j, natoms)];
            const double invr2 = invD2[idx2(i, j, natoms)];
            const double s2 = std::log1p(eta2 * invr2);  // sigma^2
            const double sigma = std::sqrt(std::max(s2, 0.0));
            if (sigma < kf::EPS)
                continue;
            const double mu = std::log(rij) - 0.5 * s2;
            const double decay_ij = rdecay2[idx2(i, j, natoms)];
            const double scaling = std::pow(rij, -two_body_decay);
            const double inv_pref_common = 1.0 / (sigma * std::sqrt(2.0 * M_PI));

            // radial_base[k] and radial[k]
            std::vector<double> radial_base(nbasis2), radial(nbasis2), exp_ln(nbasis2);
            for (size_t k = 0; k < nbasis2; ++k) {
                const double dlog = log_Rs2[k] - mu;
                const double g = std::exp(-0.5 * dlog * dlog / s2);
                exp_ln[k] = g * std::sqrt(2.0);  // as in Fortran (exp_ln used with sqrt(2))
                radial_base[k] = (inv_pref_common / Rs2[k]) * g;
                radial[k] = radial_base[k] * scaling * decay_ij;
            }

            // accumulate values to rows i and j, channels of counterpart element
            const size_t ch_j = (size_t)elem_j * nbasis2;
            const size_t ch_i = (size_t)elem_i * nbasis2;
            for (size_t k = 0; k < nbasis2; ++k) {
                rep[idx2(i, ch_j + k, rep_size)] += radial[k];
                rep[idx2(j, ch_i + k, rep_size)] += radial[k];
            }

            // gradient contributions wrt coordinates of i and j
            const double exp_s2 = std::exp(s2);
            const double sqrt_exp_s2 = std::sqrt(exp_s2);
            for (int t = 0; t < 3; ++t) {
                const double dx = -(coords[3 * i + t] - coords[3 * j + t]);  // -(ri - rj)
                // dscal = d(1/rij^p)/d x_i_t (with dx = rj_t - ri_t)
                const double dscal = two_body_decay * dx * std::pow(rij, -(two_body_decay + 2.0));
                const double ddecay = dx * 0.5 * M_PI *
                                      std::sin(M_PI * rij * (rcut > 0 ? 1.0 / rcut : 0.0)) *
                                      (rcut > 0 ? 1.0 / rcut : 0.0) * invr;

                for (size_t k = 0; k < nbasis2; ++k) {
                    const double L = log_Rs2[k] - mu;
                    // term inside Fortran's big bracket
                    const double term1 =
                        L * (-dx * (rij * rij * exp_s2 + eta2) / std::pow(rij * sqrt_exp_s2, 3)) *
                        (sqrt_exp_s2 / (s2 * rij));
                    const double term2 =
                        (L * L) * eta2 * dx / ((s2 * s2) * std::pow(rij, 4) * exp_s2);
                    const double A =
                        (term1 + term2) * (exp_ln[k] / (Rs2[k] * sigma * std::sqrt(M_PI) * 2.0)) -
                        (exp_ln[k] * eta2 * dx) / (Rs2[k] * (s2 * std::sqrt(M_PI)) * sigma *
                                                   std::pow(rij, 4) * exp_s2 * 2.0);
                    double part = A * scaling * decay_ij + radial_base[k] * dscal * decay_ij +
                                  radial_base[k] * scaling * ddecay;

                    // write to grad tensor
                    size_t feat_i = ch_j + k;
                    size_t feat_j = ch_i + k;
                    grad[gidx(i, feat_i, i, (size_t)t, rep_size, natoms)] += part;
                    grad[gidx(i, feat_i, j, (size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, j, (size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, i, (size_t)t, rep_size, natoms)] += part;
                }
            }
        }
    }

    // ---------------- Three-body: values + gradients ----------------
    const size_t three_offset = nelements * nbasis2;
    const size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;
    const double ang_w_pre = std::exp(-0.5 * zeta * zeta) * 2.0;
    const double tbd_over_w_pre =
        (three_body_weight != 0.0) ? (three_body_decay / three_body_weight) : 0.0;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < natoms; ++i) {
        // per-center buffers for the 3-body block
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        // Pre-allocated scratch buffers (avoid heap alloc per triplet)
        std::vector<double> radial(nbasis3), d_radial(nbasis3);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];
        for (size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i)
                continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut)
                continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            // Precompute normalized edge vector eij = (A-B)/rij
            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;

            // Precompute decay derivative scalar for ij
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (size_t k = j + 1; k < natoms; ++k) {
                if (k == i)
                    continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut)
                    continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                // Normalized edge vectors
                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                // Cosines from dot products (no acos/cos/sin needed)
                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // Raw dot product (A-B)·(C-B) — needed for angle gradient
                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);

                // Radial parts per Rs3 center (reuse pre-allocated buffer)
                for (size_t l = 0; l < nbasis3; ++l) {
                    const double rbar = 0.5 * (rij + rik) - Rs3[l];
                    const double base = std::exp(-eta3 * rbar * rbar);
                    d_radial[l] = base * eta3 * rbar;
                    radial[l] = base;
                }

                // Angular basis from cos_i, sin_i (no trig calls)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w_pre * cos_i;
                d_angular[0] = ang_w_pre * sin_i;
                if (nabasis >= 2) {
                    angular[1] = ang_w_pre * sin_i;
                    d_angular[1] = -ang_w_pre * cos_i;
                }

                // d angle / d coordinates (denominator)
                const double denom = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                // direction vectors multiplying d_angular
                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);

                // Decay and decay derivatives
                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                // d_ijdecay[t] = s_ij * (B[t] - A[t])
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double s_ik = -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);

                // ATM factor and its pieces
                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;
                const double atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik;
                const double atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk;
                const double atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik;

                const double vi = dot;  // (A-B)·(C-B)
                const double vj =
                    (Cx - Ax) * (Bx - Ax) + (Cy - Ay) * (By - Ay) + (Cz - Az) * (Bz - Az);
                const double vk =
                    (Bx - Cx) * (Ax - Cx) + (By - Cy) * (Ay - Cy) + (Bz - Cz) * (Az - Cz);

                // ATM derivative direction vectors (3 atoms x 3 derivative sources)
                const double d_atm_ii0 =
                    2 * Bx - Ax - Cx - vi * ((Bx - Ax) * invrij2 + (Bx - Cx) * invrik2);
                const double d_atm_ii1 =
                    2 * By - Ay - Cy - vi * ((By - Ay) * invrij2 + (By - Cy) * invrik2);
                const double d_atm_ii2 =
                    2 * Bz - Az - Cz - vi * ((Bz - Az) * invrij2 + (Bz - Cz) * invrik2);
                const double d_atm_ij0 = Cx - Ax - vj * (Bx - Ax) * invrij2;
                const double d_atm_ij1 = Cy - Ay - vj * (By - Ay) * invrij2;
                const double d_atm_ij2 = Cz - Az - vj * (Bz - Az) * invrij2;
                const double d_atm_ik0 = Ax - Cx - vk * (Bx - Cx) * invrik2;
                const double d_atm_ik1 = Ay - Cy - vk * (By - Cy) * invrik2;
                const double d_atm_ik2 = Az - Cz - vk * (Bz - Cz) * invrik2;

                const double d_atm_ji0 = Cx - Bx - vi * (Ax - Bx) * invrij2;
                const double d_atm_ji1 = Cy - By - vi * (Ay - By) * invrij2;
                const double d_atm_ji2 = Cz - Bz - vi * (Az - Bz) * invrij2;
                const double d_atm_jj0 =
                    2 * Ax - Bx - Cx - vj * ((Ax - Bx) * invrij2 + (Ax - Cx) * invrjk * invrjk);
                const double d_atm_jj1 =
                    2 * Ay - By - Cy - vj * ((Ay - By) * invrij2 + (Ay - Cy) * invrjk * invrjk);
                const double d_atm_jj2 =
                    2 * Az - Bz - Cz - vj * ((Az - Bz) * invrij2 + (Az - Cz) * invrjk * invrjk);
                const double d_atm_jk0 = Bx - Cx - vk * (Ax - Cx) * invrjk * invrjk;
                const double d_atm_jk1 = By - Cy - vk * (Ay - Cy) * invrjk * invrjk;
                const double d_atm_jk2 = Bz - Cz - vk * (Az - Cz) * invrjk * invrjk;

                const double d_atm_ki0 = Ax - Bx - vi * (Cx - Bx) * invrik2;
                const double d_atm_ki1 = Ay - By - vi * (Cy - By) * invrik2;
                const double d_atm_ki2 = Az - Bz - vi * (Cz - Bz) * invrik2;
                const double d_atm_kj0 = Bx - Ax - vj * (Cx - Ax) * invrjk * invrjk;
                const double d_atm_kj1 = By - Ay - vj * (Cy - Ay) * invrjk * invrjk;
                const double d_atm_kj2 = Bz - Az - vj * (Cz - Az) * invrjk * invrjk;
                const double d_atm_kk0 =
                    2 * Cx - Ax - Bx - vk * ((Cx - Ax) * invrjk * invrjk + (Cx - Bx) * invrik2);
                const double d_atm_kk1 =
                    2 * Cy - Ay - By - vk * ((Cy - Ay) * invrjk * invrjk + (Cy - By) * invrik2);
                const double d_atm_kk2 =
                    2 * Cz - Az - Bz - vk * ((Cz - Az) * invrjk * invrjk + (Cz - Bz) * invrik2);

                const double atm_tbd = atm * tbd_over_w_pre;
                const double d_extra_i0 = ((Ax - Bx) * invrij2 + (Cx - Bx) * invrik2) * atm_tbd;
                const double d_extra_i1 = ((Ay - By) * invrij2 + (Cy - By) * invrik2) * atm_tbd;
                const double d_extra_i2 = ((Az - Bz) * invrij2 + (Cz - Bz) * invrik2) * atm_tbd;
                const double d_extra_j0 =
                    ((Bx - Ax) * invrij2 + (Cx - Ax) * invrjk * invrjk) * atm_tbd;
                const double d_extra_j1 =
                    ((By - Ay) * invrij2 + (Cy - Ay) * invrjk * invrjk) * atm_tbd;
                const double d_extra_j2 =
                    ((Bz - Az) * invrij2 + (Cz - Az) * invrjk * invrjk) * atm_tbd;
                const double d_extra_k0 =
                    ((Ax - Cx) * invrjk * invrjk + (Bx - Cx) * invrik2) * atm_tbd;
                const double d_extra_k1 =
                    ((Ay - Cy) * invrjk * invrjk + (By - Cy) * invrik2) * atm_tbd;
                const double d_extra_k2 =
                    ((Az - Cz) * invrjk * invrjk + (Bz - Cz) * invrik2) * atm_tbd;

                // Element pair index
                const size_t pair_idx0 = pair_index(nelements, elem_j, elem_k);
                const size_t base = pair_idx0 * (nbasis3 * nabasis);

                // Precompute per-dimension gradient building blocks
                const double inv_denom = 1.0 / denom;
                // d_ang direction / denom for each atom and dim
                const double dai0 = d_ang_d_i0 * inv_denom, dai1 = d_ang_d_i1 * inv_denom,
                             dai2 = d_ang_d_i2 * inv_denom;
                const double daj0 = d_ang_d_j0 * inv_denom, daj1 = d_ang_d_j1 * inv_denom,
                             daj2 = d_ang_d_j2 * inv_denom;
                const double dak0 = d_ang_d_k0 * inv_denom, dak1 = d_ang_d_k1 * inv_denom,
                             dak2 = d_ang_d_k2 * inv_denom;

                // ATM derivatives (full, including three_body_weight)
                const double atmi0 =
                    (atm_i * d_atm_ii0 + atm_j * d_atm_ij0 + atm_k * d_atm_ik0 + d_extra_i0) *
                    three_body_weight;
                const double atmi1 =
                    (atm_i * d_atm_ii1 + atm_j * d_atm_ij1 + atm_k * d_atm_ik1 + d_extra_i1) *
                    three_body_weight;
                const double atmi2 =
                    (atm_i * d_atm_ii2 + atm_j * d_atm_ij2 + atm_k * d_atm_ik2 + d_extra_i2) *
                    three_body_weight;
                const double atmj0 =
                    (atm_i * d_atm_ji0 + atm_j * d_atm_jj0 + atm_k * d_atm_jk0 + d_extra_j0) *
                    three_body_weight;
                const double atmj1 =
                    (atm_i * d_atm_ji1 + atm_j * d_atm_jj1 + atm_k * d_atm_jk1 + d_extra_j1) *
                    three_body_weight;
                const double atmj2 =
                    (atm_i * d_atm_ji2 + atm_j * d_atm_jj2 + atm_k * d_atm_jk2 + d_extra_j2) *
                    three_body_weight;
                const double atmk0 =
                    (atm_i * d_atm_ki0 + atm_j * d_atm_kj0 + atm_k * d_atm_kk0 + d_extra_k0) *
                    three_body_weight;
                const double atmk1 =
                    (atm_i * d_atm_ki1 + atm_j * d_atm_kj1 + atm_k * d_atm_kk1 + d_extra_k1) *
                    three_body_weight;
                const double atmk2 =
                    (atm_i * d_atm_ki2 + atm_j * d_atm_kj2 + atm_k * d_atm_kk2 + d_extra_k2) *
                    three_body_weight;

                // Decay combination derivatives per dim
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;

                // Radial derivative directional parts (precomputed per dim)
                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                const double decay_prod = decay_ij * decay_ik;

                for (size_t l = 0; l < nbasis3; ++l) {
                    const double scale_val = radial[l] * atm * decay_prod;
                    const double scale_ang = decay_prod * radial[l];
                    const double d_rad_l = d_radial[l];
                    const size_t z0 = base + l * nabasis;

                    // values
                    for (size_t aidx = 0; aidx < nabasis; ++aidx) {
                        atom_rep[z0 + aidx] += angular[aidx] * scale_val;
                    }

                    // radial derivative directional parts for this l
                    const double dri0 = d_rad_l * (-(BmA0 + BmC0));
                    const double dri1 = d_rad_l * (-(BmA1 + BmC1));
                    const double dri2 = d_rad_l * (-(BmA2 + BmC2));
                    const double drj0 = d_rad_l * BmA0;
                    const double drj1 = d_rad_l * BmA1;
                    const double drj2 = d_rad_l * BmA2;
                    const double drk0 = d_rad_l * BmC0;
                    const double drk1 = d_rad_l * BmC1;
                    const double drk2 = d_rad_l * BmC2;

                    const double rad_l = radial[l];

                    for (size_t aidx = 0; aidx < nabasis; ++aidx) {
                        const double ang = angular[aidx];
                        const double dang = d_angular[aidx];

                        // dim 0
                        {
                            const double gi =
                                dang * dai0 * scale_ang * atm + ang * dri0 * atm * decay_prod +
                                ang * rad_l * atmi0 * decay_prod + ang * rad_l * dec_i0 * atm;
                            const double gj =
                                dang * daj0 * scale_ang * atm + ang * drj0 * atm * decay_prod +
                                ang * rad_l * atmj0 * decay_prod + ang * rad_l * dec_j0 * atm;
                            const double gk =
                                dang * dak0 * scale_ang * atm + ang * drk0 * atm * decay_prod +
                                ang * rad_l * atmk0 * decay_prod + ang * rad_l * dec_k0 * atm;
                            atom_grad[((z0 + aidx) * natoms + i) * 3 + 0] += gi;
                            atom_grad[((z0 + aidx) * natoms + j) * 3 + 0] += gj;
                            atom_grad[((z0 + aidx) * natoms + k) * 3 + 0] += gk;
                        }
                        // dim 1
                        {
                            const double gi =
                                dang * dai1 * scale_ang * atm + ang * dri1 * atm * decay_prod +
                                ang * rad_l * atmi1 * decay_prod + ang * rad_l * dec_i1 * atm;
                            const double gj =
                                dang * daj1 * scale_ang * atm + ang * drj1 * atm * decay_prod +
                                ang * rad_l * atmj1 * decay_prod + ang * rad_l * dec_j1 * atm;
                            const double gk =
                                dang * dak1 * scale_ang * atm + ang * drk1 * atm * decay_prod +
                                ang * rad_l * atmk1 * decay_prod + ang * rad_l * dec_k1 * atm;
                            atom_grad[((z0 + aidx) * natoms + i) * 3 + 1] += gi;
                            atom_grad[((z0 + aidx) * natoms + j) * 3 + 1] += gj;
                            atom_grad[((z0 + aidx) * natoms + k) * 3 + 1] += gk;
                        }
                        // dim 2
                        {
                            const double gi =
                                dang * dai2 * scale_ang * atm + ang * dri2 * atm * decay_prod +
                                ang * rad_l * atmi2 * decay_prod + ang * rad_l * dec_i2 * atm;
                            const double gj =
                                dang * daj2 * scale_ang * atm + ang * drj2 * atm * decay_prod +
                                ang * rad_l * atmj2 * decay_prod + ang * rad_l * dec_j2 * atm;
                            const double gk =
                                dang * dak2 * scale_ang * atm + ang * drk2 * atm * decay_prod +
                                ang * rad_l * atmk2 * decay_prod + ang * rad_l * dec_k2 * atm;
                            atom_grad[((z0 + aidx) * natoms + i) * 3 + 2] += gi;
                            atom_grad[((z0 + aidx) * natoms + j) * 3 + 2] += gj;
                            atom_grad[((z0 + aidx) * natoms + k) * 3 + 2] += gk;
                        }
                    }
                }
            }
        }

        // scatter into global rep/grad for center i (3-body block only)
        for (size_t off = 0; off < three_block_size; ++off) {
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        }
        for (size_t off = 0; off < three_block_size; ++off) {
            for (size_t a = 0; a < natoms; ++a) {
                for (size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0)
                        grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
            }
        }
    }
}

}  // namespace fchl19
}  // namespace kf
