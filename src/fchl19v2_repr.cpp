// Own header
#include "fchl19v2_repr.hpp"

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
namespace fchl19v2 {

// ==================== String-to-enum conversion ====================

TwoBodyType two_body_type_from_string(const std::string &s) {
    if (s == "log_normal") return TwoBodyType::LogNormal;
    if (s == "gaussian_r") return TwoBodyType::GaussianR;
    if (s == "gaussian_log_r") return TwoBodyType::GaussianLogR;
    if (s == "gaussian_r_no_pow") return TwoBodyType::GaussianRNoPow;
    if (s == "bessel") return TwoBodyType::Bessel;
    throw std::invalid_argument("Unknown two_body_type: " + s);
}

ThreeBodyType three_body_type_from_string(const std::string &s) {
    if (s == "odd_fourier_rbar") return ThreeBodyType::OddFourier_Rbar;
    if (s == "cosine_rbar") return ThreeBodyType::CosineSeries_Rbar;
    if (s == "odd_fourier_split_r") return ThreeBodyType::OddFourier_SplitR;
    if (s == "cosine_split_r") return ThreeBodyType::CosineSeries_SplitR;
    if (s == "cosine_split_r_no_atm") return ThreeBodyType::CosineSeries_SplitR_NoATM;
    if (s == "odd_fourier_element_resolved") return ThreeBodyType::OddFourier_ElementResolved;
    if (s == "cosine_element_resolved") return ThreeBodyType::CosineSeries_ElementResolved;
    throw std::invalid_argument("Unknown three_body_type: " + s);
}

// ==================== Helper functions (file-local) ====================

// Flat 2D indexing: (i, j) -> i*ncols + j
static inline std::size_t idx2(std::size_t i, std::size_t j, std::size_t ncols) {
    return i * ncols + j;
}

// Gradient indexing: (i, feat, a, d) -> flat index for shape (natoms, rep_size, natoms, 3)
static inline std::size_t gidx(
    std::size_t i, std::size_t feat, std::size_t a, std::size_t d, std::size_t rep_size,
    std::size_t natoms
) {
    return (((i * rep_size + feat) * natoms + a) * 3 + d);
}

// Symmetric pair index for elements p, q
static inline std::size_t pair_index(std::size_t nelements, int p, int q) {
    if (p > q) std::swap(p, q);
    long long llp = p, llq = q, llN = static_cast<long long>(nelements);
    long long idx = -llp * (llp + 1) / 2 + llq + llN * llp;
    return static_cast<std::size_t>(idx);
}

// Element-resolved block offset for three_body_a6/a7.
// Layout:
//   - nelements diagonal (B==C) blocks of size nbasis3*nbasis3_minus*nabasis,
//     stored first at index elem_b (0 .. nelements-1)
//   - nelements*(nelements-1) ordered off-diagonal (B!=C) blocks of size
//     nbasis3*nbasis3*nabasis, at indices nelements + (elem_j*nelements + elem_k)
//     where elem_j != elem_k (we visit k > j so elem_j, elem_k are unordered
//     by atom index but ordered by their own values; we write into the ordered
//     block directly).
// Returns the offset within the three-body section (not the full rep).
static inline std::size_t er_block_offset(
    std::size_t nelements, int elem_j, int elem_k,
    std::size_t nbasis3, std::size_t nbasis3_minus, std::size_t nabasis
) {
    if (elem_j == elem_k) {
        // Diagonal block: index = elem_j
        return static_cast<std::size_t>(elem_j) * nbasis3 * nbasis3_minus * nabasis;
    }
    // Off-diagonal ordered block.
    // First skip all diagonal blocks:
    const std::size_t diag_total = nelements * nbasis3 * nbasis3_minus * nabasis;
    // Ordered index: (elem_j * nelements + elem_k) but skip diagonal entries.
    // We store blocks in row-major order, skipping same-element pairs, so the
    // ordered flat index among off-diagonal pairs is:
    //   block_id = elem_j * (nelements-1) + (elem_k < elem_j ? elem_k : elem_k-1)
    const std::size_t ej = static_cast<std::size_t>(elem_j);
    const std::size_t ek = static_cast<std::size_t>(elem_k);
    const std::size_t col = (ek < ej) ? ek : ek - 1;
    const std::size_t block_id = ej * (nelements - 1) + col;
    return diag_total + block_id * nbasis3 * nbasis3 * nabasis;
}

// Half-cosine cutoff decay: 0.5*(cos(pi*r/rc) + 1)
static void decay_matrix(
    const std::vector<double> &rmat, double invrc, std::size_t natoms, std::vector<double> &out
) {
    out.resize(natoms * natoms);
    const double f = M_PI * invrc;
    for (std::size_t i = 0; i < natoms * natoms; ++i) {
        out[i] = 0.5 * (std::cos(f * rmat[i]) + 1.0);
    }
}

// Compute full pairwise distance matrix (natoms x natoms, row-major)
static std::vector<double> pairwise_distances(
    const std::vector<double> &coords, std::size_t natoms
) {
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

// Map nuclear charges to element indices
static std::vector<int> build_elem_map(
    const std::vector<int> &nuclear_z, const std::vector<int> &elements, std::size_t natoms
) {
    std::unordered_map<int, int> z2idx;
    z2idx.reserve(elements.size() * 2);
    for (std::size_t j = 0; j < elements.size(); ++j)
        z2idx[elements[j]] = static_cast<int>(j);

    std::vector<int> elem_of_atom(natoms, -1);
    for (std::size_t i = 0; i < natoms; ++i) {
        auto it = z2idx.find(nuclear_z[i]);
        if (it == z2idx.end())
            throw std::runtime_error("nuclear_z contains an element not present in elements");
        elem_of_atom[i] = it->second;
    }
    return elem_of_atom;
}

// ==================== Rep size computation ====================

std::size_t compute_rep_size(
    std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3, std::size_t nabasis,
    ThreeBodyType three_body_type, std::size_t nbasis3_minus
) {
    const std::size_t two_body = nelements * nbasis2;
    const std::size_t n_pairs = nelements * (nelements + 1) / 2;

    std::size_t three_body = 0;
    switch (three_body_type) {
        case ThreeBodyType::OddFourier_Rbar:
        case ThreeBodyType::CosineSeries_Rbar:
            three_body = n_pairs * nbasis3 * nabasis;
            break;
        case ThreeBodyType::OddFourier_SplitR:
        case ThreeBodyType::CosineSeries_SplitR:
        case ThreeBodyType::CosineSeries_SplitR_NoATM:
            if (nbasis3_minus == 0)
                throw std::invalid_argument("nbasis3_minus must be > 0 for SplitR variants");
            three_body = n_pairs * nbasis3 * nbasis3_minus * nabasis;
            break;
        case ThreeBodyType::OddFourier_ElementResolved:
        case ThreeBodyType::CosineSeries_ElementResolved:
            if (nbasis3_minus == 0)
                throw std::invalid_argument(
                    "nbasis3_minus must be > 0 for ElementResolved variants (used for B==C pairs)"
                );
            // nelements diagonal (B==C) blocks: nRs3 * nRs3_minus * nabasis each
            // nelements*(nelements-1) ordered off-diagonal (B!=C) blocks: nRs3 * nRs3 * nabasis each
            three_body = nelements * nbasis3 * nbasis3_minus * nabasis +
                         nelements * (nelements - 1) * nbasis3 * nbasis3 * nabasis;
            break;
    }
    return two_body + three_body;
}

// ==================== Two-body forward implementations ====================

// T1: LogNormal (baseline FCHL19)
static void two_body_log_normal_forward(
    std::size_t natoms, std::size_t nbasis2, std::size_t nelements,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    // Precompute log(Rs2) and 1/Rs2
    std::vector<double> log_Rs2(nbasis2), inv_Rs2(nbasis2);
    for (std::size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0) throw std::invalid_argument("All Rs2 must be > 0");
        log_Rs2[k] = std::log(Rs2[k]);
        inv_Rs2[k] = 1.0 / Rs2[k];
    }

#pragma omp parallel
    {
        std::vector<double> rep_local(natoms * rep_size, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (long long ii = 0; ii < static_cast<long long>(natoms); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const int elem_i = elem_of_atom[i];
            for (std::size_t j = i + 1; j < natoms; ++j) {
                const int elem_j = elem_of_atom[j];
                const double rij = D[idx2(i, j, natoms)];
                if (rij > rcut) continue;

                const double rij2 = rij * rij;
                const double t = eta2 / std::max(rij2, kf::EPS);
                const double log1pt = std::log1p(t);
                const double sigma = std::sqrt(std::max(log1pt, 0.0));
                if (sigma < kf::EPS) continue;
                const double mu = std::log(rij) - 0.5 * log1pt;
                const double decay_ij = rdecay2[idx2(i, j, natoms)];
                const double inv_pref =
                    decay_ij / (sigma * kf::SQRT_2PI * std::pow(rij, two_body_decay));
                const double inv_sigma_sq = 1.0 / (sigma * sigma);

                const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
                const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dlog = log_Rs2[k] - mu;
                    const double g = std::exp(-0.5 * dlog * dlog * inv_sigma_sq);
                    const double val = inv_pref * g * inv_Rs2[k];
                    rep_local[idx2(i, ch_j + k, rep_size)] += val;
                    rep_local[idx2(j, ch_i + k, rep_size)] += val;
                }
            }
        }

#pragma omp critical
        {
            for (std::size_t idx = 0; idx < natoms * rep_size; ++idx) {
                rep[idx] += rep_local[idx];
            }
        }
    }
}

// T2: Fixed-width Gaussian in r
static void two_body_gaussian_r_forward(
    std::size_t natoms, std::size_t nbasis2, std::size_t nelements,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
#pragma omp parallel
    {
        std::vector<double> rep_local(natoms * rep_size, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (long long ii = 0; ii < static_cast<long long>(natoms); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const int elem_i = elem_of_atom[i];
            for (std::size_t j = i + 1; j < natoms; ++j) {
                const int elem_j = elem_of_atom[j];
                const double rij = D[idx2(i, j, natoms)];
                if (rij > rcut) continue;

                const double decay_ij = rdecay2[idx2(i, j, natoms)];
                const double scaling = decay_ij / std::pow(rij, two_body_decay);

                const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
                const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dr = rij - Rs2[k];
                    const double val = scaling * std::exp(-eta2 * dr * dr);
                    rep_local[idx2(i, ch_j + k, rep_size)] += val;
                    rep_local[idx2(j, ch_i + k, rep_size)] += val;
                }
            }
        }

#pragma omp critical
        {
            for (std::size_t idx = 0; idx < natoms * rep_size; ++idx) {
                rep[idx] += rep_local[idx];
            }
        }
    }
}

// T3: Fixed-width Gaussian in ln(r)
static void two_body_gaussian_log_r_forward(
    std::size_t natoms, std::size_t nbasis2, std::size_t nelements,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    std::vector<double> log_Rs2(nbasis2);
    for (std::size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0) throw std::invalid_argument("All Rs2 must be > 0");
        log_Rs2[k] = std::log(Rs2[k]);
    }

#pragma omp parallel
    {
        std::vector<double> rep_local(natoms * rep_size, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (long long ii = 0; ii < static_cast<long long>(natoms); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const int elem_i = elem_of_atom[i];
            for (std::size_t j = i + 1; j < natoms; ++j) {
                const int elem_j = elem_of_atom[j];
                const double rij = D[idx2(i, j, natoms)];
                if (rij > rcut) continue;

                const double decay_ij = rdecay2[idx2(i, j, natoms)];
                const double scaling = decay_ij / std::pow(rij, two_body_decay);
                const double log_rij = std::log(rij);

                const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
                const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dlog = log_rij - log_Rs2[k];
                    const double val = scaling * std::exp(-eta2 * dlog * dlog);
                    rep_local[idx2(i, ch_j + k, rep_size)] += val;
                    rep_local[idx2(j, ch_i + k, rep_size)] += val;
                }
            }
        }

#pragma omp critical
        {
            for (std::size_t idx = 0; idx < natoms * rep_size; ++idx) {
                rep[idx] += rep_local[idx];
            }
        }
    }
}

// T4: Fixed-width Gaussian in r, no power-law decay
static void two_body_gaussian_r_no_pow_forward(
    std::size_t natoms, std::size_t nbasis2, std::size_t /* nelements */,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double /* two_body_decay */, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
#pragma omp parallel
    {
        std::vector<double> rep_local(natoms * rep_size, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (long long ii = 0; ii < static_cast<long long>(natoms); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const int elem_i = elem_of_atom[i];
            for (std::size_t j = i + 1; j < natoms; ++j) {
                const int elem_j = elem_of_atom[j];
                const double rij = D[idx2(i, j, natoms)];
                if (rij > rcut) continue;

                const double decay_ij = rdecay2[idx2(i, j, natoms)];

                const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
                const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dr = rij - Rs2[k];
                    const double val = decay_ij * std::exp(-eta2 * dr * dr);
                    rep_local[idx2(i, ch_j + k, rep_size)] += val;
                    rep_local[idx2(j, ch_i + k, rep_size)] += val;
                }
            }
        }

#pragma omp critical
        {
            for (std::size_t idx = 0; idx < natoms * rep_size; ++idx) {
                rep[idx] += rep_local[idx];
            }
        }
    }
}

// T5: Radial Bessel basis
static void two_body_bessel_forward(
    std::size_t natoms, std::size_t nbasis2, std::size_t /* nelements */,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &/* Rs2 */, double /* eta2 */, double /* two_body_decay */,
    double rcut, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep
) {
    const double norm = std::sqrt(2.0 / rcut);
    const double pi_over_rcut = M_PI / rcut;

#pragma omp parallel
    {
        std::vector<double> rep_local(natoms * rep_size, 0.0);

#pragma omp for schedule(dynamic) nowait
        for (long long ii = 0; ii < static_cast<long long>(natoms); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const int elem_i = elem_of_atom[i];
            for (std::size_t j = i + 1; j < natoms; ++j) {
                const int elem_j = elem_of_atom[j];
                const double rij = D[idx2(i, j, natoms)];
                if (rij > rcut || rij < kf::EPS) continue;

                const double decay_ij = rdecay2[idx2(i, j, natoms)];
                const double inv_rij = 1.0 / rij;

                const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
                const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double n_k = static_cast<double>(k + 1);
                    const double val =
                        decay_ij * norm * std::sin(n_k * pi_over_rcut * rij) * inv_rij;
                    rep_local[idx2(i, ch_j + k, rep_size)] += val;
                    rep_local[idx2(j, ch_i + k, rep_size)] += val;
                }
            }
        }

#pragma omp critical
        {
            for (std::size_t idx = 0; idx < natoms * rep_size; ++idx) {
                rep[idx] += rep_local[idx];
            }
        }
    }
}

// Dispatch two-body forward
static void two_body_forward(
    TwoBodyType type, std::size_t natoms, std::size_t nbasis2, std::size_t nelements,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    switch (type) {
        case TwoBodyType::LogNormal:
            two_body_log_normal_forward(
                natoms, nbasis2, nelements, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep
            );
            break;
        case TwoBodyType::GaussianR:
            two_body_gaussian_r_forward(
                natoms, nbasis2, nelements, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep
            );
            break;
        case TwoBodyType::GaussianLogR:
            two_body_gaussian_log_r_forward(
                natoms, nbasis2, nelements, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep
            );
            break;
        case TwoBodyType::GaussianRNoPow:
            two_body_gaussian_r_no_pow_forward(
                natoms, nbasis2, nelements, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep
            );
            break;
        case TwoBodyType::Bessel:
            two_body_bessel_forward(
                natoms, nbasis2, nelements, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep
            );
            break;
    }
}

// ==================== Two-body gradient implementations ====================

// T1: LogNormal gradient (ported from existing fchl19_repr.cpp)
static void two_body_log_normal_grad(
    std::size_t natoms, std::size_t nbasis2, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    std::vector<double> log_Rs2(nbasis2), inv_Rs2(nbasis2);
    for (std::size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0) throw std::invalid_argument("All Rs2 must be > 0");
        log_Rs2[k] = std::log(Rs2[k]);
        inv_Rs2[k] = 1.0 / Rs2[k];
    }

    for (std::size_t i = 0; i < natoms; ++i) {
        const int elem_i = elem_of_atom[i];
        for (std::size_t j = i + 1; j < natoms; ++j) {
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i, j, natoms)];
            if (rij > rcut) continue;
            const double invr = 1.0 / rij;
            const double invr2 = invr * invr;
            const double s2 = std::log1p(eta2 * invr2);  // sigma^2
            const double sigma = std::sqrt(std::max(s2, 0.0));
            if (sigma < kf::EPS) continue;
            const double mu = std::log(rij) - 0.5 * s2;
            const double decay_ij = rdecay2[idx2(i, j, natoms)];
            const double scaling = std::pow(rij, -two_body_decay);
            const double inv_pref_common = 1.0 / (sigma * std::sqrt(2.0 * M_PI));

            std::vector<double> radial_base(nbasis2), radial(nbasis2), exp_ln(nbasis2);
            for (std::size_t k = 0; k < nbasis2; ++k) {
                const double dlog = log_Rs2[k] - mu;
                const double g = std::exp(-0.5 * dlog * dlog / s2);
                exp_ln[k] = g * std::sqrt(2.0);
                radial_base[k] = (inv_pref_common / Rs2[k]) * g;
                radial[k] = radial_base[k] * scaling * decay_ij;
            }

            const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
            const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;
            for (std::size_t k = 0; k < nbasis2; ++k) {
                rep[idx2(i, ch_j + k, rep_size)] += radial[k];
                rep[idx2(j, ch_i + k, rep_size)] += radial[k];
            }

            const double exp_s2 = std::exp(s2);
            const double sqrt_exp_s2 = std::sqrt(exp_s2);
            for (int t = 0; t < 3; ++t) {
                const double dx = -(coords[3 * i + t] - coords[3 * j + t]);
                const double dscal = two_body_decay * dx * std::pow(rij, -(two_body_decay + 2.0));
                const double ddecay = dx * 0.5 * M_PI *
                                      std::sin(M_PI * rij * (rcut > 0 ? 1.0 / rcut : 0.0)) *
                                      (rcut > 0 ? 1.0 / rcut : 0.0) * invr;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double L = log_Rs2[k] - mu;
                    const double term1 =
                        L *
                        (-dx * (rij * rij * exp_s2 + eta2) / std::pow(rij * sqrt_exp_s2, 3)) *
                        (sqrt_exp_s2 / (s2 * rij));
                    const double term2 =
                        (L * L) * eta2 * dx / ((s2 * s2) * std::pow(rij, 4) * exp_s2);
                    const double A =
                        (term1 + term2) * (exp_ln[k] / (Rs2[k] * sigma * std::sqrt(M_PI) * 2.0)) -
                        (exp_ln[k] * eta2 * dx) / (Rs2[k] * (s2 * std::sqrt(M_PI)) * sigma *
                                                    std::pow(rij, 4) * exp_s2 * 2.0);
                    double part = A * scaling * decay_ij + radial_base[k] * dscal * decay_ij +
                                  radial_base[k] * scaling * ddecay;

                    std::size_t feat_i = ch_j + k;
                    std::size_t feat_j = ch_i + k;
                    grad[gidx(i, feat_i, i, (std::size_t)t, rep_size, natoms)] += part;
                    grad[gidx(i, feat_i, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, i, (std::size_t)t, rep_size, natoms)] += part;
                }
            }
        }
    }
}

// T2: GaussianR gradient
static void two_body_gaussian_r_grad(
    std::size_t natoms, std::size_t nbasis2, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const double inv_rcut = (rcut > 0) ? 1.0 / rcut : 0.0;

    for (std::size_t i = 0; i < natoms; ++i) {
        const int elem_i = elem_of_atom[i];
        for (std::size_t j = i + 1; j < natoms; ++j) {
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i, j, natoms)];
            if (rij > rcut) continue;

            const double invr = 1.0 / rij;
            const double decay_ij = rdecay2[idx2(i, j, natoms)];
            const double rpow = std::pow(rij, -two_body_decay);

            const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
            const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

            // Compute values
            std::vector<double> vals(nbasis2);
            for (std::size_t k = 0; k < nbasis2; ++k) {
                const double dr = rij - Rs2[k];
                const double gauss = std::exp(-eta2 * dr * dr);
                vals[k] = decay_ij * rpow * gauss;
                rep[idx2(i, ch_j + k, rep_size)] += vals[k];
                rep[idx2(j, ch_i + k, rep_size)] += vals[k];
            }

            // Gradient: d/d(x_i_t)
            // dr/dx_i = (x_i - x_j)/r, so use dx = (x_i - x_j)
            for (int t = 0; t < 3; ++t) {
                const double dx = coords[3 * i + t] - coords[3 * j + t];  // x_i - x_j
                const double dx_over_r = dx * invr;  // (x_i-x_j)/r = dr/dx_i

                const double ddecay_dr =
                    -0.5 * M_PI * inv_rcut * std::sin(M_PI * rij * inv_rcut);
                const double ddecay = ddecay_dr * dx_over_r;

                // d(rpow)/dx_i = d(r^-p)/dr * dr/dx_i = -p * r^(-p-2) * dx
                const double drpow = -two_body_decay * std::pow(rij, -(two_body_decay + 2.0)) * dx;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dr = rij - Rs2[k];
                    const double gauss = std::exp(-eta2 * dr * dr);

                    // d(gauss)/dx_i = d(gauss)/dr * dr/dx_i
                    //               = gauss * (-2*eta2*dr) * dx/r
                    const double dgauss = gauss * (-2.0 * eta2 * dr) * dx_over_r;

                    // Product rule: d(decay * rpow * gauss)/dx_i
                    const double part =
                        ddecay * rpow * gauss + decay_ij * drpow * gauss + decay_ij * rpow * dgauss;

                    std::size_t feat_i = ch_j + k;
                    std::size_t feat_j = ch_i + k;
                    grad[gidx(i, feat_i, i, (std::size_t)t, rep_size, natoms)] += part;
                    grad[gidx(i, feat_i, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, i, (std::size_t)t, rep_size, natoms)] += part;
                }
            }
        }
    }
}

// T3: GaussianLogR gradient
static void two_body_gaussian_log_r_grad(
    std::size_t natoms, std::size_t nbasis2, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double two_body_decay, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const double inv_rcut = (rcut > 0) ? 1.0 / rcut : 0.0;

    std::vector<double> log_Rs2(nbasis2);
    for (std::size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0) throw std::invalid_argument("All Rs2 must be > 0");
        log_Rs2[k] = std::log(Rs2[k]);
    }

    for (std::size_t i = 0; i < natoms; ++i) {
        const int elem_i = elem_of_atom[i];
        for (std::size_t j = i + 1; j < natoms; ++j) {
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i, j, natoms)];
            if (rij > rcut) continue;

            const double invr = 1.0 / rij;
            const double decay_ij = rdecay2[idx2(i, j, natoms)];
            const double rpow = std::pow(rij, -two_body_decay);
            const double log_rij = std::log(rij);

            const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
            const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

            // Compute values
            for (std::size_t k = 0; k < nbasis2; ++k) {
                const double dlog = log_rij - log_Rs2[k];
                const double gauss = std::exp(-eta2 * dlog * dlog);
                const double val = decay_ij * rpow * gauss;
                rep[idx2(i, ch_j + k, rep_size)] += val;
                rep[idx2(j, ch_i + k, rep_size)] += val;
            }

            // Gradient: dr/dx_i = (x_i - x_j)/r
            for (int t = 0; t < 3; ++t) {
                const double dx = coords[3 * i + t] - coords[3 * j + t];  // x_i - x_j
                const double dx_over_r = dx * invr;

                const double ddecay_dr =
                    -0.5 * M_PI * inv_rcut * std::sin(M_PI * rij * inv_rcut);
                const double ddecay = ddecay_dr * dx_over_r;
                const double drpow = -two_body_decay * std::pow(rij, -(two_body_decay + 2.0)) * dx;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dlog = log_rij - log_Rs2[k];
                    const double gauss = std::exp(-eta2 * dlog * dlog);

                    // d(gauss)/dx_i = gauss * (-2*eta2*dlog) * d(log_r)/dx_i
                    // d(log_r)/dx_i = (x_i-x_j)/r^2
                    const double dgauss = gauss * (-2.0 * eta2 * dlog) * dx * invr * invr;

                    const double part =
                        ddecay * rpow * gauss + decay_ij * drpow * gauss + decay_ij * rpow * dgauss;

                    std::size_t feat_i = ch_j + k;
                    std::size_t feat_j = ch_i + k;
                    grad[gidx(i, feat_i, i, (std::size_t)t, rep_size, natoms)] += part;
                    grad[gidx(i, feat_i, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, i, (std::size_t)t, rep_size, natoms)] += part;
                }
            }
        }
    }
}

// T4: GaussianRNoPow gradient
static void two_body_gaussian_r_no_pow_grad(
    std::size_t natoms, std::size_t nbasis2, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &Rs2, double eta2, double /* two_body_decay */, double rcut,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const double inv_rcut = (rcut > 0) ? 1.0 / rcut : 0.0;

    for (std::size_t i = 0; i < natoms; ++i) {
        const int elem_i = elem_of_atom[i];
        for (std::size_t j = i + 1; j < natoms; ++j) {
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i, j, natoms)];
            if (rij > rcut) continue;

            const double invr = 1.0 / rij;
            const double decay_ij = rdecay2[idx2(i, j, natoms)];

            const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
            const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

            for (std::size_t k = 0; k < nbasis2; ++k) {
                const double dr = rij - Rs2[k];
                const double gauss = std::exp(-eta2 * dr * dr);
                const double val = decay_ij * gauss;
                rep[idx2(i, ch_j + k, rep_size)] += val;
                rep[idx2(j, ch_i + k, rep_size)] += val;
            }

            for (int t = 0; t < 3; ++t) {
                const double dx = coords[3 * i + t] - coords[3 * j + t];  // x_i - x_j
                const double dx_over_r = dx * invr;

                const double ddecay_dr =
                    -0.5 * M_PI * inv_rcut * std::sin(M_PI * rij * inv_rcut);
                const double ddecay = ddecay_dr * dx_over_r;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double dr = rij - Rs2[k];
                    const double gauss = std::exp(-eta2 * dr * dr);
                    const double dgauss = gauss * (-2.0 * eta2 * dr) * dx_over_r;

                    // No rpow term: d(decay * gauss)/dx_i
                    const double part = ddecay * gauss + decay_ij * dgauss;

                    std::size_t feat_i = ch_j + k;
                    std::size_t feat_j = ch_i + k;
                    grad[gidx(i, feat_i, i, (std::size_t)t, rep_size, natoms)] += part;
                    grad[gidx(i, feat_i, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, i, (std::size_t)t, rep_size, natoms)] += part;
                }
            }
        }
    }
}

// T5: Bessel gradient
static void two_body_bessel_grad(
    std::size_t natoms, std::size_t nbasis2, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay2,
    const std::vector<double> &/* Rs2 */, double /* eta2 */, double /* two_body_decay */,
    double rcut, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep, std::vector<double> &grad
) {
    const double norm = std::sqrt(2.0 / rcut);
    const double pi_over_rcut = M_PI / rcut;
    const double inv_rcut = (rcut > 0) ? 1.0 / rcut : 0.0;

    for (std::size_t i = 0; i < natoms; ++i) {
        const int elem_i = elem_of_atom[i];
        for (std::size_t j = i + 1; j < natoms; ++j) {
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i, j, natoms)];
            if (rij > rcut || rij < kf::EPS) continue;

            const double invr = 1.0 / rij;
            const double invr2 = invr * invr;
            const double decay_ij = rdecay2[idx2(i, j, natoms)];

            const std::size_t ch_j = static_cast<std::size_t>(elem_j) * nbasis2;
            const std::size_t ch_i = static_cast<std::size_t>(elem_i) * nbasis2;

            // Values
            for (std::size_t k = 0; k < nbasis2; ++k) {
                const double n_k = static_cast<double>(k + 1);
                const double sval = std::sin(n_k * pi_over_rcut * rij);
                const double val = decay_ij * norm * sval * invr;
                rep[idx2(i, ch_j + k, rep_size)] += val;
                rep[idx2(j, ch_i + k, rep_size)] += val;
            }

            // Gradient: dr/dx_i = (x_i - x_j)/r
            for (int t = 0; t < 3; ++t) {
                const double dx = coords[3 * i + t] - coords[3 * j + t];  // x_i - x_j
                const double dx_over_r = dx * invr;

                const double ddecay_dr =
                    -0.5 * M_PI * inv_rcut * std::sin(M_PI * rij * inv_rcut);
                const double ddecay = ddecay_dr * dx_over_r;

                for (std::size_t k = 0; k < nbasis2; ++k) {
                    const double n_k = static_cast<double>(k + 1);
                    const double arg = n_k * pi_over_rcut * rij;
                    const double sval = std::sin(arg);
                    const double cval = std::cos(arg);
                    const double bessel_val = norm * sval * invr;

                    // d(sin(n*pi*r/rc)/r) / dr = (n*pi/rc * cos(arg) * r - sin(arg)) / r^2
                    //                           = n*pi/rc * cos(arg)/r - sin(arg)/r^2
                    const double dbessel_dr =
                        norm * (n_k * pi_over_rcut * cval * invr - sval * invr2);
                    const double dbessel = dbessel_dr * dx_over_r;

                    const double part = ddecay * bessel_val + decay_ij * dbessel;

                    std::size_t feat_i = ch_j + k;
                    std::size_t feat_j = ch_i + k;
                    grad[gidx(i, feat_i, i, (std::size_t)t, rep_size, natoms)] += part;
                    grad[gidx(i, feat_i, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, j, (std::size_t)t, rep_size, natoms)] -= part;
                    grad[gidx(j, feat_j, i, (std::size_t)t, rep_size, natoms)] += part;
                }
            }
        }
    }
}

// Dispatch two-body gradient
static void two_body_grad(
    TwoBodyType type, std::size_t natoms, std::size_t nbasis2,
    const std::vector<double> &coords, const std::vector<double> &D,
    const std::vector<double> &rdecay2, const std::vector<double> &Rs2, double eta2,
    double two_body_decay, double rcut, const std::vector<int> &elem_of_atom,
    std::size_t rep_size, std::vector<double> &rep, std::vector<double> &grad
) {
    switch (type) {
        case TwoBodyType::LogNormal:
            two_body_log_normal_grad(
                natoms, nbasis2, coords, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep, grad
            );
            break;
        case TwoBodyType::GaussianR:
            two_body_gaussian_r_grad(
                natoms, nbasis2, coords, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep, grad
            );
            break;
        case TwoBodyType::GaussianLogR:
            two_body_gaussian_log_r_grad(
                natoms, nbasis2, coords, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep, grad
            );
            break;
        case TwoBodyType::GaussianRNoPow:
            two_body_gaussian_r_no_pow_grad(
                natoms, nbasis2, coords, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep, grad
            );
            break;
        case TwoBodyType::Bessel:
            two_body_bessel_grad(
                natoms, nbasis2, coords, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
                elem_of_atom, rep_size, rep, grad
            );
            break;
    }
}

// ==================== Three-body forward (A1 baseline only for now) ====================

// A1: OddFourier + Rbar + ATM (ported from existing fchl19_repr.cpp)
static void three_body_a1_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nabasis, const std::vector<double> &coords, const std::vector<double> &D,
    const std::vector<double> &rdecay3, const std::vector<double> &Rs3, double eta3, double zeta,
    double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t n_harm = nabasis / 2;

    // Precompute angular weights
    std::vector<double> ang_w(n_harm);
    std::vector<int> ang_o(n_harm);
    for (std::size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2 * l + 1);
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // ATM factor
                const double denom =
                    std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // Angular basis via Chebyshev recurrence
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w[0] * cos_i;
                if (nabasis > 1) angular[1] = ang_w[0] * sin_i;

                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0;
                    double cn_1 = cos_i, sn_1 = sin_i;
                    std::size_t harm_stored = 1;
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
                            if (++harm_stored >= n_harm) break;
                        }
                    }
                }

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = three_offset + pair_idx * (nbasis3 * nabasis);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double rbar = 0.5 * (rij + rik);

                for (std::size_t l = 0; l < nbasis3; ++l) {
                    const double dr = rbar - Rs3[l];
                    const double radial_l = std::exp(-eta3 * dr * dr) * decay_ij * decay_ik;
                    const double scale = radial_l * ksi3;
                    const std::size_t z = base + l * nabasis;
                    double *dst = &rep[idx2(i, z, rep_size)];
                    for (std::size_t t = 0; t < nabasis; ++t)
                        dst[t] += angular[t] * scale;
                }
            }
        }
    }
}

// A2: CosineSeries + Rbar + ATM
// Angular basis: A[m] = cos(m * theta_i), m = 0, 1, ..., nabasis-1
// dA[m]/d(cos_theta) = -m * sin(m * theta_i)  (chain rule via Chebyshev)
static void three_body_a2_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nabasis, const std::vector<double> &coords, const std::vector<double> &D,
    const std::vector<double> &rdecay3, const std::vector<double> &Rs3, double eta3, double acut,
    double three_body_decay, double three_body_weight, const std::vector<int> &elem_of_atom,
    std::size_t rep_size, std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // ATM factor
                const double denom =
                    std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // Cosine series angular basis via Chebyshev recurrence
                // T_0(x)=1, T_1(x)=x, T_n(x)=2x*T_{n-1} - T_{n-2}
                if (nabasis > 0) angular[0] = 1.0;
                if (nabasis > 1) angular[1] = cos_i;
                for (std::size_t m = 2; m < nabasis; ++m) {
                    angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];
                }

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = three_offset + pair_idx * (nbasis3 * nabasis);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double rbar = 0.5 * (rij + rik);

                for (std::size_t l = 0; l < nbasis3; ++l) {
                    const double dr = rbar - Rs3[l];
                    const double radial_l = std::exp(-eta3 * dr * dr) * decay_ij * decay_ik;
                    const double scale = radial_l * ksi3;
                    const std::size_t z = base + l * nabasis;
                    double *dst = &rep[idx2(i, z, rep_size)];
                    for (std::size_t t = 0; t < nabasis; ++t)
                        dst[t] += angular[t] * scale;
                }
            }
        }
    }
}

// A3: OddFourier + SplitR + ATM
// Same angular as A1 but radial uses r_plus = rij+rik and r_minus = |rij-rik|
static void three_body_a3_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double zeta, double acut, double three_body_decay,
    double three_body_weight, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t n_harm = nabasis / 2;

    // Precompute angular weights (same as A1)
    std::vector<double> ang_w(n_harm);
    std::vector<int> ang_o(n_harm);
    for (std::size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2 * l + 1);
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // ATM factor
                const double denom =
                    std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // OddFourier angular basis (same as A1)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w[0] * cos_i;
                if (nabasis > 1) angular[1] = ang_w[0] * sin_i;

                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0;
                    double cn_1 = cos_i, sn_1 = sin_i;
                    std::size_t harm_stored = 1;
                    const int max_o = ang_o[n_harm - 1];
                    for (int n = 2; n <= max_o; ++n) {
                        const double cn = two_cos * cn_1 - cn_2;
                        const double sn = two_cos * sn_1 - sn_2;
                        cn_2 = cn_1; sn_2 = sn_1;
                        cn_1 = cn; sn_1 = sn;
                        if (n == ang_o[harm_stored]) {
                            angular[2 * harm_stored] = ang_w[harm_stored] * cn;
                            angular[2 * harm_stored + 1] = ang_w[harm_stored] * sn;
                            if (++harm_stored >= n_harm) break;
                        }
                    }
                }

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base =
                    three_offset + pair_idx * (nbasis3 * nbasis3_minus * nabasis);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double r_plus = rij + rik;
                const double r_minus = std::abs(rij - rik);

                for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                    const double dp = r_plus - Rs3[l1];
                    const double phi_p = std::exp(-eta3 * dp * dp);
                    for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                        const double dm = r_minus - Rs3_minus[l2];
                        const double phi_m = std::exp(-eta3_minus * dm * dm);
                        const double radial = phi_p * phi_m * decay_ij * decay_ik;
                        const double scale = radial * ksi3;
                        const std::size_t z =
                            base + (l1 * nbasis3_minus + l2) * nabasis;
                        double *dst = &rep[idx2(i, z, rep_size)];
                        for (std::size_t t = 0; t < nabasis; ++t)
                            dst[t] += angular[t] * scale;
                    }
                }
            }
        }
    }
}

// A4: CosineSeries + SplitR + ATM
static void three_body_a4_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // ATM factor
                const double denom =
                    std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // Cosine series angular basis (Chebyshev)
                if (nabasis > 0) angular[0] = 1.0;
                if (nabasis > 1) angular[1] = cos_i;
                for (std::size_t m = 2; m < nabasis; ++m)
                    angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base =
                    three_offset + pair_idx * (nbasis3 * nbasis3_minus * nabasis);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double r_plus = rij + rik;
                const double r_minus = std::abs(rij - rik);

                for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                    const double dp = r_plus - Rs3[l1];
                    const double phi_p = std::exp(-eta3 * dp * dp);
                    for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                        const double dm = r_minus - Rs3_minus[l2];
                        const double phi_m = std::exp(-eta3_minus * dm * dm);
                        const double radial = phi_p * phi_m * decay_ij * decay_ik;
                        const double scale = radial * ksi3;
                        const std::size_t z =
                            base + (l1 * nbasis3_minus + l2) * nabasis;
                        double *dst = &rep[idx2(i, z, rep_size)];
                        for (std::size_t t = 0; t < nabasis; ++t)
                            dst[t] += angular[t] * scale;
                    }
                }
            }
        }
    }
}

// A5: CosineSeries + SplitR, ATM factor = 1
static void three_body_a5_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double acut, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));

                // Cosine series angular basis (Chebyshev)
                if (nabasis > 0) angular[0] = 1.0;
                if (nabasis > 1) angular[1] = cos_i;
                for (std::size_t m = 2; m < nabasis; ++m)
                    angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base =
                    three_offset + pair_idx * (nbasis3 * nbasis3_minus * nabasis);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double r_plus = rij + rik;
                const double r_minus = std::abs(rij - rik);

                for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                    const double dp = r_plus - Rs3[l1];
                    const double phi_p = std::exp(-eta3 * dp * dp);
                    for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                        const double dm = r_minus - Rs3_minus[l2];
                        const double phi_m = std::exp(-eta3_minus * dm * dm);
                        const double radial = phi_p * phi_m * decay_ij * decay_ik;
                        const double scale = radial * three_body_weight;
                        const std::size_t z =
                            base + (l1 * nbasis3_minus + l2) * nabasis;
                        double *dst = &rep[idx2(i, z, rep_size)];
                        for (std::size_t t = 0; t < nabasis; ++t)
                            dst[t] += angular[t] * scale;
                    }
                }
            }
        }
    }
}

// A6: OddFourier + ElementResolved radial + ATM
// B != C: radial = exp(-eta3*(r_ij-Rs3[l1])^2) * exp(-eta3*(r_ik-Rs3[l2])^2) * decay
//         stored in ordered block (elem_j, elem_k), size nRs3*nRs3*nabasis
// B == C: radial = exp(-eta3*(r_plus-Rs3[l1])^2) * exp(-eta3_minus*(r_minus-Rs3m[l2])^2) * decay
//         stored in diagonal block (elem_j==elem_k), size nRs3*nRs3_minus*nabasis
static void three_body_a6_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double zeta, double acut, double three_body_decay,
    double three_body_weight, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t n_harm = nabasis / 2;

    std::vector<double> ang_w(n_harm);
    std::vector<int> ang_o(n_harm);
    for (std::size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2 * l + 1);
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // ATM factor
                const double denom =
                    std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // OddFourier angular basis (same as A1)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w[0] * cos_i;
                if (nabasis > 1) angular[1] = ang_w[0] * sin_i;
                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0, cn_1 = cos_i, sn_1 = sin_i;
                    std::size_t harm_stored = 1;
                    const int max_o = ang_o[n_harm - 1];
                    for (int n = 2; n <= max_o; ++n) {
                        const double cn = two_cos * cn_1 - cn_2;
                        const double sn = two_cos * sn_1 - sn_2;
                        cn_2 = cn_1; sn_2 = sn_1; cn_1 = cn; sn_1 = sn;
                        if (n == ang_o[harm_stored]) {
                            angular[2 * harm_stored] = ang_w[harm_stored] * cn;
                            angular[2 * harm_stored + 1] = ang_w[harm_stored] * sn;
                            if (++harm_stored >= n_harm) break;
                        }
                    }
                }

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double decay_prod = decay_ij * decay_ik;

                const std::size_t block_off =
                    er_block_offset(nelements, elem_j, elem_k, nbasis3, nbasis3_minus, nabasis);
                const std::size_t base = three_offset + block_off;

                if (elem_j == elem_k) {
                    // Diagonal (B==C): SplitR basis in (r_plus, r_minus)
                    const double r_plus = rij + rik;
                    const double r_minus = std::abs(rij - rik);
                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dp = r_plus - Rs3[l1];
                        const double phi_p = std::exp(-eta3 * dp * dp);
                        for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                            const double dm = r_minus - Rs3_minus[l2];
                            const double phi_m = std::exp(-eta3_minus * dm * dm);
                            const double scale = phi_p * phi_m * decay_prod * ksi3;
                            const std::size_t z = base + (l1 * nbasis3_minus + l2) * nabasis;
                            double *dst = &rep[idx2(i, z, rep_size)];
                            for (std::size_t t = 0; t < nabasis; ++t)
                                dst[t] += angular[t] * scale;
                        }
                    }
                } else {
                    // Off-diagonal (B!=C): ordered (r_ij, r_ik) product basis
                    // Block (elem_j, elem_k): r_ij on outer index, r_ik on inner index
                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dj = rij - Rs3[l1];
                        const double phi_j = std::exp(-eta3 * dj * dj);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dk = rik - Rs3[l2];
                            const double phi_k = std::exp(-eta3 * dk * dk);
                            const double scale = phi_j * phi_k * decay_prod * ksi3;
                            const std::size_t z = base + (l1 * nbasis3 + l2) * nabasis;
                            double *dst = &rep[idx2(i, z, rep_size)];
                            for (std::size_t t = 0; t < nabasis; ++t)
                                dst[t] += angular[t] * scale;
                        }
                    }
                    // Also accumulate into the (elem_k, elem_j) block with r_ij/r_ik swapped
                    const std::size_t block_off_kj =
                        er_block_offset(nelements, elem_k, elem_j, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base_kj = three_offset + block_off_kj;
                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dk = rik - Rs3[l1];
                        const double phi_k = std::exp(-eta3 * dk * dk);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dj = rij - Rs3[l2];
                            const double phi_j = std::exp(-eta3 * dj * dj);
                            const double scale = phi_k * phi_j * decay_prod * ksi3;
                            const std::size_t z = base_kj + (l1 * nbasis3 + l2) * nabasis;
                            double *dst = &rep[idx2(i, z, rep_size)];
                            for (std::size_t t = 0; t < nabasis; ++t)
                                dst[t] += angular[t] * scale;
                        }
                    }
                }
            }
        }
    }
}

// A7: CosineSeries + ElementResolved radial + ATM
// Same as A6 but with cosine (Chebyshev) angular basis instead of OddFourier
static void three_body_a7_forward(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep
) {
    const std::size_t three_offset = nelements * nbasis2;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> angular(nabasis);

        const double *rb = &coords[3 * i];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];

            const double inv_rij = 1.0 / std::max(rij, kf::EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];

                const double inv_rik = 1.0 / std::max(rik, kf::EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                const double rjk = D[idx2(j, k, natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, kf::EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                // ATM factor
                const double denom =
                    std::pow(std::max(rik * rij * rjk, kf::EPS), three_body_decay);
                const double ksi3 =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // Cosine series (Chebyshev) angular basis
                if (nabasis > 0) angular[0] = 1.0;
                if (nabasis > 1) angular[1] = cos_i;
                for (std::size_t m = 2; m < nabasis; ++m)
                    angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double decay_prod = decay_ij * decay_ik;

                const std::size_t block_off =
                    er_block_offset(nelements, elem_j, elem_k, nbasis3, nbasis3_minus, nabasis);
                const std::size_t base = three_offset + block_off;

                if (elem_j == elem_k) {
                    // Diagonal (B==C): SplitR basis
                    const double r_plus = rij + rik;
                    const double r_minus = std::abs(rij - rik);
                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dp = r_plus - Rs3[l1];
                        const double phi_p = std::exp(-eta3 * dp * dp);
                        for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                            const double dm = r_minus - Rs3_minus[l2];
                            const double phi_m = std::exp(-eta3_minus * dm * dm);
                            const double scale = phi_p * phi_m * decay_prod * ksi3;
                            const std::size_t z = base + (l1 * nbasis3_minus + l2) * nabasis;
                            double *dst = &rep[idx2(i, z, rep_size)];
                            for (std::size_t t = 0; t < nabasis; ++t)
                                dst[t] += angular[t] * scale;
                        }
                    }
                } else {
                    // Off-diagonal (B!=C): ordered (r_ij, r_ik) product basis
                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dj = rij - Rs3[l1];
                        const double phi_j = std::exp(-eta3 * dj * dj);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dk = rik - Rs3[l2];
                            const double phi_k = std::exp(-eta3 * dk * dk);
                            const double scale = phi_j * phi_k * decay_prod * ksi3;
                            const std::size_t z = base + (l1 * nbasis3 + l2) * nabasis;
                            double *dst = &rep[idx2(i, z, rep_size)];
                            for (std::size_t t = 0; t < nabasis; ++t)
                                dst[t] += angular[t] * scale;
                        }
                    }
                    // Also accumulate into the (elem_k, elem_j) block with swapped radial indices
                    const std::size_t block_off_kj =
                        er_block_offset(nelements, elem_k, elem_j, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base_kj = three_offset + block_off_kj;
                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dk = rik - Rs3[l1];
                        const double phi_k = std::exp(-eta3 * dk * dk);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dj = rij - Rs3[l2];
                            const double phi_j = std::exp(-eta3 * dj * dj);
                            const double scale = phi_k * phi_j * decay_prod * ksi3;
                            const std::size_t z = base_kj + (l1 * nbasis3 + l2) * nabasis;
                            double *dst = &rep[idx2(i, z, rep_size)];
                            for (std::size_t t = 0; t < nabasis; ++t)
                                dst[t] += angular[t] * scale;
                        }
                    }
                }
            }
        }
    }
}

// Dispatch three-body forward
static void three_body_forward(
    ThreeBodyType type, std::size_t natoms, std::size_t nelements, std::size_t nbasis2,
    std::size_t nbasis3, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double zeta, double acut, double three_body_decay,
    double three_body_weight, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep
) {
    const std::size_t nbasis3_minus = Rs3_minus.size();
    switch (type) {
        case ThreeBodyType::OddFourier_Rbar:
            three_body_a1_forward(
                natoms, nelements, nbasis2, nbasis3, nabasis, coords, D, rdecay3, Rs3, eta3,
                zeta, acut, three_body_decay, three_body_weight, elem_of_atom, rep_size, rep
            );
            break;
        case ThreeBodyType::CosineSeries_Rbar:
            three_body_a2_forward(
                natoms, nelements, nbasis2, nbasis3, nabasis, coords, D, rdecay3, Rs3, eta3,
                acut, three_body_decay, three_body_weight, elem_of_atom, rep_size, rep
            );
            break;
        case ThreeBodyType::OddFourier_SplitR:
            three_body_a3_forward(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, rdecay3,
                Rs3, Rs3_minus, eta3, eta3_minus, zeta, acut, three_body_decay, three_body_weight,
                elem_of_atom, rep_size, rep
            );
            break;
        case ThreeBodyType::CosineSeries_SplitR:
            three_body_a4_forward(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, rdecay3,
                Rs3, Rs3_minus, eta3, eta3_minus, acut, three_body_decay, three_body_weight,
                elem_of_atom, rep_size, rep
            );
            break;
        case ThreeBodyType::CosineSeries_SplitR_NoATM:
            three_body_a5_forward(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, rdecay3,
                Rs3, Rs3_minus, eta3, eta3_minus, acut, three_body_weight, elem_of_atom,
                rep_size, rep
            );
            break;
        case ThreeBodyType::OddFourier_ElementResolved:
            three_body_a6_forward(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, rdecay3,
                Rs3, Rs3_minus, eta3, eta3_minus, zeta, acut, three_body_decay, three_body_weight,
                elem_of_atom, rep_size, rep
            );
            break;
        case ThreeBodyType::CosineSeries_ElementResolved:
            three_body_a7_forward(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, rdecay3,
                Rs3, Rs3_minus, eta3, eta3_minus, acut, three_body_decay, three_body_weight,
                elem_of_atom, rep_size, rep
            );
            break;
    }
}

// ==================== Three-body gradient ====================

// A1: OddFourier + Rbar + ATM gradient (ported from existing fchl19_repr.cpp)
static void three_body_a1_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nabasis, const std::vector<double> &coords, const std::vector<double> &D,
    const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, double eta3, double zeta, double acut,
    double three_body_decay, double three_body_weight, const std::vector<int> &elem_of_atom,
    std::size_t rep_size, std::vector<double> &rep, std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;
    const double ang_w_pre = std::exp(-0.5 * zeta * zeta) * 2.0;
    const double tbd_over_w_pre =
        (three_body_weight != 0.0) ? (three_body_decay / three_body_weight) : 0.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> radial(nbasis3), d_radial(nbasis3);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];
        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;

            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);

                for (std::size_t l = 0; l < nbasis3; ++l) {
                    const double rbar = 0.5 * (rij + rik) - Rs3[l];
                    const double base = std::exp(-eta3 * rbar * rbar);
                    d_radial[l] = base * eta3 * rbar;
                    radial[l] = base;
                }

                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w_pre * cos_i;
                d_angular[0] = ang_w_pre * sin_i;
                if (nabasis >= 2) {
                    angular[1] = ang_w_pre * sin_i;
                    d_angular[1] = -ang_w_pre * cos_i;
                }

                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double s_ik =
                    -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);

                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;
                const double atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik;
                const double atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk;
                const double atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik;

                const double vi = dot;
                const double vj =
                    (Cx - Ax) * (Bx - Ax) + (Cy - Ay) * (By - Ay) + (Cz - Az) * (Bz - Az);
                const double vk =
                    (Bx - Cx) * (Ax - Cx) + (By - Cy) * (Ay - Cy) + (Bz - Cz) * (Az - Cz);

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
                const double d_atm_jj0 = 2 * Ax - Bx - Cx -
                                          vj * ((Ax - Bx) * invrij2 + (Ax - Cx) * invrjk * invrjk);
                const double d_atm_jj1 = 2 * Ay - By - Cy -
                                          vj * ((Ay - By) * invrij2 + (Ay - Cy) * invrjk * invrjk);
                const double d_atm_jj2 = 2 * Az - Bz - Cz -
                                          vj * ((Az - Bz) * invrij2 + (Az - Cz) * invrjk * invrjk);
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
                    2 * Cx - Ax - Bx -
                    vk * ((Cx - Ax) * invrjk * invrjk + (Cx - Bx) * invrik2);
                const double d_atm_kk1 =
                    2 * Cy - Ay - By -
                    vk * ((Cy - Ay) * invrjk * invrjk + (Cy - By) * invrik2);
                const double d_atm_kk2 =
                    2 * Cz - Az - Bz -
                    vk * ((Cz - Az) * invrjk * invrjk + (Cz - Bz) * invrik2);

                const double atm_tbd = atm * tbd_over_w_pre;
                const double d_extra_i0 =
                    ((Ax - Bx) * invrij2 + (Cx - Bx) * invrik2) * atm_tbd;
                const double d_extra_i1 =
                    ((Ay - By) * invrij2 + (Cy - By) * invrik2) * atm_tbd;
                const double d_extra_i2 =
                    ((Az - Bz) * invrij2 + (Cz - Bz) * invrik2) * atm_tbd;
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

                const std::size_t pair_idx0 = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = pair_idx0 * (nbasis3 * nabasis);

                const double inv_denom_ang = 1.0 / denom_ang;
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

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

                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;

                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                const double decay_prod = decay_ij * decay_ik;

                for (std::size_t l = 0; l < nbasis3; ++l) {
                    const double scale_val = radial[l] * atm * decay_prod;
                    const double scale_ang = decay_prod * radial[l];
                    const double d_rad_l = d_radial[l];
                    const std::size_t z0 = base + l * nabasis;

                    for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                        atom_rep[z0 + aidx] += angular[aidx] * scale_val;
                    }

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

                    for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                        const double ang = angular[aidx];
                        const double dang = d_angular[aidx];

                        // dim 0
                        {
                            const double gi = dang * dai0 * scale_ang * atm +
                                              ang * dri0 * atm * decay_prod +
                                              ang * rad_l * atmi0 * decay_prod +
                                              ang * rad_l * dec_i0 * atm;
                            const double gj = dang * daj0 * scale_ang * atm +
                                              ang * drj0 * atm * decay_prod +
                                              ang * rad_l * atmj0 * decay_prod +
                                              ang * rad_l * dec_j0 * atm;
                            const double gk = dang * dak0 * scale_ang * atm +
                                              ang * drk0 * atm * decay_prod +
                                              ang * rad_l * atmk0 * decay_prod +
                                              ang * rad_l * dec_k0 * atm;
                            atom_grad[((z0 + aidx) * natoms + i) * 3 + 0] += gi;
                            atom_grad[((z0 + aidx) * natoms + j) * 3 + 0] += gj;
                            atom_grad[((z0 + aidx) * natoms + k) * 3 + 0] += gk;
                        }
                        // dim 1
                        {
                            const double gi = dang * dai1 * scale_ang * atm +
                                              ang * dri1 * atm * decay_prod +
                                              ang * rad_l * atmi1 * decay_prod +
                                              ang * rad_l * dec_i1 * atm;
                            const double gj = dang * daj1 * scale_ang * atm +
                                              ang * drj1 * atm * decay_prod +
                                              ang * rad_l * atmj1 * decay_prod +
                                              ang * rad_l * dec_j1 * atm;
                            const double gk = dang * dak1 * scale_ang * atm +
                                              ang * drk1 * atm * decay_prod +
                                              ang * rad_l * atmk1 * decay_prod +
                                              ang * rad_l * dec_k1 * atm;
                            atom_grad[((z0 + aidx) * natoms + i) * 3 + 1] += gi;
                            atom_grad[((z0 + aidx) * natoms + j) * 3 + 1] += gj;
                            atom_grad[((z0 + aidx) * natoms + k) * 3 + 1] += gk;
                        }
                        // dim 2
                        {
                            const double gi = dang * dai2 * scale_ang * atm +
                                              ang * dri2 * atm * decay_prod +
                                              ang * rad_l * atmi2 * decay_prod +
                                              ang * rad_l * dec_i2 * atm;
                            const double gj = dang * daj2 * scale_ang * atm +
                                              ang * drj2 * atm * decay_prod +
                                              ang * rad_l * atmj2 * decay_prod +
                                              ang * rad_l * dec_j2 * atm;
                            const double gk = dang * dak2 * scale_ang * atm +
                                              ang * drk2 * atm * decay_prod +
                                              ang * rad_l * atmk2 * decay_prod +
                                              ang * rad_l * dec_k2 * atm;
                            atom_grad[((z0 + aidx) * natoms + i) * 3 + 2] += gi;
                            atom_grad[((z0 + aidx) * natoms + j) * 3 + 2] += gj;
                            atom_grad[((z0 + aidx) * natoms + k) * 3 + 2] += gk;
                        }
                    }
                }
            }
        }

        // scatter into global rep/grad
        for (std::size_t off = 0; off < three_block_size; ++off) {
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        }
        for (std::size_t off = 0; off < three_block_size; ++off) {
            for (std::size_t a = 0; a < natoms; ++a) {
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
            }
        }
    }
}

// ==================== Three-body gradient helpers (A2-A5) ====================

// Shared ATM gradient computation (used by A2/A3/A4; not used by A5)
// Returns the pre-multiplied ATM gradient vectors for atoms i,j,k given the
// triangle geometry.  Caller multiplies by the appropriate radial*decay product.
struct AtmGrad {
    double i0, i1, i2;
    double j0, j1, j2;
    double k0, k1, k2;
};

static inline AtmGrad compute_atm_grad(
    double Ax, double Ay, double Az,
    double Bx, double By, double Bz,
    double Cx, double Cy, double Cz,
    double invrij, double invrik, double invrjk,
    double invrij2, double invrik2,
    double cos_i, double cos_j, double cos_k,
    double three_body_decay, double three_body_weight,
    double atm  // pre-computed ATM value (including weight)
) {
    const double invrjk2 = invrjk * invrjk;

    const double atm_i = (3.0 * cos_j * cos_k) * (atm / three_body_weight) * invrij * invrik /
                         ((atm / three_body_weight) > 0
                              ? 1.0
                              : 1.0);  // just use the ratios directly

    // Recompute partial cos derivatives from the vectors
    // vi = dot(B->A, B->C), vj = dot(A->B, A->C), vk = dot(C->A, C->B) ... using coordinates
    const double vi = (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
    const double vj =
        (Cx - Ax) * (Bx - Ax) + (Cy - Ay) * (By - Ay) + (Cz - Az) * (Bz - Az);
    const double vk =
        (Bx - Cx) * (Ax - Cx) + (By - Cy) * (Ay - Cy) + (Bz - Cz) * (Az - Cz);

    // invr_atm = (rij*rik*rjk)^-decay
    const double invr_atm_raw = std::pow(invrij * invrik * invrjk, three_body_decay);
    const double a_i = (3.0 * cos_j * cos_k) * invr_atm_raw * invrij * invrik;
    const double a_j = (3.0 * cos_k * cos_i) * invr_atm_raw * invrij * invrjk;
    const double a_k = (3.0 * cos_i * cos_j) * invr_atm_raw * invrjk * invrik;

    const double tbd_over_w = three_body_decay / three_body_weight;
    const double atm_val = atm;  // includes weight

    // d_atm matrices (same as A1)
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
        2 * Ax - Bx - Cx - vj * ((Ax - Bx) * invrij2 + (Ax - Cx) * invrjk2);
    const double d_atm_jj1 =
        2 * Ay - By - Cy - vj * ((Ay - By) * invrij2 + (Ay - Cy) * invrjk2);
    const double d_atm_jj2 =
        2 * Az - Bz - Cz - vj * ((Az - Bz) * invrij2 + (Az - Cz) * invrjk2);
    const double d_atm_jk0 = Bx - Cx - vk * (Ax - Cx) * invrjk2;
    const double d_atm_jk1 = By - Cy - vk * (Ay - Cy) * invrjk2;
    const double d_atm_jk2 = Bz - Cz - vk * (Az - Cz) * invrjk2;

    const double d_atm_ki0 = Ax - Bx - vi * (Cx - Bx) * invrik2;
    const double d_atm_ki1 = Ay - By - vi * (Cy - By) * invrik2;
    const double d_atm_ki2 = Az - Bz - vi * (Cz - Bz) * invrik2;
    const double d_atm_kj0 = Bx - Ax - vj * (Cx - Ax) * invrjk2;
    const double d_atm_kj1 = By - Ay - vj * (Cy - Ay) * invrjk2;
    const double d_atm_kj2 = Bz - Az - vj * (Cz - Az) * invrjk2;
    const double d_atm_kk0 =
        2 * Cx - Ax - Bx - vk * ((Cx - Ax) * invrjk2 + (Cx - Bx) * invrik2);
    const double d_atm_kk1 =
        2 * Cy - Ay - By - vk * ((Cy - Ay) * invrjk2 + (Cy - By) * invrik2);
    const double d_atm_kk2 =
        2 * Cz - Az - Bz - vk * ((Cz - Az) * invrjk2 + (Cz - Bz) * invrik2);

    const double atm_tbd = atm_val * tbd_over_w;
    const double d_extra_i0 = ((Ax - Bx) * invrij2 + (Cx - Bx) * invrik2) * atm_tbd;
    const double d_extra_i1 = ((Ay - By) * invrij2 + (Cy - By) * invrik2) * atm_tbd;
    const double d_extra_i2 = ((Az - Bz) * invrij2 + (Cz - Bz) * invrik2) * atm_tbd;
    const double d_extra_j0 =
        ((Bx - Ax) * invrij2 + (Cx - Ax) * invrjk2) * atm_tbd;
    const double d_extra_j1 =
        ((By - Ay) * invrij2 + (Cy - Ay) * invrjk2) * atm_tbd;
    const double d_extra_j2 =
        ((Bz - Az) * invrij2 + (Cz - Az) * invrjk2) * atm_tbd;
    const double d_extra_k0 =
        ((Ax - Cx) * invrjk2 + (Bx - Cx) * invrik2) * atm_tbd;
    const double d_extra_k1 =
        ((Ay - Cy) * invrjk2 + (By - Cy) * invrik2) * atm_tbd;
    const double d_extra_k2 =
        ((Az - Cz) * invrjk2 + (Bz - Cz) * invrik2) * atm_tbd;

    AtmGrad ag;
    ag.i0 = (a_i * d_atm_ii0 + a_j * d_atm_ij0 + a_k * d_atm_ik0 + d_extra_i0) *
            three_body_weight;
    ag.i1 = (a_i * d_atm_ii1 + a_j * d_atm_ij1 + a_k * d_atm_ik1 + d_extra_i1) *
            three_body_weight;
    ag.i2 = (a_i * d_atm_ii2 + a_j * d_atm_ij2 + a_k * d_atm_ik2 + d_extra_i2) *
            three_body_weight;
    ag.j0 = (a_i * d_atm_ji0 + a_j * d_atm_jj0 + a_k * d_atm_jk0 + d_extra_j0) *
            three_body_weight;
    ag.j1 = (a_i * d_atm_ji1 + a_j * d_atm_jj1 + a_k * d_atm_jk1 + d_extra_j1) *
            three_body_weight;
    ag.j2 = (a_i * d_atm_ji2 + a_j * d_atm_jj2 + a_k * d_atm_jk2 + d_extra_j2) *
            three_body_weight;
    ag.k0 = (a_i * d_atm_ki0 + a_j * d_atm_kj0 + a_k * d_atm_kk0 + d_extra_k0) *
            three_body_weight;
    ag.k1 = (a_i * d_atm_ki1 + a_j * d_atm_kj1 + a_k * d_atm_kk1 + d_extra_k1) *
            three_body_weight;
    ag.k2 = (a_i * d_atm_ki2 + a_j * d_atm_kj2 + a_k * d_atm_kk2 + d_extra_k2) *
            three_body_weight;
    return ag;
}

// A2 gradient: CosineSeries_Rbar + ATM
static void three_body_a2_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nabasis, const std::vector<double> &coords, const std::vector<double> &D,
    const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, double eta3, double acut, double three_body_decay,
    double three_body_weight, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep, std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> radial(nbasis3), d_radial(nbasis3);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];

        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double inv_denom_ang = 1.0 / denom_ang;

                // d(cos_i)/d(atom positions) -- same formula as A1
                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

                // Cosine series (Chebyshev T_m) angular basis + analytic derivative
                // Parametrized same as A1: d_angular[m] = dT_m/d(theta_i)
                //   = dT_m/d(cos_i) * d(cos_i)/d(theta_i) = m * U_{m-1}(cos_i) * sin(theta_i)
                // so that  d(angular[m])/d(pos) = d_angular[m] * da?0 / sin_i  (cancels sin_i)
                //         = m * U_{m-1} * d(cos_i)/d(pos)  ✓
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                if (nabasis > 0) { angular[0] = 1.0;   d_angular[0] = 0.0; }
                if (nabasis > 1) { angular[1] = cos_i; d_angular[1] = sin_i; }
                {
                    double Um2 = 1.0, Um1 = 2.0 * cos_i;  // U_0, U_1
                    for (std::size_t m = 2; m < nabasis; ++m) {
                        angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];
                        d_angular[m] = static_cast<double>(m) * Um1 * sin_i;
                        double Um = 2.0 * cos_i * Um1 - Um2;
                        Um2 = Um1;
                        Um1 = Um;
                    }
                }

                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

                AtmGrad ag = compute_atm_grad(
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                    invrij, invrik, invrjk, invrij2, invrik2,
                    cos_i, cos_j, cos_k, three_body_decay, three_body_weight, atm
                );

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double s_ik =
                    -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;

                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;
                const double decay_prod = decay_ij * decay_ik;

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = pair_idx * (nbasis3 * nabasis);

                for (std::size_t l = 0; l < nbasis3; ++l) {
                    const double rbar = 0.5 * (rij + rik) - Rs3[l];
                    const double rad_base = std::exp(-eta3 * rbar * rbar);
                    radial[l] = rad_base;
                    d_radial[l] = rad_base * eta3 * rbar;
                }

                for (std::size_t l = 0; l < nbasis3; ++l) {
                    const double scale_val = radial[l] * atm * decay_prod;
                    const std::size_t z0 = base + l * nabasis;

                    for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                        atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                    const double d_rad_l = d_radial[l];
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
                    const double scale_ang = decay_prod * rad_l;

                    for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                        const double ang = angular[aidx];
                        const double dang = d_angular[aidx];
                        const std::size_t feat = z0 + aidx;

                        atom_grad[(feat * natoms + i) * 3 + 0] +=
                            dang * dai0 * scale_ang * atm + ang * dri0 * atm * decay_prod +
                            ang * rad_l * ag.i0 * decay_prod + ang * rad_l * dec_i0 * atm;
                        atom_grad[(feat * natoms + j) * 3 + 0] +=
                            dang * daj0 * scale_ang * atm + ang * drj0 * atm * decay_prod +
                            ang * rad_l * ag.j0 * decay_prod + ang * rad_l * dec_j0 * atm;
                        atom_grad[(feat * natoms + k) * 3 + 0] +=
                            dang * dak0 * scale_ang * atm + ang * drk0 * atm * decay_prod +
                            ang * rad_l * ag.k0 * decay_prod + ang * rad_l * dec_k0 * atm;

                        atom_grad[(feat * natoms + i) * 3 + 1] +=
                            dang * dai1 * scale_ang * atm + ang * dri1 * atm * decay_prod +
                            ang * rad_l * ag.i1 * decay_prod + ang * rad_l * dec_i1 * atm;
                        atom_grad[(feat * natoms + j) * 3 + 1] +=
                            dang * daj1 * scale_ang * atm + ang * drj1 * atm * decay_prod +
                            ang * rad_l * ag.j1 * decay_prod + ang * rad_l * dec_j1 * atm;
                        atom_grad[(feat * natoms + k) * 3 + 1] +=
                            dang * dak1 * scale_ang * atm + ang * drk1 * atm * decay_prod +
                            ang * rad_l * ag.k1 * decay_prod + ang * rad_l * dec_k1 * atm;

                        atom_grad[(feat * natoms + i) * 3 + 2] +=
                            dang * dai2 * scale_ang * atm + ang * dri2 * atm * decay_prod +
                            ang * rad_l * ag.i2 * decay_prod + ang * rad_l * dec_i2 * atm;
                        atom_grad[(feat * natoms + j) * 3 + 2] +=
                            dang * daj2 * scale_ang * atm + ang * drj2 * atm * decay_prod +
                            ang * rad_l * ag.j2 * decay_prod + ang * rad_l * dec_j2 * atm;
                        atom_grad[(feat * natoms + k) * 3 + 2] +=
                            dang * dak2 * scale_ang * atm + ang * drk2 * atm * decay_prod +
                            ang * rad_l * ag.k2 * decay_prod + ang * rad_l * dec_k2 * atm;
                    }
                }
            }
        }

        for (std::size_t off = 0; off < three_block_size; ++off)
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        for (std::size_t off = 0; off < three_block_size; ++off)
            for (std::size_t a = 0; a < natoms; ++a)
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
    }
}

// Helper: gradient of SplitR radial basis wrt rij, rik
// r_plus = rij + rik, r_minus = |rij - rik| (smoothed with eps)
struct SplitRRadGrad {
    double d_phi_d_rij;
    double d_phi_d_rik;
};

static inline SplitRRadGrad splitr_radial_grad(
    double rij, double rik, double Rs3_plus, double Rs3_minus_val,
    double eta3, double eta3_minus, double phi_p, double phi_m
) {
    const double dp = rij + rik - Rs3_plus;
    const double diff = rij - rik;
    // smoothed |diff|: use sqrt(diff^2 + eps) to avoid discontinuity at diff=0
    const double eps_rm = 1e-8;
    const double abs_diff = std::sqrt(diff * diff + eps_rm);
    const double dm = abs_diff - Rs3_minus_val;

    // d(phi_p)/d(rij) = phi_p * (-2*eta3*(rij+rik-Rs3_plus)) * 1
    // d(phi_p)/d(rik) = same
    const double dphi_p_dp = phi_p * (-2.0 * eta3 * dp);

    // d(abs_diff)/d(rij) = (rij-rik)/abs_diff
    // d(abs_diff)/d(rik) = -(rij-rik)/abs_diff
    const double sign_ij = diff / abs_diff;
    const double dphi_m_dm = phi_m * (-2.0 * eta3_minus * dm);

    SplitRRadGrad g;
    g.d_phi_d_rij = dphi_p_dp * phi_m + phi_p * dphi_m_dm * sign_ij;
    g.d_phi_d_rik = dphi_p_dp * phi_m + phi_p * dphi_m_dm * (-sign_ij);
    return g;
}

// A3 gradient: OddFourier_SplitR + ATM
static void three_body_a3_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double zeta, double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;
    const std::size_t n_harm = nabasis / 2;

    std::vector<double> ang_w(n_harm);
    std::vector<int> ang_o(n_harm);
    for (std::size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2 * l + 1);
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }
    const double ang_w_pre = (n_harm > 0) ? ang_w[0] : 1.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];

        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double inv_denom_ang = 1.0 / denom_ang;

                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

                // OddFourier angular + d_angular (same as A1)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w_pre * cos_i;
                d_angular[0] = ang_w_pre * sin_i;
                if (nabasis >= 2) {
                    angular[1] = ang_w_pre * sin_i;
                    d_angular[1] = -ang_w_pre * cos_i;
                }
                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0, cn_1 = cos_i, sn_1 = sin_i;
                    std::size_t harm_stored = 1;
                    const int max_o = ang_o[n_harm - 1];
                    for (int n = 2; n <= max_o; ++n) {
                        const double cn = two_cos * cn_1 - cn_2;
                        const double sn = two_cos * sn_1 - sn_2;
                        cn_2 = cn_1; sn_2 = sn_1; cn_1 = cn; sn_1 = sn;
                        if (n == ang_o[harm_stored]) {
                            angular[2 * harm_stored] = ang_w[harm_stored] * cn;
                            angular[2 * harm_stored + 1] = ang_w[harm_stored] * sn;
                            d_angular[2 * harm_stored] = ang_w[harm_stored] * sn;
                            d_angular[2 * harm_stored + 1] = -ang_w[harm_stored] * cn;
                            if (++harm_stored >= n_harm) break;
                        }
                    }
                }

                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

                AtmGrad ag = compute_atm_grad(
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                    invrij, invrik, invrjk, invrij2, invrik2,
                    cos_i, cos_j, cos_k, three_body_decay, three_body_weight, atm
                );

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double s_ik =
                    -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;
                const double decay_prod = decay_ij * decay_ik;

                // dr/dx_i components: rij depends on i,j; rik depends on i,k
                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = pair_idx * (nbasis3 * nbasis3_minus * nabasis);

                for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                    const double dp = rij + rik - Rs3[l1];
                    const double phi_p = std::exp(-eta3 * dp * dp);
                    for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                        const double phi_m_base =
                            std::exp(-eta3_minus *
                                     std::pow(std::abs(rij - rik) - Rs3_minus[l2], 2));
                        const double radial_val = phi_p * phi_m_base * decay_prod;
                        const double scale_val = radial_val * atm;
                        const std::size_t z0 = base + (l1 * nbasis3_minus + l2) * nabasis;

                        for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                            atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                        SplitRRadGrad srg =
                            splitr_radial_grad(rij, rik, Rs3[l1], Rs3_minus[l2], eta3,
                                               eta3_minus, phi_p, phi_m_base);
                        // d(radial)/d(rij) = (d(phi_p*phi_m)/d(rij))*decay_prod
                        // d(radial)/d(rik) = (d(phi_p*phi_m)/d(rik))*decay_prod
                        const double drad_drij = srg.d_phi_d_rij * decay_prod;
                        const double drad_drik = srg.d_phi_d_rik * decay_prod;

                        // d(rij)/d(pos_a) vectors (unit vector from j to i via center i)
                        // dr_ij/d(B) = BmA (B is atom i center)
                        // dr_ij/d(A) = -BmA
                        // dr_ik/d(B) = BmC
                        // dr_ik/d(C) = -BmC
                        const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                        const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                        const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                        const double drad_j0 = drad_drij * (-BmA0);
                        const double drad_j1 = drad_drij * (-BmA1);
                        const double drad_j2 = drad_drij * (-BmA2);
                        const double drad_k0 = drad_drik * (-BmC0);
                        const double drad_k1 = drad_drik * (-BmC1);
                        const double drad_k2 = drad_drik * (-BmC2);

                        const double scale_ang = radial_val;

                        for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                            const double ang = angular[aidx];
                            const double dang = d_angular[aidx];
                            const std::size_t feat = z0 + aidx;

                            atom_grad[(feat * natoms + i) * 3 + 0] +=
                                dang * dai0 * scale_ang * atm + ang * drad_i0 * atm +
                                ang * radial_val * ag.i0 + ang * radial_val * dec_i0 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + j) * 3 + 0] +=
                                dang * daj0 * scale_ang * atm + ang * drad_j0 * atm +
                                ang * radial_val * ag.j0 + ang * radial_val * dec_j0 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + k) * 3 + 0] +=
                                dang * dak0 * scale_ang * atm + ang * drad_k0 * atm +
                                ang * radial_val * ag.k0 + ang * radial_val * dec_k0 * atm /
                                    decay_prod;

                            atom_grad[(feat * natoms + i) * 3 + 1] +=
                                dang * dai1 * scale_ang * atm + ang * drad_i1 * atm +
                                ang * radial_val * ag.i1 + ang * radial_val * dec_i1 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + j) * 3 + 1] +=
                                dang * daj1 * scale_ang * atm + ang * drad_j1 * atm +
                                ang * radial_val * ag.j1 + ang * radial_val * dec_j1 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + k) * 3 + 1] +=
                                dang * dak1 * scale_ang * atm + ang * drad_k1 * atm +
                                ang * radial_val * ag.k1 + ang * radial_val * dec_k1 * atm /
                                    decay_prod;

                            atom_grad[(feat * natoms + i) * 3 + 2] +=
                                dang * dai2 * scale_ang * atm + ang * drad_i2 * atm +
                                ang * radial_val * ag.i2 + ang * radial_val * dec_i2 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + j) * 3 + 2] +=
                                dang * daj2 * scale_ang * atm + ang * drad_j2 * atm +
                                ang * radial_val * ag.j2 + ang * radial_val * dec_j2 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + k) * 3 + 2] +=
                                dang * dak2 * scale_ang * atm + ang * drad_k2 * atm +
                                ang * radial_val * ag.k2 + ang * radial_val * dec_k2 * atm /
                                    decay_prod;
                        }
                    }
                }
            }
        }

        for (std::size_t off = 0; off < three_block_size; ++off)
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        for (std::size_t off = 0; off < three_block_size; ++off)
            for (std::size_t a = 0; a < natoms; ++a)
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
    }
}

// A4 gradient: CosineSeries_SplitR + ATM
// Identical structure to A3 gradient but with cosine angular basis instead of OddFourier
static void three_body_a4_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];

        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double inv_denom_ang = 1.0 / denom_ang;

                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

                // Cosine series + Chebyshev derivative (same as A2):
                // d_angular[m] = m * U_{m-1}(cos_i) * sin(theta_i)
                const double sin_i4 = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                if (nabasis > 0) { angular[0] = 1.0;   d_angular[0] = 0.0; }
                if (nabasis > 1) { angular[1] = cos_i; d_angular[1] = sin_i4; }
                {
                    double Um2 = 1.0, Um1 = 2.0 * cos_i;
                    for (std::size_t m = 2; m < nabasis; ++m) {
                        angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];
                        d_angular[m] = static_cast<double>(m) * Um1 * sin_i4;
                        double Um = 2.0 * cos_i * Um1 - Um2;
                        Um2 = Um1; Um1 = Um;
                    }
                }

                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

                AtmGrad ag = compute_atm_grad(
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                    invrij, invrik, invrjk, invrij2, invrik2,
                    cos_i, cos_j, cos_k, three_body_decay, three_body_weight, atm
                );

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double s_ik =
                    -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;
                const double decay_prod = decay_ij * decay_ik;

                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = pair_idx * (nbasis3 * nbasis3_minus * nabasis);

                for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                    const double dp = rij + rik - Rs3[l1];
                    const double phi_p = std::exp(-eta3 * dp * dp);
                    for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                        const double phi_m_base =
                            std::exp(-eta3_minus *
                                     std::pow(std::abs(rij - rik) - Rs3_minus[l2], 2));
                        const double radial_val = phi_p * phi_m_base * decay_prod;
                        const double scale_val = radial_val * atm;
                        const std::size_t z0 = base + (l1 * nbasis3_minus + l2) * nabasis;

                        for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                            atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                        SplitRRadGrad srg =
                            splitr_radial_grad(rij, rik, Rs3[l1], Rs3_minus[l2], eta3,
                                               eta3_minus, phi_p, phi_m_base);
                        const double drad_drij = srg.d_phi_d_rij * decay_prod;
                        const double drad_drik = srg.d_phi_d_rik * decay_prod;

                        const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                        const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                        const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                        const double drad_j0 = drad_drij * (-BmA0);
                        const double drad_j1 = drad_drij * (-BmA1);
                        const double drad_j2 = drad_drij * (-BmA2);
                        const double drad_k0 = drad_drik * (-BmC0);
                        const double drad_k1 = drad_drik * (-BmC1);
                        const double drad_k2 = drad_drik * (-BmC2);

                        const double scale_ang = radial_val;

                        for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                            const double ang = angular[aidx];
                            const double dang = d_angular[aidx];
                            const std::size_t feat = z0 + aidx;

                            atom_grad[(feat * natoms + i) * 3 + 0] +=
                                dang * dai0 * scale_ang * atm + ang * drad_i0 * atm +
                                ang * radial_val * ag.i0 + ang * radial_val * dec_i0 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + j) * 3 + 0] +=
                                dang * daj0 * scale_ang * atm + ang * drad_j0 * atm +
                                ang * radial_val * ag.j0 + ang * radial_val * dec_j0 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + k) * 3 + 0] +=
                                dang * dak0 * scale_ang * atm + ang * drad_k0 * atm +
                                ang * radial_val * ag.k0 + ang * radial_val * dec_k0 * atm /
                                    decay_prod;

                            atom_grad[(feat * natoms + i) * 3 + 1] +=
                                dang * dai1 * scale_ang * atm + ang * drad_i1 * atm +
                                ang * radial_val * ag.i1 + ang * radial_val * dec_i1 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + j) * 3 + 1] +=
                                dang * daj1 * scale_ang * atm + ang * drad_j1 * atm +
                                ang * radial_val * ag.j1 + ang * radial_val * dec_j1 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + k) * 3 + 1] +=
                                dang * dak1 * scale_ang * atm + ang * drad_k1 * atm +
                                ang * radial_val * ag.k1 + ang * radial_val * dec_k1 * atm /
                                    decay_prod;

                            atom_grad[(feat * natoms + i) * 3 + 2] +=
                                dang * dai2 * scale_ang * atm + ang * drad_i2 * atm +
                                ang * radial_val * ag.i2 + ang * radial_val * dec_i2 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + j) * 3 + 2] +=
                                dang * daj2 * scale_ang * atm + ang * drad_j2 * atm +
                                ang * radial_val * ag.j2 + ang * radial_val * dec_j2 * atm /
                                    decay_prod;
                            atom_grad[(feat * natoms + k) * 3 + 2] +=
                                dang * dak2 * scale_ang * atm + ang * drad_k2 * atm +
                                ang * radial_val * ag.k2 + ang * radial_val * dec_k2 * atm /
                                    decay_prod;
                        }
                    }
                }
            }
        }

        for (std::size_t off = 0; off < three_block_size; ++off)
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        for (std::size_t off = 0; off < three_block_size; ++off)
            for (std::size_t a = 0; a < natoms; ++a)
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
    }
}

// A5 gradient: CosineSeries_SplitR_NoATM (ATM factor = 1, weight = three_body_weight)
static void three_body_a5_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double acut, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];

        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double inv_denom_ang = 1.0 / denom_ang;

                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

                // Cosine series + Chebyshev derivative (same as A2):
                // d_angular[m] = m * U_{m-1}(cos_i) * sin(theta_i)
                const double sin_i5 = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                if (nabasis > 0) { angular[0] = 1.0;   d_angular[0] = 0.0; }
                if (nabasis > 1) { angular[1] = cos_i; d_angular[1] = sin_i5; }
                {
                    double Um2 = 1.0, Um1 = 2.0 * cos_i;
                    for (std::size_t m = 2; m < nabasis; ++m) {
                        angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];
                        d_angular[m] = static_cast<double>(m) * Um1 * sin_i5;
                        double Um = 2.0 * cos_i * Um1 - Um2;
                        Um2 = Um1; Um1 = Um;
                    }
                }

                 const double decay_ij = rdecay3[idx2(i, j, natoms)];
                 const double decay_ik = rdecay3[idx2(i, k, natoms)];
                 const double s_ik =
                     -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;
                const double decay_prod = decay_ij * decay_ik;

                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                const std::size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const std::size_t base = pair_idx * (nbasis3 * nbasis3_minus * nabasis);

                for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                    const double dp = rij + rik - Rs3[l1];
                    const double phi_p = std::exp(-eta3 * dp * dp);
                    for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                        const double phi_m_base =
                            std::exp(-eta3_minus *
                                     std::pow(std::abs(rij - rik) - Rs3_minus[l2], 2));
                        const double radial_val = phi_p * phi_m_base * decay_prod;
                        const std::size_t z0 = base + (l1 * nbasis3_minus + l2) * nabasis;

                        for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                            atom_rep[z0 + aidx] += angular[aidx] * radial_val * three_body_weight;

                        SplitRRadGrad srg =
                            splitr_radial_grad(rij, rik, Rs3[l1], Rs3_minus[l2], eta3,
                                               eta3_minus, phi_p, phi_m_base);
                        const double drad_drij = srg.d_phi_d_rij * decay_prod;
                        const double drad_drik = srg.d_phi_d_rik * decay_prod;

                        const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                        const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                        const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                        const double drad_j0 = drad_drij * (-BmA0);
                        const double drad_j1 = drad_drij * (-BmA1);
                        const double drad_j2 = drad_drij * (-BmA2);
                        const double drad_k0 = drad_drik * (-BmC0);
                        const double drad_k1 = drad_drik * (-BmC1);
                        const double drad_k2 = drad_drik * (-BmC2);

                        for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                            const double ang = angular[aidx];
                            const double dang = d_angular[aidx];
                            const std::size_t feat = z0 + aidx;
                            const double w = three_body_weight;

                            // No ATM: d(ang * radial * w) = dang*d(cos_i) * radial * w
                            //                              + ang * d(radial) * w
                            // d(radial) includes d(decay) and d(phi) contributions
                            // For decay: d(decay_prod)/d(atom) = dec_X / decay_prod * radial
                            //   but simpler: radial = phi*decay_prod, so
                            //   d(radial)/d(pos) = dphi * decay_prod * drij/dpos
                            //                    + phi * d(decay_prod)/dpos
                            // We incorporate d(decay)/dpos via dec_X which are
                            // d(decay_prod)/d(rij) * drij/dpos etc.
                            // Actually dec_X = d(decay_ij * decay_ik)/d(pos_X) already
                            // Let total_rad_i = drad_i (phi part only) + phi_p*phi_m*dec_i
                            const double phi_val = phi_p * phi_m_base;
                            const double total_drad_i0 =
                                drad_i0 + phi_val * dec_i0;
                            const double total_drad_i1 =
                                drad_i1 + phi_val * dec_i1;
                            const double total_drad_i2 =
                                drad_i2 + phi_val * dec_i2;
                            const double total_drad_j0 =
                                drad_j0 + phi_val * dec_j0;
                            const double total_drad_j1 =
                                drad_j1 + phi_val * dec_j1;
                            const double total_drad_j2 =
                                drad_j2 + phi_val * dec_j2;
                            const double total_drad_k0 =
                                drad_k0 + phi_val * dec_k0;
                            const double total_drad_k1 =
                                drad_k1 + phi_val * dec_k1;
                            const double total_drad_k2 =
                                drad_k2 + phi_val * dec_k2;

                            atom_grad[(feat * natoms + i) * 3 + 0] +=
                                w * (dang * dai0 * radial_val + ang * total_drad_i0);
                            atom_grad[(feat * natoms + j) * 3 + 0] +=
                                w * (dang * daj0 * radial_val + ang * total_drad_j0);
                            atom_grad[(feat * natoms + k) * 3 + 0] +=
                                w * (dang * dak0 * radial_val + ang * total_drad_k0);

                            atom_grad[(feat * natoms + i) * 3 + 1] +=
                                w * (dang * dai1 * radial_val + ang * total_drad_i1);
                            atom_grad[(feat * natoms + j) * 3 + 1] +=
                                w * (dang * daj1 * radial_val + ang * total_drad_j1);
                            atom_grad[(feat * natoms + k) * 3 + 1] +=
                                w * (dang * dak1 * radial_val + ang * total_drad_k1);

                            atom_grad[(feat * natoms + i) * 3 + 2] +=
                                w * (dang * dai2 * radial_val + ang * total_drad_i2);
                            atom_grad[(feat * natoms + j) * 3 + 2] +=
                                w * (dang * daj2 * radial_val + ang * total_drad_j2);
                            atom_grad[(feat * natoms + k) * 3 + 2] +=
                                w * (dang * dak2 * radial_val + ang * total_drad_k2);
                        }
                    }
                }
            }
        }

        for (std::size_t off = 0; off < three_block_size; ++off)
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        for (std::size_t off = 0; off < three_block_size; ++off)
            for (std::size_t a = 0; a < natoms; ++a)
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
    }
}

// A6 gradient: OddFourier_ElementResolved + ATM
// B==C (diagonal): SplitR radial basis with er_block_offset diagonal block
// B!=C (off-diagonal): independent Gaussian in r_ij and r_ik, two ordered blocks
static void three_body_a6_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double zeta, double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;
    const std::size_t n_harm = nabasis / 2;

    std::vector<double> ang_w(n_harm);
    std::vector<int> ang_o(n_harm);
    for (std::size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2 * l + 1);
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }
    const double ang_w_pre = (n_harm > 0) ? ang_w[0] : 1.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];

        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double inv_denom_ang = 1.0 / denom_ang;

                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

                // OddFourier angular + derivative (same as A3)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                angular[0] = ang_w_pre * cos_i;
                d_angular[0] = ang_w_pre * sin_i;
                if (nabasis >= 2) {
                    angular[1] = ang_w_pre * sin_i;
                    d_angular[1] = -ang_w_pre * cos_i;
                }
                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0, cn_1 = cos_i, sn_1 = sin_i;
                    std::size_t harm_stored = 1;
                    const int max_o = ang_o[n_harm - 1];
                    for (int n = 2; n <= max_o; ++n) {
                        const double cn = two_cos * cn_1 - cn_2;
                        const double sn = two_cos * sn_1 - sn_2;
                        cn_2 = cn_1; sn_2 = sn_1; cn_1 = cn; sn_1 = sn;
                        if (n == ang_o[harm_stored]) {
                            angular[2 * harm_stored] = ang_w[harm_stored] * cn;
                            angular[2 * harm_stored + 1] = ang_w[harm_stored] * sn;
                            d_angular[2 * harm_stored] = ang_w[harm_stored] * sn;
                            d_angular[2 * harm_stored + 1] = -ang_w[harm_stored] * cn;
                            if (++harm_stored >= n_harm) break;
                        }
                    }
                }

                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

                AtmGrad ag = compute_atm_grad(
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                    invrij, invrik, invrjk, invrij2, invrik2,
                    cos_i, cos_j, cos_k, three_body_decay, three_body_weight, atm
                );

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double s_ik =
                    -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;
                const double decay_prod = decay_ij * decay_ik;

                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                if (elem_j == elem_k) {
                    // Diagonal (B==C): SplitR radial basis
                    const std::size_t block_off =
                        er_block_offset(nelements, elem_j, elem_k, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base = block_off;

                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dp = rij + rik - Rs3[l1];
                        const double phi_p = std::exp(-eta3 * dp * dp);
                        for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                            const double phi_m_base =
                                std::exp(-eta3_minus *
                                         std::pow(std::abs(rij - rik) - Rs3_minus[l2], 2));
                            const double radial_val = phi_p * phi_m_base * decay_prod;
                            const double scale_val = radial_val * atm;
                            const std::size_t z0 = base + (l1 * nbasis3_minus + l2) * nabasis;

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                                atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                            SplitRRadGrad srg =
                                splitr_radial_grad(rij, rik, Rs3[l1], Rs3_minus[l2], eta3,
                                                   eta3_minus, phi_p, phi_m_base);
                            const double drad_drij = srg.d_phi_d_rij * decay_prod;
                            const double drad_drik = srg.d_phi_d_rik * decay_prod;

                            const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                            const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                            const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                            const double drad_j0 = drad_drij * (-BmA0);
                            const double drad_j1 = drad_drij * (-BmA1);
                            const double drad_j2 = drad_drij * (-BmA2);
                            const double drad_k0 = drad_drik * (-BmC0);
                            const double drad_k1 = drad_drik * (-BmC1);
                            const double drad_k2 = drad_drik * (-BmC2);

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                                const double ang = angular[aidx];
                                const double dang = d_angular[aidx];
                                const std::size_t feat = z0 + aidx;

                                atom_grad[(feat * natoms + i) * 3 + 0] +=
                                    dang * dai0 * radial_val * atm + ang * drad_i0 * atm +
                                    ang * radial_val * ag.i0 +
                                    ang * radial_val * dec_i0 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 0] +=
                                    dang * daj0 * radial_val * atm + ang * drad_j0 * atm +
                                    ang * radial_val * ag.j0 +
                                    ang * radial_val * dec_j0 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 0] +=
                                    dang * dak0 * radial_val * atm + ang * drad_k0 * atm +
                                    ang * radial_val * ag.k0 +
                                    ang * radial_val * dec_k0 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 1] +=
                                    dang * dai1 * radial_val * atm + ang * drad_i1 * atm +
                                    ang * radial_val * ag.i1 +
                                    ang * radial_val * dec_i1 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 1] +=
                                    dang * daj1 * radial_val * atm + ang * drad_j1 * atm +
                                    ang * radial_val * ag.j1 +
                                    ang * radial_val * dec_j1 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 1] +=
                                    dang * dak1 * radial_val * atm + ang * drad_k1 * atm +
                                    ang * radial_val * ag.k1 +
                                    ang * radial_val * dec_k1 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 2] +=
                                    dang * dai2 * radial_val * atm + ang * drad_i2 * atm +
                                    ang * radial_val * ag.i2 +
                                    ang * radial_val * dec_i2 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 2] +=
                                    dang * daj2 * radial_val * atm + ang * drad_j2 * atm +
                                    ang * radial_val * ag.j2 +
                                    ang * radial_val * dec_j2 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 2] +=
                                    dang * dak2 * radial_val * atm + ang * drad_k2 * atm +
                                    ang * radial_val * ag.k2 +
                                    ang * radial_val * dec_k2 * atm / decay_prod;
                            }
                        }
                    }
                } else {
                    // Off-diagonal (B!=C): ordered (r_ij, r_ik) product basis
                    // Both blocks (elem_j,elem_k) and (elem_k,elem_j) get contributions.

                    // --- Block (elem_j, elem_k): outer=r_ij, inner=r_ik ---
                    const std::size_t block_off_jk =
                        er_block_offset(nelements, elem_j, elem_k, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base_jk = block_off_jk;

                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dj = rij - Rs3[l1];
                        const double phi_j = std::exp(-eta3 * dj * dj);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dk = rik - Rs3[l2];
                            const double phi_k = std::exp(-eta3 * dk * dk);
                            const double radial_val = phi_j * phi_k * decay_prod;
                            const double scale_val = radial_val * atm;
                            const std::size_t z0 = base_jk + (l1 * nbasis3 + l2) * nabasis;

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                                atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                            // d(phi_j)/d(rij) = phi_j * (-2*eta3*dj)
                            // d(phi_k)/d(rik) = phi_k * (-2*eta3*dk)
                            // d(radial)/d(rij) = phi_k * phi_j * (-2*eta3*dj) * decay_prod
                            // d(radial)/d(rik) = phi_j * phi_k * (-2*eta3*dk) * decay_prod
                            const double drad_drij = phi_j * (-2.0 * eta3 * dj) * phi_k * decay_prod;
                            const double drad_drik = phi_j * phi_k * (-2.0 * eta3 * dk) * decay_prod;

                            const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                            const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                            const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                            const double drad_j0 = drad_drij * (-BmA0);
                            const double drad_j1 = drad_drij * (-BmA1);
                            const double drad_j2 = drad_drij * (-BmA2);
                            const double drad_k0 = drad_drik * (-BmC0);
                            const double drad_k1 = drad_drik * (-BmC1);
                            const double drad_k2 = drad_drik * (-BmC2);

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                                const double ang = angular[aidx];
                                const double dang = d_angular[aidx];
                                const std::size_t feat = z0 + aidx;

                                atom_grad[(feat * natoms + i) * 3 + 0] +=
                                    dang * dai0 * radial_val * atm + ang * drad_i0 * atm +
                                    ang * radial_val * ag.i0 +
                                    ang * radial_val * dec_i0 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 0] +=
                                    dang * daj0 * radial_val * atm + ang * drad_j0 * atm +
                                    ang * radial_val * ag.j0 +
                                    ang * radial_val * dec_j0 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 0] +=
                                    dang * dak0 * radial_val * atm + ang * drad_k0 * atm +
                                    ang * radial_val * ag.k0 +
                                    ang * radial_val * dec_k0 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 1] +=
                                    dang * dai1 * radial_val * atm + ang * drad_i1 * atm +
                                    ang * radial_val * ag.i1 +
                                    ang * radial_val * dec_i1 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 1] +=
                                    dang * daj1 * radial_val * atm + ang * drad_j1 * atm +
                                    ang * radial_val * ag.j1 +
                                    ang * radial_val * dec_j1 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 1] +=
                                    dang * dak1 * radial_val * atm + ang * drad_k1 * atm +
                                    ang * radial_val * ag.k1 +
                                    ang * radial_val * dec_k1 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 2] +=
                                    dang * dai2 * radial_val * atm + ang * drad_i2 * atm +
                                    ang * radial_val * ag.i2 +
                                    ang * radial_val * dec_i2 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 2] +=
                                    dang * daj2 * radial_val * atm + ang * drad_j2 * atm +
                                    ang * radial_val * ag.j2 +
                                    ang * radial_val * dec_j2 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 2] +=
                                    dang * dak2 * radial_val * atm + ang * drad_k2 * atm +
                                    ang * radial_val * ag.k2 +
                                    ang * radial_val * dec_k2 * atm / decay_prod;
                            }
                        }
                    }

                    // --- Block (elem_k, elem_j): outer=r_ik (l1), inner=r_ij (l2) ---
                    const std::size_t block_off_kj =
                        er_block_offset(nelements, elem_k, elem_j, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base_kj = block_off_kj;

                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dk = rik - Rs3[l1];
                        const double phi_k = std::exp(-eta3 * dk * dk);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dj = rij - Rs3[l2];
                            const double phi_j = std::exp(-eta3 * dj * dj);
                            const double radial_val = phi_k * phi_j * decay_prod;
                            const double scale_val = radial_val * atm;
                            const std::size_t z0 = base_kj + (l1 * nbasis3 + l2) * nabasis;

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                                atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                            // Swapped: outer=r_ik (l1), inner=r_ij (l2)
                            // d(phi_k)/d(rik) = phi_k * (-2*eta3*dk)
                            // d(phi_j)/d(rij) = phi_j * (-2*eta3*dj)
                            const double drad_drij = phi_k * phi_j * (-2.0 * eta3 * dj) * decay_prod;
                            const double drad_drik = phi_k * (-2.0 * eta3 * dk) * phi_j * decay_prod;

                            const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                            const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                            const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                            const double drad_j0 = drad_drij * (-BmA0);
                            const double drad_j1 = drad_drij * (-BmA1);
                            const double drad_j2 = drad_drij * (-BmA2);
                            const double drad_k0 = drad_drik * (-BmC0);
                            const double drad_k1 = drad_drik * (-BmC1);
                            const double drad_k2 = drad_drik * (-BmC2);

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                                const double ang = angular[aidx];
                                const double dang = d_angular[aidx];
                                const std::size_t feat = z0 + aidx;

                                atom_grad[(feat * natoms + i) * 3 + 0] +=
                                    dang * dai0 * radial_val * atm + ang * drad_i0 * atm +
                                    ang * radial_val * ag.i0 +
                                    ang * radial_val * dec_i0 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 0] +=
                                    dang * daj0 * radial_val * atm + ang * drad_j0 * atm +
                                    ang * radial_val * ag.j0 +
                                    ang * radial_val * dec_j0 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 0] +=
                                    dang * dak0 * radial_val * atm + ang * drad_k0 * atm +
                                    ang * radial_val * ag.k0 +
                                    ang * radial_val * dec_k0 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 1] +=
                                    dang * dai1 * radial_val * atm + ang * drad_i1 * atm +
                                    ang * radial_val * ag.i1 +
                                    ang * radial_val * dec_i1 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 1] +=
                                    dang * daj1 * radial_val * atm + ang * drad_j1 * atm +
                                    ang * radial_val * ag.j1 +
                                    ang * radial_val * dec_j1 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 1] +=
                                    dang * dak1 * radial_val * atm + ang * drad_k1 * atm +
                                    ang * radial_val * ag.k1 +
                                    ang * radial_val * dec_k1 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 2] +=
                                    dang * dai2 * radial_val * atm + ang * drad_i2 * atm +
                                    ang * radial_val * ag.i2 +
                                    ang * radial_val * dec_i2 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 2] +=
                                    dang * daj2 * radial_val * atm + ang * drad_j2 * atm +
                                    ang * radial_val * ag.j2 +
                                    ang * radial_val * dec_j2 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 2] +=
                                    dang * dak2 * radial_val * atm + ang * drad_k2 * atm +
                                    ang * radial_val * ag.k2 +
                                    ang * radial_val * dec_k2 * atm / decay_prod;
                            }
                        }
                    }
                }  // end if elem_j == elem_k
            }
        }

        for (std::size_t off = 0; off < three_block_size; ++off)
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        for (std::size_t off = 0; off < three_block_size; ++off)
            for (std::size_t a = 0; a < natoms; ++a)
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
    }
}

// A7 gradient: CosineSeries_ElementResolved + ATM
// Identical structure to A6 gradient but with cosine (Chebyshev) angular basis instead of OddFourier
static void three_body_a7_grad(
    std::size_t natoms, std::size_t nelements, std::size_t nbasis2, std::size_t nbasis3,
    std::size_t nbasis3_minus, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double acut, double three_body_decay, double three_body_weight,
    const std::vector<int> &elem_of_atom, std::size_t rep_size, std::vector<double> &rep,
    std::vector<double> &grad
) {
    const std::size_t three_offset = nelements * nbasis2;
    const std::size_t three_block_size = rep_size - three_offset;
    const double inv_acut = (acut > 0) ? 1.0 / acut : 0.0;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < natoms; ++i) {
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double *rb = &coords[3 * i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];

        for (std::size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i, j, natoms)];
            if (rij > acut) continue;
            const double rij2 = D2[idx2(i, j, natoms)];
            const double invrij = invD[idx2(i, j, natoms)];
            const double invrij2 = invD2[idx2(i, j, natoms)];
            const int elem_j = elem_of_atom[j];
            const double *ra = &coords[3 * j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;
            const double s_ij = -M_PI * std::sin(M_PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for (std::size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i, k, natoms)];
                if (rik > acut) continue;
                const double rik2 = D2[idx2(i, k, natoms)];
                const double invrik = invD[idx2(i, k, natoms)];
                const double invrik2 = invD2[idx2(i, k, natoms)];
                const double invrjk = invD[idx2(j, k, natoms)];
                const int elem_k = elem_of_atom[k];
                const double *rc = &coords[3 * k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                double cos_i = eij0 * eik0 + eij1 * eik1 + eij2 * eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0 * ejk0 + eij1 * ejk1 + eij2 * ejk2);
                const double cos_k = eik0 * ejk0 + eik1 * ejk1 + eik2 * ejk2;

                const double dot =
                    (Ax - Bx) * (Cx - Bx) + (Ay - By) * (Cy - By) + (Az - Bz) * (Cz - Bz);
                const double denom_ang = std::sqrt(std::max(1e-10, rij2 * rik2 - dot * dot));
                const double inv_denom_ang = 1.0 / denom_ang;

                const double d_ang_d_j0 = Cx - Bx + dot * ((Bx - Ax) * invrij2);
                const double d_ang_d_j1 = Cy - By + dot * ((By - Ay) * invrij2);
                const double d_ang_d_j2 = Cz - Bz + dot * ((Bz - Az) * invrij2);
                const double d_ang_d_k0 = Ax - Bx + dot * ((Bx - Cx) * invrik2);
                const double d_ang_d_k1 = Ay - By + dot * ((By - Cy) * invrik2);
                const double d_ang_d_k2 = Az - Bz + dot * ((Bz - Cz) * invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);
                const double dai0 = d_ang_d_i0 * inv_denom_ang;
                const double dai1 = d_ang_d_i1 * inv_denom_ang;
                const double dai2 = d_ang_d_i2 * inv_denom_ang;
                const double daj0 = d_ang_d_j0 * inv_denom_ang;
                const double daj1 = d_ang_d_j1 * inv_denom_ang;
                const double daj2 = d_ang_d_j2 * inv_denom_ang;
                const double dak0 = d_ang_d_k0 * inv_denom_ang;
                const double dak1 = d_ang_d_k1 * inv_denom_ang;
                const double dak2 = d_ang_d_k2 * inv_denom_ang;

                // Cosine series + Chebyshev derivative (same as A4)
                const double sin_i7 = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                if (nabasis > 0) { angular[0] = 1.0;    d_angular[0] = 0.0; }
                if (nabasis > 1) { angular[1] = cos_i;  d_angular[1] = sin_i7; }
                {
                    double Um2 = 1.0, Um1 = 2.0 * cos_i;
                    for (std::size_t m = 2; m < nabasis; ++m) {
                        angular[m] = 2.0 * cos_i * angular[m - 1] - angular[m - 2];
                        d_angular[m] = static_cast<double>(m) * Um1 * sin_i7;
                        double Um = 2.0 * cos_i * Um1 - Um2;
                        Um2 = Um1; Um1 = Um;
                    }
                }

                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm =
                    (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;

                AtmGrad ag = compute_atm_grad(
                    Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz,
                    invrij, invrik, invrjk, invrij2, invrik2,
                    cos_i, cos_j, cos_k, three_body_decay, three_body_weight, atm
                );

                const double decay_ij = rdecay3[idx2(i, j, natoms)];
                const double decay_ik = rdecay3[idx2(i, k, natoms)];
                const double s_ik =
                    -M_PI * std::sin(M_PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ijd0 = s_ij * (Bx - Ax), d_ijd1 = s_ij * (By - Ay),
                             d_ijd2 = s_ij * (Bz - Az);
                const double d_ikd0 = s_ik * (Bx - Cx), d_ikd1 = s_ik * (By - Cy),
                             d_ikd2 = s_ik * (Bz - Cz);
                const double dec_i0 = d_ijd0 * decay_ik + decay_ij * d_ikd0;
                const double dec_i1 = d_ijd1 * decay_ik + decay_ij * d_ikd1;
                const double dec_i2 = d_ijd2 * decay_ik + decay_ij * d_ikd2;
                const double dec_j0 = -d_ijd0 * decay_ik;
                const double dec_j1 = -d_ijd1 * decay_ik;
                const double dec_j2 = -d_ijd2 * decay_ik;
                const double dec_k0 = -decay_ij * d_ikd0;
                const double dec_k1 = -decay_ij * d_ikd1;
                const double dec_k2 = -decay_ij * d_ikd2;
                const double decay_prod = decay_ij * decay_ik;

                const double BmA0 = (Bx - Ax) * invrij, BmA1 = (By - Ay) * invrij,
                             BmA2 = (Bz - Az) * invrij;
                const double BmC0 = (Bx - Cx) * invrik, BmC1 = (By - Cy) * invrik,
                             BmC2 = (Bz - Cz) * invrik;

                if (elem_j == elem_k) {
                    // Diagonal (B==C): SplitR radial basis
                    const std::size_t block_off =
                        er_block_offset(nelements, elem_j, elem_k, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base = block_off;

                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dp = rij + rik - Rs3[l1];
                        const double phi_p = std::exp(-eta3 * dp * dp);
                        for (std::size_t l2 = 0; l2 < nbasis3_minus; ++l2) {
                            const double phi_m_base =
                                std::exp(-eta3_minus *
                                         std::pow(std::abs(rij - rik) - Rs3_minus[l2], 2));
                            const double radial_val = phi_p * phi_m_base * decay_prod;
                            const double scale_val = radial_val * atm;
                            const std::size_t z0 = base + (l1 * nbasis3_minus + l2) * nabasis;

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                                atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                            SplitRRadGrad srg =
                                splitr_radial_grad(rij, rik, Rs3[l1], Rs3_minus[l2], eta3,
                                                   eta3_minus, phi_p, phi_m_base);
                            const double drad_drij = srg.d_phi_d_rij * decay_prod;
                            const double drad_drik = srg.d_phi_d_rik * decay_prod;

                            const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                            const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                            const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                            const double drad_j0 = drad_drij * (-BmA0);
                            const double drad_j1 = drad_drij * (-BmA1);
                            const double drad_j2 = drad_drij * (-BmA2);
                            const double drad_k0 = drad_drik * (-BmC0);
                            const double drad_k1 = drad_drik * (-BmC1);
                            const double drad_k2 = drad_drik * (-BmC2);

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                                const double ang = angular[aidx];
                                const double dang = d_angular[aidx];
                                const std::size_t feat = z0 + aidx;

                                atom_grad[(feat * natoms + i) * 3 + 0] +=
                                    dang * dai0 * radial_val * atm + ang * drad_i0 * atm +
                                    ang * radial_val * ag.i0 +
                                    ang * radial_val * dec_i0 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 0] +=
                                    dang * daj0 * radial_val * atm + ang * drad_j0 * atm +
                                    ang * radial_val * ag.j0 +
                                    ang * radial_val * dec_j0 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 0] +=
                                    dang * dak0 * radial_val * atm + ang * drad_k0 * atm +
                                    ang * radial_val * ag.k0 +
                                    ang * radial_val * dec_k0 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 1] +=
                                    dang * dai1 * radial_val * atm + ang * drad_i1 * atm +
                                    ang * radial_val * ag.i1 +
                                    ang * radial_val * dec_i1 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 1] +=
                                    dang * daj1 * radial_val * atm + ang * drad_j1 * atm +
                                    ang * radial_val * ag.j1 +
                                    ang * radial_val * dec_j1 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 1] +=
                                    dang * dak1 * radial_val * atm + ang * drad_k1 * atm +
                                    ang * radial_val * ag.k1 +
                                    ang * radial_val * dec_k1 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 2] +=
                                    dang * dai2 * radial_val * atm + ang * drad_i2 * atm +
                                    ang * radial_val * ag.i2 +
                                    ang * radial_val * dec_i2 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 2] +=
                                    dang * daj2 * radial_val * atm + ang * drad_j2 * atm +
                                    ang * radial_val * ag.j2 +
                                    ang * radial_val * dec_j2 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 2] +=
                                    dang * dak2 * radial_val * atm + ang * drad_k2 * atm +
                                    ang * radial_val * ag.k2 +
                                    ang * radial_val * dec_k2 * atm / decay_prod;
                            }
                        }
                    }
                } else {
                    // Off-diagonal (B!=C): ordered product basis, two blocks

                    // --- Block (elem_j, elem_k): outer=r_ij, inner=r_ik ---
                    const std::size_t block_off_jk =
                        er_block_offset(nelements, elem_j, elem_k, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base_jk = block_off_jk;

                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dj = rij - Rs3[l1];
                        const double phi_j = std::exp(-eta3 * dj * dj);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dk = rik - Rs3[l2];
                            const double phi_k = std::exp(-eta3 * dk * dk);
                            const double radial_val = phi_j * phi_k * decay_prod;
                            const double scale_val = radial_val * atm;
                            const std::size_t z0 = base_jk + (l1 * nbasis3 + l2) * nabasis;

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                                atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                            const double drad_drij = phi_j * (-2.0 * eta3 * dj) * phi_k * decay_prod;
                            const double drad_drik = phi_j * phi_k * (-2.0 * eta3 * dk) * decay_prod;

                            const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                            const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                            const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                            const double drad_j0 = drad_drij * (-BmA0);
                            const double drad_j1 = drad_drij * (-BmA1);
                            const double drad_j2 = drad_drij * (-BmA2);
                            const double drad_k0 = drad_drik * (-BmC0);
                            const double drad_k1 = drad_drik * (-BmC1);
                            const double drad_k2 = drad_drik * (-BmC2);

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                                const double ang = angular[aidx];
                                const double dang = d_angular[aidx];
                                const std::size_t feat = z0 + aidx;

                                atom_grad[(feat * natoms + i) * 3 + 0] +=
                                    dang * dai0 * radial_val * atm + ang * drad_i0 * atm +
                                    ang * radial_val * ag.i0 +
                                    ang * radial_val * dec_i0 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 0] +=
                                    dang * daj0 * radial_val * atm + ang * drad_j0 * atm +
                                    ang * radial_val * ag.j0 +
                                    ang * radial_val * dec_j0 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 0] +=
                                    dang * dak0 * radial_val * atm + ang * drad_k0 * atm +
                                    ang * radial_val * ag.k0 +
                                    ang * radial_val * dec_k0 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 1] +=
                                    dang * dai1 * radial_val * atm + ang * drad_i1 * atm +
                                    ang * radial_val * ag.i1 +
                                    ang * radial_val * dec_i1 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 1] +=
                                    dang * daj1 * radial_val * atm + ang * drad_j1 * atm +
                                    ang * radial_val * ag.j1 +
                                    ang * radial_val * dec_j1 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 1] +=
                                    dang * dak1 * radial_val * atm + ang * drad_k1 * atm +
                                    ang * radial_val * ag.k1 +
                                    ang * radial_val * dec_k1 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 2] +=
                                    dang * dai2 * radial_val * atm + ang * drad_i2 * atm +
                                    ang * radial_val * ag.i2 +
                                    ang * radial_val * dec_i2 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 2] +=
                                    dang * daj2 * radial_val * atm + ang * drad_j2 * atm +
                                    ang * radial_val * ag.j2 +
                                    ang * radial_val * dec_j2 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 2] +=
                                    dang * dak2 * radial_val * atm + ang * drad_k2 * atm +
                                    ang * radial_val * ag.k2 +
                                    ang * radial_val * dec_k2 * atm / decay_prod;
                            }
                        }
                    }

                    // --- Block (elem_k, elem_j): outer=r_ik (l1), inner=r_ij (l2) ---
                    const std::size_t block_off_kj =
                        er_block_offset(nelements, elem_k, elem_j, nbasis3, nbasis3_minus, nabasis);
                    const std::size_t base_kj = block_off_kj;

                    for (std::size_t l1 = 0; l1 < nbasis3; ++l1) {
                        const double dk = rik - Rs3[l1];
                        const double phi_k = std::exp(-eta3 * dk * dk);
                        for (std::size_t l2 = 0; l2 < nbasis3; ++l2) {
                            const double dj = rij - Rs3[l2];
                            const double phi_j = std::exp(-eta3 * dj * dj);
                            const double radial_val = phi_k * phi_j * decay_prod;
                            const double scale_val = radial_val * atm;
                            const std::size_t z0 = base_kj + (l1 * nbasis3 + l2) * nabasis;

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx)
                                atom_rep[z0 + aidx] += angular[aidx] * scale_val;

                            const double drad_drij = phi_k * phi_j * (-2.0 * eta3 * dj) * decay_prod;
                            const double drad_drik = phi_k * (-2.0 * eta3 * dk) * phi_j * decay_prod;

                            const double drad_i0 = drad_drij * BmA0 + drad_drik * BmC0;
                            const double drad_i1 = drad_drij * BmA1 + drad_drik * BmC1;
                            const double drad_i2 = drad_drij * BmA2 + drad_drik * BmC2;
                            const double drad_j0 = drad_drij * (-BmA0);
                            const double drad_j1 = drad_drij * (-BmA1);
                            const double drad_j2 = drad_drij * (-BmA2);
                            const double drad_k0 = drad_drik * (-BmC0);
                            const double drad_k1 = drad_drik * (-BmC1);
                            const double drad_k2 = drad_drik * (-BmC2);

                            for (std::size_t aidx = 0; aidx < nabasis; ++aidx) {
                                const double ang = angular[aidx];
                                const double dang = d_angular[aidx];
                                const std::size_t feat = z0 + aidx;

                                atom_grad[(feat * natoms + i) * 3 + 0] +=
                                    dang * dai0 * radial_val * atm + ang * drad_i0 * atm +
                                    ang * radial_val * ag.i0 +
                                    ang * radial_val * dec_i0 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 0] +=
                                    dang * daj0 * radial_val * atm + ang * drad_j0 * atm +
                                    ang * radial_val * ag.j0 +
                                    ang * radial_val * dec_j0 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 0] +=
                                    dang * dak0 * radial_val * atm + ang * drad_k0 * atm +
                                    ang * radial_val * ag.k0 +
                                    ang * radial_val * dec_k0 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 1] +=
                                    dang * dai1 * radial_val * atm + ang * drad_i1 * atm +
                                    ang * radial_val * ag.i1 +
                                    ang * radial_val * dec_i1 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 1] +=
                                    dang * daj1 * radial_val * atm + ang * drad_j1 * atm +
                                    ang * radial_val * ag.j1 +
                                    ang * radial_val * dec_j1 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 1] +=
                                    dang * dak1 * radial_val * atm + ang * drad_k1 * atm +
                                    ang * radial_val * ag.k1 +
                                    ang * radial_val * dec_k1 * atm / decay_prod;

                                atom_grad[(feat * natoms + i) * 3 + 2] +=
                                    dang * dai2 * radial_val * atm + ang * drad_i2 * atm +
                                    ang * radial_val * ag.i2 +
                                    ang * radial_val * dec_i2 * atm / decay_prod;
                                atom_grad[(feat * natoms + j) * 3 + 2] +=
                                    dang * daj2 * radial_val * atm + ang * drad_j2 * atm +
                                    ang * radial_val * ag.j2 +
                                    ang * radial_val * dec_j2 * atm / decay_prod;
                                atom_grad[(feat * natoms + k) * 3 + 2] +=
                                    dang * dak2 * radial_val * atm + ang * drad_k2 * atm +
                                    ang * radial_val * ag.k2 +
                                    ang * radial_val * dec_k2 * atm / decay_prod;
                            }
                        }
                    }
                }  // end if elem_j == elem_k
            }
        }

        for (std::size_t off = 0; off < three_block_size; ++off)
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        for (std::size_t off = 0; off < three_block_size; ++off)
            for (std::size_t a = 0; a < natoms; ++a)
                for (std::size_t t = 0; t < 3; ++t) {
                    const double v = atom_grad[(off * natoms + a) * 3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
    }
}

// Dispatch three-body gradient
static void three_body_grad(
    ThreeBodyType type, std::size_t natoms, std::size_t nelements, std::size_t nbasis2,
    std::size_t nbasis3, std::size_t nabasis, const std::vector<double> &coords,
    const std::vector<double> &D, const std::vector<double> &D2, const std::vector<double> &invD,
    const std::vector<double> &invD2, const std::vector<double> &rdecay3,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta3,
    double eta3_minus, double zeta, double acut, double three_body_decay,
    double three_body_weight, const std::vector<int> &elem_of_atom, std::size_t rep_size,
    std::vector<double> &rep, std::vector<double> &grad
) {
    const std::size_t nbasis3_minus = Rs3_minus.size();
    switch (type) {
        case ThreeBodyType::OddFourier_Rbar:
            three_body_a1_grad(
                natoms, nelements, nbasis2, nbasis3, nabasis, coords, D, D2, invD, invD2, rdecay3,
                Rs3, eta3, zeta, acut, three_body_decay, three_body_weight, elem_of_atom, rep_size,
                rep, grad
            );
            break;
        case ThreeBodyType::CosineSeries_Rbar:
            three_body_a2_grad(
                natoms, nelements, nbasis2, nbasis3, nabasis, coords, D, D2, invD, invD2, rdecay3,
                Rs3, eta3, acut, three_body_decay, three_body_weight, elem_of_atom, rep_size,
                rep, grad
            );
            break;
        case ThreeBodyType::OddFourier_SplitR:
            three_body_a3_grad(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, D2, invD,
                invD2, rdecay3, Rs3, Rs3_minus, eta3, eta3_minus, zeta, acut, three_body_decay,
                three_body_weight, elem_of_atom, rep_size, rep, grad
            );
            break;
        case ThreeBodyType::CosineSeries_SplitR:
            three_body_a4_grad(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, D2, invD,
                invD2, rdecay3, Rs3, Rs3_minus, eta3, eta3_minus, acut, three_body_decay,
                three_body_weight, elem_of_atom, rep_size, rep, grad
            );
            break;
        case ThreeBodyType::CosineSeries_SplitR_NoATM:
            three_body_a5_grad(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, D2, invD,
                invD2, rdecay3, Rs3, Rs3_minus, eta3, eta3_minus, acut, three_body_weight,
                elem_of_atom, rep_size, rep, grad
            );
            break;
        case ThreeBodyType::OddFourier_ElementResolved:
            three_body_a6_grad(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, D2, invD,
                invD2, rdecay3, Rs3, Rs3_minus, eta3, eta3_minus, zeta, acut, three_body_decay,
                three_body_weight, elem_of_atom, rep_size, rep, grad
            );
            break;
        case ThreeBodyType::CosineSeries_ElementResolved:
            three_body_a7_grad(
                natoms, nelements, nbasis2, nbasis3, nbasis3_minus, nabasis, coords, D, D2, invD,
                invD2, rdecay3, Rs3, Rs3_minus, eta3, eta3_minus, acut, three_body_decay,
                three_body_weight, elem_of_atom, rep_size, rep, grad
            );
            break;
    }
}

// ==================== Public API ====================

void generate(
    const std::vector<double> &coords, const std::vector<int> &nuclear_z,
    const std::vector<int> &elements, const std::vector<double> &Rs2,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta2,
    double eta3, double eta3_minus, double zeta, double rcut, double acut,
    double two_body_decay, double three_body_decay, double three_body_weight,
    TwoBodyType two_body_type, ThreeBodyType three_body_type, int nabasis,
    std::vector<double> &rep
) {
    const std::size_t natoms = nuclear_z.size();
    if (coords.size() != natoms * 3) throw std::invalid_argument("coords size must be natoms*3");

    const std::size_t nelements = elements.size();
    const std::size_t nbasis2 = Rs2.size();
    const std::size_t nbasis3 = Rs3.size();
    const std::size_t nab = static_cast<std::size_t>(nabasis);

    const std::size_t rep_size =
        compute_rep_size(nelements, nbasis2, nbasis3, nab, three_body_type, Rs3_minus.size());
    rep.assign(natoms * rep_size, 0.0);

    const std::vector<int> elem_of_atom = build_elem_map(nuclear_z, elements, natoms);

    // Distances and decays
    const std::vector<double> D = pairwise_distances(coords, natoms);
    std::vector<double> rdecay2, rdecay3;
    decay_matrix(D, (rcut > 0 ? 1.0 / rcut : 0.0), natoms, rdecay2);
    decay_matrix(D, (acut > 0 ? 1.0 / acut : 0.0), natoms, rdecay3);

    // Two-body
    two_body_forward(
        two_body_type, natoms, nbasis2, nelements, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
        elem_of_atom, rep_size, rep
    );

    // Three-body
    if (three_body_weight != 0.0) {
        three_body_forward(
            three_body_type, natoms, nelements, nbasis2, nbasis3, nab, coords, D, rdecay3, Rs3,
            Rs3_minus, eta3, eta3_minus, zeta, acut, three_body_decay, three_body_weight,
            elem_of_atom, rep_size, rep
        );
    }
}

void generate_and_gradients(
    const std::vector<double> &coords, const std::vector<int> &nuclear_z,
    const std::vector<int> &elements, const std::vector<double> &Rs2,
    const std::vector<double> &Rs3, const std::vector<double> &Rs3_minus, double eta2,
    double eta3, double eta3_minus, double zeta, double rcut, double acut,
    double two_body_decay, double three_body_decay, double three_body_weight,
    TwoBodyType two_body_type, ThreeBodyType three_body_type, int nabasis,
    std::vector<double> &rep, std::vector<double> &grad
) {
    const std::size_t natoms = nuclear_z.size();
    if (coords.size() != natoms * 3) throw std::invalid_argument("coords size must be natoms*3");

    const std::size_t nelements = elements.size();
    const std::size_t nbasis2 = Rs2.size();
    const std::size_t nbasis3 = Rs3.size();
    const std::size_t nab = static_cast<std::size_t>(nabasis);

    const std::size_t rep_size =
        compute_rep_size(nelements, nbasis2, nbasis3, nab, three_body_type, Rs3_minus.size());
    rep.assign(natoms * rep_size, 0.0);
    grad.assign(natoms * rep_size * natoms * 3, 0.0);

    const std::vector<int> elem_of_atom = build_elem_map(nuclear_z, elements, natoms);

    // Distances and derived quantities
    const std::vector<double> D = pairwise_distances(coords, natoms);
    std::vector<double> D2(natoms * natoms, 0.0), invD(natoms * natoms, 0.0),
        invD2(natoms * natoms, 0.0);
    for (std::size_t i = 0; i < natoms; ++i) {
        for (std::size_t j = i + 1; j < natoms; ++j) {
            double rij = D[idx2(i, j, natoms)];
            double rij2 = std::max(rij * rij, kf::EPS);
            double invr = 1.0 / rij;
            double invr2 = 1.0 / rij2;
            D2[idx2(i, j, natoms)] = D2[idx2(j, i, natoms)] = rij2;
            invD[idx2(i, j, natoms)] = invD[idx2(j, i, natoms)] = invr;
            invD2[idx2(i, j, natoms)] = invD2[idx2(j, i, natoms)] = invr2;
        }
    }

    std::vector<double> rdecay2, rdecay3;
    if (rcut > 0)
        decay_matrix(D, 1.0 / rcut, natoms, rdecay2);
    else
        rdecay2.assign(natoms * natoms, 1.0);
    if (acut > 0)
        decay_matrix(D, 1.0 / acut, natoms, rdecay3);
    else
        rdecay3.assign(natoms * natoms, 1.0);

    // Two-body (values + gradients)
    two_body_grad(
        two_body_type, natoms, nbasis2, coords, D, rdecay2, Rs2, eta2, two_body_decay, rcut,
        elem_of_atom, rep_size, rep, grad
    );

    // Three-body (values + gradients)
    if (three_body_weight != 0.0) {
        three_body_grad(
            three_body_type, natoms, nelements, nbasis2, nbasis3, nab, coords, D, D2, invD, invD2,
            rdecay3, Rs3, Rs3_minus, eta3, eta3_minus, zeta, acut, three_body_decay,
            three_body_weight, elem_of_atom, rep_size, rep, grad
        );
    }
}

}  // namespace fchl19v2
}  // namespace kf
