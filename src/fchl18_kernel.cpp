// Own header
#include "fchl18_kernel.hpp"

#include "fchl18_repr.hpp"

// C++ standard library
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// OpenMP
#include <omp.h>

namespace kf {
namespace fchl18 {

// =============================================================================
// Helper math functions (ported from ffchl_module.f90)
// =============================================================================

// Fast integer power: x^n for small non-negative integer n.
static inline double ipow(double x, int n) {
    double r = 1.0;
    for (; n > 0; --n)
        r *= x;
    return r;
}

// Generalised power: uses ipow when exponent is a small integer, std::pow otherwise.
static inline double fast_pow(double x, double p) {
    const int ip = static_cast<int>(p);
    if (static_cast<double>(ip) == p && ip >= 0 && ip <= 16) return ipow(x, ip);
    return std::pow(x, p);
}

// Cosine-based cutoff damping function.
// Returns 1 below rl=cut_start*cut_distance, 0 above cut_distance,
// and a smooth quintic interpolant in between.
static inline double cut_function(double r, double cut_start, double cut_distance) {
    const double ru = cut_distance;
    const double rl = cut_start * cut_distance;

    if (r >= ru) return 0.0;
    if (r <= rl) return 1.0;

    const double x = (ru - r) / (ru - rl);
    return 10.0 * x * x * x - 15.0 * x * x * x * x + 6.0 * x * x * x * x * x;
}

// Angular normalisation constant for three-body term.
// Integrates exp(-(t_width*n)^2) * 2*(1-cos(n*pi)) over n in [-limit, limit].
static double get_angular_norm2(double t_width) {
    const double pi = 4.0 * std::atan(1.0);
    const int limit = 10000;
    double ang_norm2 = 0.0;

    for (int n = -limit; n <= limit; ++n) {
        const double tn = t_width * n;
        ang_norm2 += std::exp(-(tn * tn)) * (2.0 - 2.0 * std::cos(n * pi));
    }
    return std::sqrt(ang_norm2 * pi) * 2.0;
}

// clamp to [-1, 1]
static inline double clamp11(double x) {
    return x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x);
}

// =============================================================================
// Per-atom precomputed quantities
// =============================================================================

// Compute two-body weights ksi[k] = cut(r_k) / r_k^power for each neighbour k.
// ksi[0] = 0 (self-atom, distance 0).
static void compute_ksi(
    const double *atom_chan,  // (5, max_size) slice for one atom
    int max_size, int n_neigh, double power, double cut_start, double cut_distance,
    std::vector<double> &ksi  // size n_neigh, output
) {
    ksi.assign(n_neigh, 0.0);
    for (int k = 1; k < n_neigh; ++k) {  // k=0 is self, skip
        const double r = atom_chan[k];   // channel 0
        if (r < 1e-14) continue;
        ksi[k] = cut_function(r, cut_start, cut_distance) / fast_pow(r, power);
    }
}

// Compute three-body Fourier terms for one atom.
//
// Optimised: precomputes unit vectors and distances for all neighbours once,
// so the inner triplet loop uses only dot products (no per-triplet sqrt).
// The angle at the centre (theta) is computed once per triplet and shared
// between the ksi3 weight and the Fourier cos/sin terms (no redundant acos).
//
// use_atm: if false, replaces the Axilrod-Teller-Muto factor
//          (1 + 3*cos_i*cos_j*cos_k) with 1.0 — skips cos_j/cos_k computation.
//
// Output: cosp/sinp with layout (pmax, order, max_size) flat.
static void compute_threebody_fourier(
    const double *atom_chan,  // (5, max_size) slice for one atom
    int max_size, int n_neigh, double three_body_power, double cut_start, double cut_distance,
    int order, int pmax,
    const std::vector<int> &z_to_idx,  // Z -> compact index (size 256)
    bool use_atm,
    std::vector<double> &cosp,  // (pmax, order, max_size) flat
    std::vector<double> &sinp   // (pmax, order, max_size) flat
) {
    const std::size_t sz = static_cast<std::size_t>(pmax) * order * max_size;
    cosp.assign(sz, 0.0);
    sinp.assign(sz, 0.0);

    auto idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size +
               static_cast<std::size_t>(m) * max_size + neigh;
    };

    // Channel pointers
    const double *dist_chan = atom_chan;          // channel 0: distances
    const double *z_chan = atom_chan + max_size;  // channel 1: nuclear charges
    const double *xc = atom_chan + 2 * max_size;  // channel 2: dx
    const double *yc = atom_chan + 3 * max_size;  // channel 3: dy
    const double *zc = atom_chan + 4 * max_size;  // channel 4: dz

    // --- Precompute unit vectors and cutoff values for each neighbour ---
    // unit_[xyz][k]: unit vector from centre to neighbour k (already displacement, dist stored)
    // cut[k]: cut_function(dist_k)
    std::vector<double> ux(n_neigh), uy(n_neigh), uz(n_neigh), cut_k(n_neigh);
    for (int k = 1; k < n_neigh; ++k) {
        const double r = dist_chan[k];
        cut_k[k] = cut_function(r, cut_start, cut_distance);
        const double inv_r = (r > 1e-14) ? 1.0 / r : 0.0;
        ux[k] = xc[k] * inv_r;
        uy[k] = yc[k] * inv_r;
        uz[k] = zc[k] * inv_r;
    }

    for (int j = 1; j < n_neigh; ++j) {
        const double dj = dist_chan[j];  // dist(centre, j)
        if (dj < 1e-14) continue;
        const double cutj = cut_k[j];
        if (cutj == 0.0) continue;

        for (int k = j + 1; k < n_neigh; ++k) {
            const double dk = dist_chan[k];  // dist(centre, k)
            if (dk < 1e-14) continue;
            const double cutk = cut_k[k];
            if (cutk == 0.0) continue;

            // Distance between neighbours j and k
            const double dxjk = xc[j] - xc[k];
            const double dyjk = yc[j] - yc[k];
            const double dzjk = zc[j] - zc[k];
            const double di2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
            const double di = std::sqrt(di2);
            if (di < 1e-14) continue;

            const double cut_jk = cut_function(di, cut_start, cut_distance);
            if (cut_jk == 0.0) continue;

            // cos_i = angle at centre between j and k: dot(unit_j, unit_k)
            const double cos_i = clamp11(ux[j] * ux[k] + uy[j] * uy[k] + uz[j] * uz[k]);

            double atm;
            if (use_atm) {
                // cos_j = angle at j between k and centre: dot(unit_k_from_j, unit_centre_from_j)
                const double inv_di = 1.0 / di;
                const double ukj_x = -dxjk * inv_di, ukj_y = -dyjk * inv_di, ukj_z = -dzjk * inv_di;
                const double cos_j =
                    clamp11(ukj_x * (-ux[j]) + ukj_y * (-uy[j]) + ukj_z * (-uz[j]));

                // cos_k = angle at k between j and centre: dot(unit_j_from_k, unit_centre_from_k)
                const double ujk_x = dxjk * inv_di, ujk_y = dyjk * inv_di, ujk_z = dzjk * inv_di;
                const double cos_k =
                    clamp11(ujk_x * (-ux[k]) + ujk_y * (-uy[k]) + ujk_z * (-uz[k]));

                atm = 1.0 + 3.0 * cos_i * cos_j * cos_k;
            } else {
                atm = 1.0;
            }
            if (atm == 0.0) continue;

            const double dijk = di * dj * dk;
            const double denom = fast_pow(dijk, three_body_power);
            const double cut = cutj * cutk * cut_jk;
            const double ksi3 = cut * atm / denom;
            if (ksi3 == 0.0) continue;

            // theta = angle at centre (same as cos_i, just one acos)
            const double theta = std::acos(cos_i);

            const int zk = static_cast<int>(z_chan[k]);
            const int zj = static_cast<int>(z_chan[j]);
            if (zk <= 0 || zk >= 256 || zj <= 0 || zj >= 256) continue;
            const int pj = z_to_idx[zk];  // compact index for Z of k-th neighbour
            const int pk = z_to_idx[zj];  // compact index for Z of j-th neighbour
            if (pj < 0 || pk < 0) continue;

            // Only odd Fourier indices n=m+1 contribute: cos(nθ)-cos(n(θ+π)) = 2 cos(nθ)
            // for odd n and 0 for even n. Step by 2 (m = 0,2,4,… → n = 1,3,5,…).
            for (int m = 0; m < order; m += 2) {
                const double mf = static_cast<double>(m + 1);
                const double mth = mf * theta;
                const double c = std::cos(mth);
                const double s = std::sin(mth);
                const double cos_m = 2.0 * c * ksi3;
                const double sin_m = 2.0 * s * ksi3;

                cosp[idx(pj, m, j)] += cos_m;
                sinp[idx(pj, m, j)] += sin_m;
                cosp[idx(pk, m, k)] += cos_m;
                sinp[idx(pk, m, k)] += sin_m;
            }
        }
    }
}

// =============================================================================
// FCHL18 scalar product between two atoms (no-alchemy version)
//
// Ported from scalar_noalchemy in ffchl_module.f90.
// Only neighbours with matching nuclear charge contribute.
// =============================================================================
// s_prefactor[m] = g1 * exp(-(t_width*(m+1))^2/2), precomputed once per kernel call.
static double scalar_noalchemy(
    // Atom i data: (5, max_size_1) channel-major
    const double *x1_chan, int max_size1, int n1, const double *ksi1,
    const std::vector<double> &cos1,  // (pmax, order, max_size1)
    const std::vector<double> &sin1,
    // Atom j data: (5, max_size_2) channel-major
    const double *x2_chan, int max_size2, int n2, const double *ksi2,
    const std::vector<double> &cos2,  // (pmax, order, max_size2)
    const std::vector<double> &sin2,
    // Kernel hyperparameters
    double d_width, int order, int pmax,
    const double *s_prefactor,  // (order) precomputed Fourier prefactors
    double distance_scale, double angular_scale
) {
    // Early exit: central atoms must have the same nuclear charge (matches Fortran
    // scalar_noalchemy)
    const int Z1 = static_cast<int>(x1_chan[1 * max_size1 + 0]);  // channel 1, neighbour 0 = self
    const int Z2 = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    if (Z1 != Z2) return 0.0;

    const double inv_width = -1.0 / (4.0 * d_width * d_width);
    const double maxgausdist2 = (8.0 * d_width) * (8.0 * d_width);

    // Scalar product starts at 1 (self-contribution)
    double aadist = 1.0;

    // Sum over neighbour pairs (i from atom1, j from atom2) with same Z
    auto cos1_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size1 +
               static_cast<std::size_t>(m) * max_size1 + neigh;
    };
    auto cos2_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size2 +
               static_cast<std::size_t>(m) * max_size2 + neigh;
    };

    for (int i = 1; i < n1; ++i) {
        const int Zi = static_cast<int>(x1_chan[1 * max_size1 + i]);
        const double ri = x1_chan[i];  // channel 0, neighbour i

        for (int j = 1; j < n2; ++j) {
            const int Zj = static_cast<int>(x2_chan[1 * max_size2 + j]);
            if (Zi != Zj) continue;

            const double rj = x2_chan[j];  // channel 0
            const double r2 = (rj - ri) * (rj - ri);
            if (r2 >= maxgausdist2) continue;

            const double d = std::exp(r2 * inv_width);

            // Angular contribution: only odd Fourier n=m+1 are nonzero (see three-body).
            double angular = 0.0;
            for (int m = 0; m < order; m += 2) {
                double ang_m = 0.0;
                // We sum over element types p that exist in both atoms' neighbourhoods.
                // In the no-alchemy path elements must match (same p used for both sides).
                for (int p = 0; p < pmax; ++p) {
                    ang_m += cos1[cos1_idx(p, m, i)] * cos2[cos2_idx(p, m, j)] +
                             sin1[cos1_idx(p, m, i)] * sin2[cos2_idx(p, m, j)];
                }
                angular += ang_m * s_prefactor[m];
            }

            aadist += d * (ksi1[i] * ksi2[j] * distance_scale + angular * angular_scale);
        }
    }

    return aadist;
}

// =============================================================================
// Internal struct: per-atom precomputed data
// =============================================================================
struct AtomData {
    int n_neigh;
    std::vector<double> ksi;
    std::vector<double> cosp;
    std::vector<double> sinp;
};

// Precompute all per-atom data for an entire set of molecules.
// Returns a flat vector indexed [mol * max_size + atom].
// Parallelised over the flat (mol, atom) space for fine-grained load balancing.
static std::vector<AtomData> precompute_atom_data(
    const std::vector<double> &x,  // (nm, max_size, 5, max_size) row-major
    const std::vector<int> &n,     // (nm) atom counts
    const std::vector<int> &nn,    // (nm * max_size) neighbour counts
    int nm, int max_size, double two_body_power, double cut_start, double cut_distance,
    double three_body_power, int order, int pmax,
    const std::vector<int> &z_to_idx,  // compact element map (size 256)
    bool use_atm
) {
    std::vector<AtomData> data(static_cast<std::size_t>(nm) * max_size);

    // Build a flat list of (mol, atom) pairs for all real atoms so that OpenMP
    // gets one task per atom instead of one task per molecule — much better load
    // balance when molecules have different sizes.
    std::vector<std::pair<int, int>> atom_list;
    atom_list.reserve(static_cast<std::size_t>(nm) * max_size);
    for (int a = 0; a < nm; ++a)
        for (int i = 0; i < n[a]; ++i)
            atom_list.emplace_back(a, i);
    const int total_atoms = static_cast<int>(atom_list.size());

    const std::size_t mol_stride = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t atom_stride = static_cast<std::size_t>(5) * max_size;

#pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < total_atoms; ++t) {
        const int a = atom_list[t].first;
        const int i = atom_list[t].second;

        AtomData &ad = data[static_cast<std::size_t>(a) * max_size + i];
        const int n_neigh = nn[static_cast<std::size_t>(a) * max_size + i];
        ad.n_neigh = n_neigh;

        const double *atom_chan = x.data() + static_cast<std::size_t>(a) * mol_stride +
                                  static_cast<std::size_t>(i) * atom_stride;

        compute_ksi(atom_chan, max_size, n_neigh, two_body_power, cut_start, cut_distance, ad.ksi);

        compute_threebody_fourier(
            atom_chan,
            max_size,
            n_neigh,
            three_body_power,
            cut_start,
            cut_distance,
            order,
            pmax,
            z_to_idx,
            use_atm,
            ad.cosp,
            ad.sinp
        );
    }
    return data;
}

// =============================================================================
// Compute self-scalar for every real atom in a set.
// self_scalar[a * max_size + i] = scalar(atom_i from mol_a, atom_i from mol_a)
// =============================================================================
static std::vector<double> compute_self_scalars(
    const std::vector<double> &x,  // (nm, max_size, 5, max_size)
    const std::vector<int> &n,     // (nm)
    const std::vector<AtomData> &ad, int nm, int max_size, int order, int pmax, double t_width,
    double d_width, double ang_norm2, double distance_scale, double angular_scale
) {
    const std::size_t mol_stride = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t atom_stride = static_cast<std::size_t>(5) * max_size;

    // Precompute Fourier prefactors once (not per scalar_noalchemy call)
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    std::vector<double> s_prefactor(order);
    for (int m = 0; m < order; ++m) {
        const double mf = static_cast<double>(m + 1);
        s_prefactor[m] = g1 * std::exp(-(t_width * mf) * (t_width * mf) / 2.0);
    }

    // Build flat atom list for fine-grained parallelism (same trick as precompute_atom_data)
    std::vector<std::pair<int, int>> atom_list;
    atom_list.reserve(static_cast<std::size_t>(nm) * max_size);
    for (int a = 0; a < nm; ++a)
        for (int i = 0; i < n[a]; ++i)
            atom_list.emplace_back(a, i);
    const int total_atoms = static_cast<int>(atom_list.size());

    std::vector<double> ss(static_cast<std::size_t>(nm) * max_size, 0.0);

#pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < total_atoms; ++t) {
        const int a = atom_list[t].first;
        const int i = atom_list[t].second;
        const AtomData &adi = ad[static_cast<std::size_t>(a) * max_size + i];
        const double *atom_chan = x.data() + static_cast<std::size_t>(a) * mol_stride +
                                  static_cast<std::size_t>(i) * atom_stride;

        ss[static_cast<std::size_t>(a) * max_size + i] = scalar_noalchemy(
            atom_chan,
            max_size,
            adi.n_neigh,
            adi.ksi.data(),
            adi.cosp,
            adi.sinp,
            atom_chan,
            max_size,
            adi.n_neigh,
            adi.ksi.data(),
            adi.cosp,
            adi.sinp,
            d_width,
            order,
            pmax,
            s_prefactor.data(),
            distance_scale,
            angular_scale
        );
    }
    return ss;
}

// =============================================================================
// Build a compact element map: Z -> compact index (0-based).
// Returns the number of distinct elements (new pmax).
// Also fills z_to_idx[Z] = compact_idx for Z in [0, 255]; -1 if not present.
// Only scans real neighbour slots (up to nn[a,i] per atom) to avoid UB from
// casting the 1e100 padding value in channel 1 to int.
// =============================================================================
static int build_element_map(
    const std::vector<double> &x1, const std::vector<int> &n1, const std::vector<int> &nn1, int nm1,
    int max_size1, const std::vector<double> &x2, const std::vector<int> &n2,
    const std::vector<int> &nn2, int nm2, int max_size2,
    std::vector<int> &z_to_idx  // size 256, output
) {
    z_to_idx.assign(256, -1);
    std::vector<int> present;  // Z values seen so far

    auto scan = [&](const std::vector<double> &x,
                    const std::vector<int> &nv,
                    const std::vector<int> &nn,
                    int nm,
                    int ms) {
        const std::size_t mol_s = static_cast<std::size_t>(ms) * 5 * ms;
        const std::size_t atom_s = static_cast<std::size_t>(5) * ms;
        for (int a = 0; a < nm; ++a) {
            for (int i = 0; i < nv[a]; ++i) {
                const double *chan = x.data() + a * mol_s + i * atom_s;
                const int n_neigh = nn[static_cast<std::size_t>(a) * ms + i];
                for (int k = 0; k < n_neigh; ++k) {
                    const int z = static_cast<int>(chan[ms + k]);  // channel 1
                    if (z > 0 && z < 256 && z_to_idx[z] < 0) {
                        z_to_idx[z] = static_cast<int>(present.size());
                        present.push_back(z);
                    }
                }
            }
        }
    };
    scan(x1, n1, nn1, nm1, max_size1);
    scan(x2, n2, nn2, nm2, max_size2);
    return static_cast<int>(present.size());  // compact pmax
}

// =============================================================================
// Public API: kernel_gaussian (asymmetric)
// =============================================================================
void kernel_gaussian(
    const std::vector<double> &x1, const std::vector<double> &x2, const std::vector<int> &n1,
    const std::vector<int> &n2, const std::vector<int> &nn1, const std::vector<int> &nn2, int nm1,
    int nm2, int max_size1, int max_size2, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm, double *kernel_out
) {
    if (!kernel_out) throw std::invalid_argument("kernel_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

    // Apply the Fortran convention scaling correction
    const double true_distance_scale = two_body_scaling / 16.0;
    const double true_angular_scale = three_body_scaling / std::sqrt(8.0);

    const double ang_norm2 = get_angular_norm2(three_body_width);

    // Build compact element map: only distinct Z values seen in real neighbour slots.
    // This reduces pmax from max(Z) to the number of distinct elements (e.g. 4 for H/C/N/O).
    std::vector<int> z_to_idx;
    const int pmax =
        build_element_map(x1, n1, nn1, nm1, max_size1, x2, n2, nn2, nm2, max_size2, z_to_idx);

    if (pmax == 0) {
        std::memset(kernel_out, 0, sizeof(double) * nm1 * nm2);
        return;
    }

    // Precompute per-atom data
    auto ad1 = precompute_atom_data(
        x1,
        n1,
        nn1,
        nm1,
        max_size1,
        two_body_power,
        cut_start,
        cut_distance,
        three_body_power,
        fourier_order,
        pmax,
        z_to_idx,
        use_atm
    );
    auto ad2 = precompute_atom_data(
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
        two_body_power,
        cut_start,
        cut_distance,
        three_body_power,
        fourier_order,
        pmax,
        z_to_idx,
        use_atm
    );

    // Self-scalars
    auto ss1 = compute_self_scalars(
        x1,
        n1,
        ad1,
        nm1,
        max_size1,
        fourier_order,
        pmax,
        three_body_width,
        two_body_width,
        ang_norm2,
        true_distance_scale,
        true_angular_scale
    );
    auto ss2 = compute_self_scalars(
        x2,
        n2,
        ad2,
        nm2,
        max_size2,
        fourier_order,
        pmax,
        three_body_width,
        two_body_width,
        ang_norm2,
        true_distance_scale,
        true_angular_scale
    );

    // Zero output
    std::memset(kernel_out, 0, sizeof(double) * nm1 * nm2);

    // qmllib convention: parameters = -0.5/sigma^2 (see fchl_kernel_functions.py)
    const double inv_sigma2 = -0.5 / (sigma * sigma);

    // Precompute Fourier prefactors once for the entire kernel call
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    std::vector<double> s_prefactor(fourier_order);
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        s_prefactor[m] = g1 * std::exp(-(three_body_width * mf) * (three_body_width * mf) / 2.0);
    }

    const std::size_t mol1 = static_cast<std::size_t>(max_size1) * 5 * max_size1;
    const std::size_t at1 = static_cast<std::size_t>(5) * max_size1;
    const std::size_t mol2 = static_cast<std::size_t>(max_size2) * 5 * max_size2;
    const std::size_t at2 = static_cast<std::size_t>(5) * max_size2;

    // Main loop: O(nm1 * nm2 * na1 * na2)
#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < nm1; ++a) {
        const int na = n1[a];
        for (int b = 0; b < nm2; ++b) {
            const int nb = n2[b];
            double kab = 0.0;

            for (int i = 0; i < na; ++i) {
                const AtomData &adi = ad1[static_cast<std::size_t>(a) * max_size1 + i];
                const double *x1_chan = x1.data() + a * mol1 + i * at1;

                const double sii = ss1[static_cast<std::size_t>(a) * max_size1 + i];

                for (int j = 0; j < nb; ++j) {
                    const AtomData &adj = ad2[static_cast<std::size_t>(b) * max_size2 + j];
                    const double *x2_chan = x2.data() + b * mol2 + j * at2;

                    const double sjj = ss2[static_cast<std::size_t>(b) * max_size2 + j];

                    const double s12 = scalar_noalchemy(
                        x1_chan,
                        max_size1,
                        adi.n_neigh,
                        adi.ksi.data(),
                        adi.cosp,
                        adi.sinp,
                        x2_chan,
                        max_size2,
                        adj.n_neigh,
                        adj.ksi.data(),
                        adj.cosp,
                        adj.sinp,
                        two_body_width,
                        fourier_order,
                        pmax,
                        s_prefactor.data(),
                        true_distance_scale,
                        true_angular_scale
                    );

                    kab += std::exp((sii + sjj - 2.0 * s12) * inv_sigma2);
                }
            }

            kernel_out[static_cast<std::size_t>(a) * nm2 + b] = kab;
        }
    }
}

// =============================================================================
// Public API: kernel_gaussian_symm
// =============================================================================
void kernel_gaussian_symm(
    const std::vector<double> &x, const std::vector<int> &n, const std::vector<int> &nn, int nm,
    int max_size, double sigma, double two_body_scaling, double two_body_width,
    double two_body_power, double three_body_scaling, double three_body_width,
    double three_body_power, double cut_start, double cut_distance, int fourier_order, bool use_atm,
    double *kernel_out
) {
    if (!kernel_out) throw std::invalid_argument("kernel_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

    const double true_distance_scale = two_body_scaling / 16.0;
    const double true_angular_scale = three_body_scaling / std::sqrt(8.0);

    const double ang_norm2 = get_angular_norm2(three_body_width);

    std::vector<int> z_to_idx;
    const int pmax = build_element_map(x, n, nn, nm, max_size, x, n, nn, nm, max_size, z_to_idx);

    if (pmax == 0) {
        std::memset(kernel_out, 0, sizeof(double) * nm * nm);
        return;
    }

    auto ad = precompute_atom_data(
        x,
        n,
        nn,
        nm,
        max_size,
        two_body_power,
        cut_start,
        cut_distance,
        three_body_power,
        fourier_order,
        pmax,
        z_to_idx,
        use_atm
    );

    auto ss = compute_self_scalars(
        x,
        n,
        ad,
        nm,
        max_size,
        fourier_order,
        pmax,
        three_body_width,
        two_body_width,
        ang_norm2,
        true_distance_scale,
        true_angular_scale
    );

    // Zero output
    std::memset(kernel_out, 0, sizeof(double) * nm * nm);

    // qmllib convention: parameters = -0.5/sigma^2 (see fchl_kernel_functions.py)
    const double inv_sigma2 = -0.5 / (sigma * sigma);

    // Precompute Fourier prefactors once for the entire kernel call
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    std::vector<double> s_prefactor(fourier_order);
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        s_prefactor[m] = g1 * std::exp(-(three_body_width * mf) * (three_body_width * mf) / 2.0);
    }

    const std::size_t mol_s = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t at_s = static_cast<std::size_t>(5) * max_size;

    // Build flat list of all upper-triangle (a, b) pairs for load-balanced parallelism.
    // Previously parallelised over 'a' only: the last thread gets work proportional to
    // just 1 pair while the first gets nm pairs — terrible balance for large nm.
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<std::size_t>(nm) * (nm + 1) / 2);
    for (int a = 0; a < nm; ++a)
        for (int b = a; b < nm; ++b)
            pairs.emplace_back(a, b);
    const int npairs = static_cast<int>(pairs.size());

#pragma omp parallel for schedule(dynamic, 4)
    for (int p = 0; p < npairs; ++p) {
        const int a = pairs[p].first;
        const int b = pairs[p].second;
        const int na = n[a];
        const int nb = n[b];
        double kab = 0.0;

        for (int i = 0; i < na; ++i) {
            const AtomData &adi = ad[static_cast<std::size_t>(a) * max_size + i];
            const double *xa_chan = x.data() + a * mol_s + i * at_s;

            const double sii = ss[static_cast<std::size_t>(a) * max_size + i];

            for (int j = 0; j < nb; ++j) {
                const AtomData &adj = ad[static_cast<std::size_t>(b) * max_size + j];
                const double *xb_chan = x.data() + b * mol_s + j * at_s;

                const double sjj = ss[static_cast<std::size_t>(b) * max_size + j];

                const double s12 = scalar_noalchemy(
                    xa_chan,
                    max_size,
                    adi.n_neigh,
                    adi.ksi.data(),
                    adi.cosp,
                    adi.sinp,
                    xb_chan,
                    max_size,
                    adj.n_neigh,
                    adj.ksi.data(),
                    adj.cosp,
                    adj.sinp,
                    two_body_width,
                    fourier_order,
                    pmax,
                    s_prefactor.data(),
                    true_distance_scale,
                    true_angular_scale
                );

                kab += std::exp((sii + sjj - 2.0 * s12) * inv_sigma2);
            }
        }

        kernel_out[static_cast<std::size_t>(a) * nm + b] = kab;
        kernel_out[static_cast<std::size_t>(b) * nm + a] = kab;
    }
}

// =============================================================================
// Gradient of the FCHL18 kernel w.r.t. Cartesian coordinates of molecule A.
//
// kernel_gaussian_gradient computes:
//
//   G[alpha, mu, b] = dK[A,b] / dR_A[alpha, mu]
//
// where R_A[alpha, mu] is the mu-th coordinate (0=x,1=y,2=z) of atom alpha
// in molecule A, and b indexes training molecules in the second set.
//
// Implementation strategy
// -----------------------
// The kernel factorises as:
//
//   K[A,b] = sum_{i in A, j in b: Zi=Zj}  exp( -(s_ii + s_jj - 2*s_ij) / sigma^2 )
//
// Differentiating:
//
//   dK[A,b]/dR[a,u] = sum_{i,j} k(i,j)/sigma^2 * (-ds_ii/dR[a,u] + 2*ds_ij/dR[a,u])
//
// s_jj (training self-scalar) is independent of R_A.
// Only s_ii and s_ij depend on R_A.
//
// The scalar product s(i,j) depends on R_A through:
//   - ksi2_i[p]  (two-body weight, function of r_{ip})
//   - cosp_i, sinp_i  (three-body Fourier, function of angles + distances in A)
//
// For s_ii (self-scalar of atom i), both sides are from A, so both ksi sets
// carry gradients, but by symmetry ds_ii/dR = 2 * d(s_ij)/dR|_{j=i,fixed}.
// We use the fact that scalar_noalchemy is symmetric in its two atom arguments
// and the self-scalar diagonal double-counts each pair — we handle this
// directly in the gradient accumulation.
//
// Per-atom gradient data stored in AtomGrad struct (parallel to AtomData).
// =============================================================================

// Derivative of cut_function w.r.t. r.
static inline double cut_function_deriv(double r, double cut_start, double cut_distance) {
    const double ru = cut_distance;
    const double rl = cut_start * cut_distance;

    if (r >= ru || r <= rl) return 0.0;

    const double x = (ru - r) / (ru - rl);
    const double dxdr = -1.0 / (ru - rl);
    // f(x) = 10x^3 - 15x^4 + 6x^5  =>  df/dx = 30x^2 - 60x^3 + 30x^4
    const double dfdx = 30.0 * x * x - 60.0 * x * x * x + 30.0 * x * x * x * x;
    return dfdx * dxdr;
}

// Derivative of fast_pow(r, p) w.r.t. r: p * r^(p-1)
static inline double fast_pow_deriv(double r, double p) {
    if (p == 0.0) return 0.0;
    return p * fast_pow(r, p - 1.0);
}

// =============================================================================
// AtomGrad: gradient of per-atom precomputed quantities w.r.t. R_A.
//
// For atom i of molecule A (with n_atoms_A atoms):
//
//   dksi[k][alpha][mu]  = d(ksi[k]) / d(R_A[alpha, mu])
//                          shape: (n_neigh, n_atoms_A, 3)   flat: [k * na3 + alpha*3 + mu]
//
//   dcosp[p][m][k][alpha][mu]  shape: (pmax * order * max_size, n_atoms_A, 3)
//                               flat: [pmax_order_neigh * na3 + alpha*3 + mu]
//   dsinp: same layout as dcosp
//
// Memory layout for dksi: (n_neigh * n_atoms_A * 3)
//                  dcosp: (pmax * order * max_size * n_atoms_A * 3)
// =============================================================================
struct AtomGrad {
    // dksi[k * na3 + alpha * 3 + mu]  -- na3 = n_atoms_A * 3
    std::vector<double> dksi;
    // dcosp/dsinp: [cosp_flat_idx * na3 + alpha * 3 + mu]
    std::vector<double> dcosp;
    std::vector<double> dsinp;
};

// =============================================================================
// Compute ksi and its gradient for one atom i of molecule A.
//
// atom_chan: (5, max_size) row for atom i
// centre_idx: which atom index is atom i in molecule A (for coordinate mapping)
// n_atoms_A: total real atoms in A
// For each neighbour k (1-based):
//   ksi[k] = f_cut(r_k) / r_k^power
//   d(ksi[k]) / d(R_A[alpha, mu]):
//     non-zero only for alpha = centre_idx (i) or alpha = nbr_atom_idx[k]
//     dr_k / d(R_A[alpha, mu]) = +(x_k[mu] - x_i[mu]) / r_k  for alpha=k_atom
//                               = -(x_k[mu] - x_i[mu]) / r_k  for alpha=i
//
// nbr_atom_idx[k]: original atom index in molecule A for neighbour slot k
// (this is stored in a separate array, built during repr generation)
// =============================================================================
static void compute_ksi_and_grad(
    const double *atom_chan,  // (5, max_size) slice for atom i
    int max_size, int n_neigh, double power, double cut_start, double cut_distance,
    int centre_atom_idx,  // index of atom i within molecule A
    int n_atoms_A,
    const std::vector<int> &nbr_atom_idx_i,  // [k] -> atom index in A for nbr slot k
    std::vector<double> &ksi,                // size n_neigh, output
    std::vector<double> &dksi                // size n_neigh * n_atoms_A * 3, output
) {
    const int na3 = n_atoms_A * 3;
    ksi.assign(n_neigh, 0.0);
    dksi.assign(static_cast<std::size_t>(n_neigh) * na3, 0.0);

    const double *dist_chan = atom_chan;          // channel 0
    const double *xc = atom_chan + 2 * max_size;  // channel 2: dx
    const double *yc = atom_chan + 3 * max_size;  // channel 3: dy
    const double *zc = atom_chan + 4 * max_size;  // channel 4: dz

    for (int k = 1; k < n_neigh; ++k) {
        const double r = dist_chan[k];
        if (r < 1e-14) continue;
        const double fc = cut_function(r, cut_start, cut_distance);
        const double rp = fast_pow(r, power);
        ksi[k] = fc / rp;

        // d(ksi[k])/dr = (fc' * rp - fc * p * r^(p-1)) / rp^2
        //              = (fc' - fc * p / r) / rp
        const double fcp = cut_function_deriv(r, cut_start, cut_distance);
        const double dksi_dr = (fcp - fc * power / r) / rp;

        // dr/dR[alpha, mu] = (dx, dy, dz)[k] / r  for alpha = k_atom
        //                  = -(dx, dy, dz)[k] / r  for alpha = centre
        const double dxk = xc[k], dyk = yc[k], dzk = zc[k];
        const double inv_r = 1.0 / r;

        const int k_atom = nbr_atom_idx_i[k];
        const std::size_t base_k = static_cast<std::size_t>(k) * na3 + k_atom * 3;
        const std::size_t base_centre = static_cast<std::size_t>(k) * na3 + centre_atom_idx * 3;

        // dksi[k] / dR[k_atom, (x,y,z)]
        dksi[base_k + 0] += dksi_dr * dxk * inv_r;
        dksi[base_k + 1] += dksi_dr * dyk * inv_r;
        dksi[base_k + 2] += dksi_dr * dzk * inv_r;

        // dksi[k] / dR[centre, (x,y,z)]
        dksi[base_centre + 0] -= dksi_dr * dxk * inv_r;
        dksi[base_centre + 1] -= dksi_dr * dyk * inv_r;
        dksi[base_centre + 2] -= dksi_dr * dzk * inv_r;
    }
}

// =============================================================================
// Compute three-body Fourier terms AND their gradients for one atom i of mol A.
//
// Gradient layout for dcosp/dsinp:
//   [cosp_flat_idx * na3 + alpha * 3 + mu]
//   where cosp_flat_idx = p * order * max_size + m * max_size + neigh
//         na3 = n_atoms_A * 3
//
// For each triplet (centre i, neighbour j, neighbour k>j):
// The contributing atoms to the gradient are: i, j_atom, k_atom.
// =============================================================================
static void compute_threebody_fourier_and_grad(
    const double *atom_chan,  // (5, max_size) slice for atom i
    int max_size, int n_neigh, double three_body_power, double cut_start, double cut_distance,
    int order, int pmax, const std::vector<int> &z_to_idx, bool use_atm, int centre_atom_idx,
    int n_atoms_A,
    const std::vector<int> &nbr_atom_idx_i,  // [k] -> atom index in A
    std::vector<double> &cosp,               // (pmax * order * max_size)        output
    std::vector<double> &sinp,               // (pmax * order * max_size)        output
    std::vector<double> &dcosp,              // (pmax * order * max_size * na3)  output
    std::vector<double> &dsinp               // (pmax * order * max_size * na3)  output
) {
    const int na3 = n_atoms_A * 3;
    const std::size_t sz = static_cast<std::size_t>(pmax) * order * max_size;
    const std::size_t sz_grad = sz * na3;

    cosp.assign(sz, 0.0);
    sinp.assign(sz, 0.0);
    dcosp.assign(sz_grad, 0.0);
    dsinp.assign(sz_grad, 0.0);

    auto flat_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size +
               static_cast<std::size_t>(m) * max_size + neigh;
    };
    // grad index: flat_idx(p,m,neigh) * na3 + alpha*3 + mu
    auto grad_offset = [&](std::size_t fidx, int alpha, int mu) -> std::size_t {
        return fidx * na3 + alpha * 3 + mu;
    };

    const double *dist_chan = atom_chan;
    const double *z_chan = atom_chan + max_size;
    const double *xc = atom_chan + 2 * max_size;
    const double *yc = atom_chan + 3 * max_size;
    const double *zc = atom_chan + 4 * max_size;

    // Precompute unit vectors, cut values, and intermediates for all neighbours
    std::vector<double> ux(n_neigh), uy(n_neigh), uz(n_neigh);
    std::vector<double> cut_k(n_neigh), dist_k(n_neigh), inv_r(n_neigh);
    for (int k = 1; k < n_neigh; ++k) {
        const double r = dist_chan[k];
        dist_k[k] = r;
        cut_k[k] = cut_function(r, cut_start, cut_distance);
        const double ir = (r > 1e-14) ? 1.0 / r : 0.0;
        inv_r[k] = ir;
        ux[k] = xc[k] * ir;
        uy[k] = yc[k] * ir;
        uz[k] = zc[k] * ir;
    }

    for (int j = 1; j < n_neigh; ++j) {
        const double dj = dist_k[j];
        if (dj < 1e-14) continue;
        const double cutj = cut_k[j];
        if (cutj == 0.0) continue;

        for (int k = j + 1; k < n_neigh; ++k) {
            const double dk = dist_k[k];
            if (dk < 1e-14) continue;
            const double cutk = cut_k[k];
            if (cutk == 0.0) continue;

            // Displacement j->k
            const double dxjk = xc[j] - xc[k];
            const double dyjk = yc[j] - yc[k];
            const double dzjk = zc[j] - zc[k];
            const double di2 = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
            const double di = std::sqrt(di2);
            if (di < 1e-14) continue;
            const double inv_di = 1.0 / di;

            const double cut_jk = cut_function(di, cut_start, cut_distance);
            if (cut_jk == 0.0) continue;

            // Angle at centre i
            const double cos_i_raw = ux[j] * ux[k] + uy[j] * uy[k] + uz[j] * uz[k];
            const double cos_i = clamp11(cos_i_raw);
            const double theta = std::acos(cos_i);
            // d(theta)/d(cos_i) = -1/sqrt(1 - cos_i^2)  (clamped away from ±1)
            const double sin_theta = std::sin(theta);  // >= 0

            // ATM factor and its derivative
            double atm;
            double datm_dcos_i = 0.0;
            double datm_dcos_j = 0.0;
            double datm_dcos_k = 0.0;
            double cos_j = 0.0, cos_k_val = 0.0;

            if (use_atm) {
                // unit vector from j toward k: ukj = (xc[k]-xc[j]) / di  (normalised j->k)
                // Angle at j: between direction to k and direction back to centre i
                // centre_from_j = -unit_vec_from_centre_to_j = (-ux[j], -uy[j], -uz[j])
                // k_from_j direction: (xc[k]-xc[j])/di
                const double ukj_x = -dxjk * inv_di;
                const double ukj_y = -dyjk * inv_di;
                const double ukj_z = -dzjk * inv_di;
                cos_j = clamp11(ukj_x * (-ux[j]) + ukj_y * (-uy[j]) + ukj_z * (-uz[j]));

                // Angle at k: direction to j from k and direction back to centre
                // j_from_k: (xc[j]-xc[k])/di = (dxjk/di, dyjk/di, dzjk/di)
                const double ujk_x = dxjk * inv_di;
                const double ujk_y = dyjk * inv_di;
                const double ujk_z = dzjk * inv_di;
                cos_k_val = clamp11(ujk_x * (-ux[k]) + ujk_y * (-uy[k]) + ujk_z * (-uz[k]));

                atm = 1.0 + 3.0 * cos_i * cos_j * cos_k_val;
                datm_dcos_i = 3.0 * cos_j * cos_k_val;
                datm_dcos_j = 3.0 * cos_i * cos_k_val;
                datm_dcos_k = 3.0 * cos_i * cos_j;
            } else {
                atm = 1.0;
            }
            if (atm == 0.0) continue;

            // ksi3 = cut_j * cut_k * cut_jk * atm / (dj * dk * di)^beta3
            const double dijk = dj * dk * di;
            const double dijk_p = fast_pow(dijk, three_body_power);
            const double cut_prod = cutj * cutk * cut_jk;
            const double ksi3 = cut_prod * atm / dijk_p;
            if (ksi3 == 0.0) continue;

            // ---------------------------------------------------------------
            // Compute gradient of ksi3 w.r.t. coordinates of i, j_atom, k_atom
            // ---------------------------------------------------------------
            // We need dksi3/dR[alpha, mu] for alpha in {i, j_atom, k_atom}.
            //
            // ksi3 = cut_prod * atm / dijk_p
            //
            // dksi3/dx = (d(cut_prod)/dx * atm + cut_prod * d(atm)/dx) / dijk_p
            //           - cut_prod * atm * d(dijk_p)/dx / dijk_p^2
            //
            // We'll accumulate into a 3x3 gradient block: grad_ksi3[atom_rel][mu]
            // atom_rel: 0=centre_i, 1=j_atom, 2=k_atom
            //
            // Intermediates:
            //   d(dj)/d(R[alpha,mu])  -- j is centre_i's neighbour
            //   d(dk)/d(R[alpha,mu])
            //   d(di)/d(R[alpha,mu])  -- di = dist between j_atom and k_atom
            // ---------------------------------------------------------------

            const int j_atom = nbr_atom_idx_i[j];
            const int k_atom = nbr_atom_idx_i[k];

            // d(dj)/dR: dj = |R[j_atom] - R[centre_i]|
            //   d(dj)/dR[j_atom, mu] = +(xc[j][mu]) / dj  (displacement j_atom - centre_i)
            //   d(dj)/dR[centre_i, mu] = -(xc[j][mu]) / dj
            // d(dk)/dR: symmetric with j
            // d(di)/dR: di = |R[k_atom] - R[j_atom]|
            //   d(di)/dR[k_atom, mu] = +(xc[k][mu] - xc[j][mu]) / di = -dxjk/di  (dxjk =
            //   xc[j]-xc[k]) d(di)/dR[j_atom, mu] = -(xc[k][mu] - xc[j][mu]) / di = +dxjk/di
            //   d(di)/dR[centre_i, mu] = 0  (di doesn't involve centre)

            // Three atoms may overlap: centre_i == j_atom or centre_i == k_atom
            // (when a molecule's atom is both a centre and a neighbour — possible since
            // k=0 is self but we start from k=1; however j_atom/k_atom could equal centre_atom_idx)
            // We handle it by just accumulating with the correct atom index.

            // Gradient storage: 3 atoms x 3 directions
            // Indices: 0=centre_i, 1=j_atom, 2=k_atom  (may coincide)
            // We'll use local arrays and scatter at the end.
            double gksi3[3][3] = {};  // [atom_local][mu]  -> atoms: {centre_i, j_atom, k_atom}

            const double fcp_j = cut_function_deriv(dj, cut_start, cut_distance);
            const double fcp_k = cut_function_deriv(dk, cut_start, cut_distance);
            const double fcp_jk = cut_function_deriv(di, cut_start, cut_distance);

            // d(log(cut_prod))/d(something) = fcp_j/cutj * ddj + fcp_k/cutk * ddk + fcp_jk/cut_jk *
            // ddi but we work with the full product to avoid division-by-zero when cuts are 0
            // (they're already guarded above)

            // d(ksi3)/d(something) = [d(cut_prod)/d(something) * atm
            //                         + cut_prod * d(atm)/d(something)] / dijk_p
            //                       - ksi3 * d(log(dijk_p))/d(something)
            //
            // d(log(dijk_p))/d(something) = beta3 * (d(log(dj)) + d(log(dk)) + d(log(di)))
            //                             = beta3 * (d(dj)/dj + d(dk)/dk + d(di)/di)

            // ---- contribution from d(cut_prod)/d(R) ----
            // d(cut_prod)/dR = fcp_j * cutk * cut_jk * d(dj)/dR
            //                + fcp_k * cutj * cut_jk * d(dk)/dR
            //                + fcp_jk * cutj * cutk  * d(di)/dR
            // scaled by atm / dijk_p

            // ---- contribution from d(atm)/d(R) via d(cos_i), d(cos_j), d(cos_k) ----
            // We need d(cos_i)/dR, d(cos_j)/dR, d(cos_k)/dR
            //
            // cos_i = ux[j]*ux[k]+uy[j]*uy[k]+uz[j]*uz[k]
            //       where u[k] = (xc[k]/dk, yc[k]/dk, zc[k]/dk)
            //
            // d(ux[k])/d(R[k_atom, mu]) = (delta_mu_x * dk - xc[k] * d(dk)/d(R[k_atom,mu])) / dk^2
            //                           = (delta_mu_x - ux[k]*ux[k]) / dk  (for mu=x, using
            //                           d(dk)/dR[k]=ux[k])
            // In general: d(u_j[mu_hat])/dR[j_atom, mu] = (delta_{mu_hat,mu} - u_j[mu_hat]*u_j[mu])
            // / dj
            //
            // This is the standard "Jacobian of unit vector" formula.

            // Local coordinate arrays
            const double ui_j[3] = {ux[j], uy[j], uz[j]};  // unit vec centre->j
            const double ui_k[3] = {ux[k], uy[k], uz[k]};  // unit vec centre->k
            // unit vec j->k (for cos_j computation)
            const double ukj[3] = {-dxjk * inv_di, -dyjk * inv_di, -dzjk * inv_di};
            // unit vec k->j (for cos_k computation)
            const double ujk[3] = {dxjk * inv_di, dyjk * inv_di, dzjk * inv_di};

            // d(cos_i)/dR[alpha, mu]:
            // cos_i = ui_j . ui_k
            // Depends on j_atom and k_atom and centre_i:
            //   d(cos_i)/dR[j_atom, mu] += sum_mu' (d(ui_j[mu'])/dR[j_atom,mu]) * ui_k[mu']
            //     = (ui_k[mu] - cos_i * ui_j[mu]) / dj
            //   d(cos_i)/dR[k_atom, mu] += (ui_j[mu] - cos_i * ui_k[mu]) / dk
            //   d(cos_i)/dR[centre_i, mu] = -d/dR[j] - d/dR[k]  (centre shift = minus both)
            //     = -(ui_k[mu] - cos_i*ui_j[mu])/dj - (ui_j[mu] - cos_i*ui_k[mu])/dk

            double dcos_i_dR[3][3] = {};  // [local_atom][mu]
            for (int mu = 0; mu < 3; ++mu) {
                dcos_i_dR[1][mu] = (ui_k[mu] - cos_i * ui_j[mu]) / dj;    // j_atom
                dcos_i_dR[2][mu] = (ui_j[mu] - cos_i * ui_k[mu]) / dk;    // k_atom
                dcos_i_dR[0][mu] = -dcos_i_dR[1][mu] - dcos_i_dR[2][mu];  // centre_i
            }

            // d(cos_j)/dR:
            // cos_j = ukj . (-ui_j)
            //       = -ukj . ui_j
            // ukj = unit vector from j toward k = (R[k_atom]-R[j_atom]) / di
            //       but xc[k]-xc[j] = (R[k]-R[i]) - (R[j]-R[i]) = R[k]-R[j]
            //       so ukj[mu] = (xc[k][mu] - xc[j][mu]) / (-di)  = -dxjk[mu]/di  (matches above)
            // In terms of positions: ukj = (R[k_atom] - R[j_atom]) / di
            //   d(ukj[mu'])/dR[k_atom,mu] = (delta_{mu',mu} - ukj[mu']*(-ukj[mu])) / di
            //     Note: d(di)/dR[k_atom,mu] = -dxjk[mu]/di * (-1) ... let's be careful.
            //     di = |R[k]-R[j]|, let v = R[k]-R[j]
            //     d(di)/dR[k,mu] = v[mu]/di  = -dxjk[mu]/di  (since dxjk[mu]=R[j][mu]-R[k][mu])
            //                                = +(-dxjk[mu])/di
            //     Hmm: xc[j] = R[j]-R[i], xc[k] = R[k]-R[i]
            //          dxjk = xc[j] - xc[k] = R[j]-R[k]
            //     So v = R[k]-R[j] = -dxjk
            //     d(di)/dR[k,mu] = -dxjk[mu]/di
            //     d(di)/dR[j,mu] = +dxjk[mu]/di
            //     d(di)/dR[i,mu] = 0
            //   d(ukj[mu'])/dR[k_atom,mu] = (delta_{mu',mu} - ukj[mu']*ukj[mu]) / di
            //     (using ukj[mu] = -dxjk[mu]/di = v[mu]/di)
            //   d(ukj[mu'])/dR[j_atom,mu] = -(delta_{mu',mu} - ukj[mu']*ukj[mu]) / di
            //
            // cos_j = -sum_{mu'} ukj[mu'] * ui_j[mu']
            //   d(cos_j)/dR[k_atom,mu] = -sum_{mu'} d(ukj[mu'])/dR[k,mu] * ui_j[mu']
            //       + 0   (ui_j doesn't depend on k)
            //     = -(1/di) * (ui_j[mu] - cos_j_from_ukj_only * ukj[mu])
            //   where "cos_j_from_ukj_only" = -cos_j (since cos_j = -ukj.ui_j)
            //     d(cos_j)/dR[k,mu] = -(1/di) * (ui_j[mu] + cos_j * ukj[mu])
            //
            //   d(cos_j)/dR[j_atom,mu]: depends on both ui_j and ukj
            //     d(ui_j[mu'])/dR[j_atom,mu] = -(delta_{mu',mu} - ui_j[mu']*ui_j[mu]) / dj
            //       (j_atom moves -> j vector shrinks: R[j]-R[i], d(unit_vec)/dR[j] = -(I -
            //       uu^T)/r) Wait: unit_j = (R[j]-R[i])/dj, d/dR[j,mu] of unit_j[mu'] =
            //       (delta_{mu',mu} - unit_j[mu']*unit_j[mu])/dj
            //     d(ukj[mu'])/dR[j_atom,mu] = -(delta_{mu',mu} - ukj[mu']*ukj[mu]) / di
            //     d(cos_j)/dR[j,mu] = -sum_{mu'} [d(ukj[mu'])/dR[j,mu]*ui_j[mu'] +
            //     ukj[mu']*d(ui_j[mu'])/dR[j,mu]]
            //       = -[-(ui_j[mu] - (ukj.ui_j)*ukj[mu])/di  +  ((-ukj.ui_j) - (ui_j.ui_j*ui_j[mu]
            //       - ui_j[mu]))/dj] hmm this is getting messy; let's be systematic
            //
            // cos_j = (ukj_x * (-ui_j_x) + ukj_y * (-ui_j_y) + ukj_z * (-ui_j_z))
            //       = -(ukj . ui_j)   = dot(ukj, -ui_j)
            //
            // ukj[mu] = -dxjk[mu] / di  =  (R[k][mu]-R[j][mu]) / di
            // ui_j[mu] = xc[j][mu] / dj  = (R[j][mu]-R[i][mu]) / dj
            //
            // d(cos_j)/dR[i, mu]:   only ui_j depends on R[i]
            //   d(ui_j[mu'])/dR[i,mu] = -(delta_{mu',mu} - ui_j[mu']*ui_j[mu]) / dj * (-1)
            //                         = +(delta_{mu',mu} - ui_j[mu']*ui_j[mu]) / dj
            //   d(cos_j)/dR[i,mu] = -sum_{mu'} ukj[mu'] * d(ui_j[mu'])/dR[i,mu]
            //     = -(1/dj)*(ukj[mu] - (ukj.ui_j)*ui_j[mu])
            //     = -(1/dj)*(ukj[mu] + cos_j * ui_j[mu])
            //
            // d(cos_j)/dR[j, mu]: both ui_j and ukj depend on R[j]
            //   d(ui_j[mu'])/dR[j,mu] = +(delta_{mu',mu} - ui_j[mu']*ui_j[mu]) / dj
            //   d(ukj[mu'])/dR[j,mu]  = -(delta_{mu',mu} - ukj[mu']*ukj[mu]) / di
            //   d(cos_j)/dR[j,mu] = -sum_{mu'} [d(ukj[mu'])/dR[j,mu]*ui_j[mu'] +
            //   ukj[mu']*d(ui_j[mu'])/dR[j,mu]]
            //     = -(1/di)*(-(ui_j[mu] - (ukj.ui_j)*ukj[mu]))  [from ukj term]
            //       -(1/dj)*(ukj[mu] - (ukj.ui_j)*ui_j[mu])     [from ui_j term, sign:
            //       -ukj.d(ui_j)]
            //     = +(1/di)*(ui_j[mu] + cos_j*ukj[mu])
            //       -(1/dj)*(ukj[mu] + cos_j*ui_j[mu])
            //
            // d(cos_j)/dR[k, mu]: only ukj depends on R[k]
            //   d(ukj[mu'])/dR[k,mu] = +(delta_{mu',mu} - ukj[mu']*ukj[mu]) / di
            //   d(cos_j)/dR[k,mu] = -sum_{mu'} d(ukj[mu'])/dR[k,mu] * ui_j[mu']
            //     = -(1/di)*(ui_j[mu] - (ukj.ui_j)*ukj[mu])
            //     = -(1/di)*(ui_j[mu] + cos_j*ukj[mu])

            double dcos_j_dR[3][3] = {};  // [local_atom: 0=centre_i, 1=j_atom, 2=k_atom][mu]
            if (use_atm) {
                for (int mu = 0; mu < 3; ++mu) {
                    // d(cos_j)/dR[centre_i, mu]
                    // cos_j = -(ukj.ui_j), ui_j = (R_j-R_i)/dj
                    // d(ui_j)/dR_i = -(I - ui_j*ui_j^T)/dj
                    // d(cos_j)/dR_i = -ukj . d(ui_j)/dR_i = +(1/dj)*(ukj - (ukj.ui_j)*ui_j)
                    //               = +(1/dj)*(ukj[mu] + cos_j*ui_j[mu])
                    dcos_j_dR[0][mu] = +(1.0 / dj) * (ukj[mu] + cos_j * ui_j[mu]);
                    // d(cos_j)/dR[j_atom, mu]
                    dcos_j_dR[1][mu] = +(1.0 / di) * (ui_j[mu] + cos_j * ukj[mu]) -
                                       (1.0 / dj) * (ukj[mu] + cos_j * ui_j[mu]);
                    // d(cos_j)/dR[k_atom, mu]
                    dcos_j_dR[2][mu] = -(1.0 / di) * (ui_j[mu] + cos_j * ukj[mu]);
                }
            }

            // d(cos_k)/dR:
            // cos_k = (ujk . (-ui_k))  where ujk = (R[j]-R[k])/di, ui_k = (R[k]-R[i])/dk
            //       = -(ujk . ui_k)
            // Symmetric to cos_j with j<->k swap:
            //   ujk[mu] = dxjk[mu]/di = (R[j][mu]-R[k][mu])/di
            //   ui_k[mu] = xc[k][mu]/dk
            //
            // d(cos_k)/dR[centre_i, mu]:  only ui_k depends on R[i]
            //   = -(1/dk)*(ujk[mu] + cos_k_val * ui_k[mu])
            //
            // d(cos_k)/dR[j_atom, mu]:  only ujk depends on R[j]
            //   d(ujk[mu'])/dR[j,mu] = +(delta_{mu',mu} - ujk[mu']*ujk[mu]) / di
            //   d(cos_k)/dR[j,mu] = -(1/di)*(ui_k[mu] + cos_k_val*ujk[mu])
            //
            // d(cos_k)/dR[k_atom, mu]:  both ujk and ui_k depend on R[k]
            //   d(ujk[mu'])/dR[k,mu] = -(delta_{mu',mu} - ujk[mu']*ujk[mu]) / di
            //   d(ui_k[mu'])/dR[k,mu] = +(delta_{mu',mu} - ui_k[mu']*ui_k[mu]) / dk
            //   d(cos_k)/dR[k,mu] = +(1/di)*(ui_k[mu] + cos_k_val*ujk[mu])
            //                        -(1/dk)*(ujk[mu] + cos_k_val*ui_k[mu])

            double dcos_k_dR[3][3] = {};  // [local_atom: 0=centre_i, 1=j_atom, 2=k_atom][mu]
            if (use_atm) {
                for (int mu = 0; mu < 3; ++mu) {
                    // d(cos_k)/dR[centre_i, mu]
                    // cos_k = -(ujk.ui_k), ui_k = (R_k-R_i)/dk
                    // d(ui_k)/dR_i = -(I - ui_k*ui_k^T)/dk
                    // d(cos_k)/dR_i = -ujk . d(ui_k)/dR_i = +(1/dk)*(ujk[mu] + cos_k*ui_k[mu])
                    dcos_k_dR[0][mu] = +(1.0 / dk) * (ujk[mu] + cos_k_val * ui_k[mu]);
                    dcos_k_dR[1][mu] = -(1.0 / di) * (ui_k[mu] + cos_k_val * ujk[mu]);
                    dcos_k_dR[2][mu] = +(1.0 / di) * (ui_k[mu] + cos_k_val * ujk[mu]) -
                                       (1.0 / dk) * (ujk[mu] + cos_k_val * ui_k[mu]);
                }
            }

            // Now assemble dksi3/dR[local_atom, mu]
            // ksi3 = cut_prod * atm / dijk_p
            //
            // dijk_p = (dj*dk*di)^beta3
            // d(dijk_p)/dR[...] = beta3 * dijk_p * (d(dj)/dj/dR + d(dk)/dk/dR + d(di)/di/dR)
            //   = beta3 * dijk_p * (d_log_dj/dR + d_log_dk/dR + d_log_di/dR)
            //
            // d(dj)/dR[j,mu] = xc[j][mu]/dj = ui_j[mu]
            // d(dj)/dR[i,mu] = -ui_j[mu]
            // d(dk)/dR[k,mu] = ui_k[mu]  (xc[k][mu]/dk)
            // d(dk)/dR[i,mu] = -ui_k[mu]
            // d(di)/dR[k,mu] = -dxjk[mu]/di = ukj[mu]  (=(R[k]-R[j])[mu]/di)
            // d(di)/dR[j,mu] = +dxjk[mu]/di = ujk[mu]  (=(R[j]-R[k])[mu]/di)
            // d(di)/dR[i,mu] = 0

            // Precompute f' / dijk_p for each cut contribution:
            const double fcp_j_over_p = (cutj > 0.0) ? (fcp_j * cutk * cut_jk) / dijk_p : 0.0;
            const double fcp_k_over_p = (cutk > 0.0) ? (fcp_k * cutj * cut_jk) / dijk_p : 0.0;
            const double fcp_jk_over_p = (cut_jk > 0.0) ? (fcp_jk * cutj * cutk) / dijk_p : 0.0;
            const double beta3_ksi3_inv =
                three_body_power / dijk_p * cut_prod;  // = beta3 * cut_prod / dijk_p

            // gksi3[local_atom][mu]:
            // local: 0=centre_i, 1=j_atom, 2=k_atom
            for (int mu = 0; mu < 3; ++mu) {
                // Displacement derivatives:
                // d(dj)/d(R[centre_i,mu]) = -ui_j[mu]
                // d(dj)/d(R[j_atom, mu])  = +ui_j[mu]
                // d(dk)/d(R[centre_i,mu]) = -ui_k[mu]
                // d(dk)/d(R[k_atom, mu])  = +ui_k[mu]
                // d(di)/d(R[j_atom, mu])  = +ujk[mu]
                // d(di)/d(R[k_atom, mu])  = -ujk[mu] = +ukj[mu]... wait:
                // di = |R[k]-R[j]|, d(di)/dR[k,mu] = (R[k]-R[j])[mu]/di = -dxjk[mu]/di
                // xc[k]-xc[j] = (R[k]-R[i])-(R[j]-R[i]) = R[k]-R[j]
                // dxjk = xc[j]-xc[k] = R[j]-R[k], so -dxjk[mu]/di = (R[k]-R[j])[mu]/di = ukj[mu]
                // d(di)/dR[k,mu] = ukj[mu]     (= -dxjk[mu]/di)
                // d(di)/dR[j,mu] = -ukj[mu]    (= +dxjk[mu]/di = ujk[mu])

                // Contributions from d(cut_prod)/dR * atm / dijk_p:
                // centre_i: d(dj)/dR[i]*fcp_j_over_p + d(dk)/dR[i]*fcp_k_over_p + 0
                //         = (-ui_j[mu]) * fcp_j_over_p * atm + (-ui_k[mu]) * fcp_k_over_p * atm
                // j_atom:  d(dj)/dR[j]*fcp_j_over_p*atm + d(di)/dR[j]*fcp_jk_over_p*atm
                //         = ui_j[mu]*fcp_j_over_p*atm + (-ukj[mu])*fcp_jk_over_p*atm
                //   Note: dxjk[mu]/di = ujk[mu], so d(di)/dR[j,mu] = ujk[mu]... let me recheck:
                //   di = |xc[k]-xc[j]|... correction: I said d(di)/dR[j,mu] = +dxjk[mu]/di =
                //   ujk[mu] but dxjk = xc[j]-xc[k], so dxjk[mu]/di means we need the unit vector
                //   from k to j. Let's be careful: ujk[mu] = dxjk[mu]/di = (xc[j]-xc[k])[mu]/di
                //   d(di)/dR[j_atom, mu] = d|xc[k]-xc[j]|/dR[j,mu]
                //     = (xc[k]-xc[j])[mu]/di * d(xc[j])/dR[j] ... but d(xc[j][mu'])/dR[j,mu] =
                //     delta_{mu',mu} = -(xc[k]-xc[j])[mu]/di = -(-dxjk[mu])/di = dxjk[mu]/di =
                //     ujk[mu]
                //   Hmm that's wrong sign. Let me redo:
                //   di = sqrt((xc[k]-xc[j])^2), d(di)/d(xc[j][mu]) = -(xc[k]-xc[j])[mu]/di
                //   and d(xc[j][mu])/dR[j_atom, mu'] = delta_{mu,mu'} (xc[j] = R[j]-R[i])
                //   So d(di)/dR[j_atom,mu] = d(di)/d(xc[j][mu]) = -(xc[k][mu]-xc[j][mu])/di =
                //   dxjk[mu]/di And d(di)/dR[k_atom,mu] = d(di)/d(xc[k][mu]) =
                //   +(xc[k][mu]-xc[j][mu])/di = -dxjk[mu]/di = ukj[mu] (since di depends on
                //   xc[k]-xc[j])
                //
                // So:
                //   d(di)/dR[j_atom, mu] = dxjk[mu]/di  = ujk[mu]  ... note ujk = dxjk/di above
                //   d(di)/dR[k_atom, mu] = -dxjk[mu]/di = -ujk[mu] = ukj defined as -dxjk/di...
                //
                //   I defined ukj[mu] = -dxjk[mu]/di above (in unit vector precompute), so:
                //   d(di)/dR[j_atom, mu] = -ukj[mu]
                //   d(di)/dR[k_atom, mu] = +ukj[mu]

                // d(cut_prod)/dR * atm / dijk_p
                // centre_i:
                gksi3[0][mu] += atm * (-ui_j[mu] * fcp_j_over_p + -ui_k[mu] * fcp_k_over_p);
                // j_atom:
                gksi3[1][mu] += atm * (ui_j[mu] * fcp_j_over_p + (-ukj[mu]) * fcp_jk_over_p);
                // k_atom:
                gksi3[2][mu] += atm * (ui_k[mu] * fcp_k_over_p + (+ukj[mu]) * fcp_jk_over_p);

                // Contribution from - ksi3 * d(log(dijk_p))/dR
                //   = - ksi3 * beta3 * (d(dj)/dR/dj + d(dk)/dR/dk + d(di)/dR/di)
                //   = - beta3 * (cut_prod*atm/dijk_p) * (d(dj)/dR/dj + d(dk)/dR/dk + d(di)/dR/di)
                //
                // d(dj)/dR[centre,mu]/dj = -ui_j[mu]/dj
                // d(dj)/dR[j_atom,mu]/dj = +ui_j[mu]/dj
                // d(dk)/dR[centre,mu]/dk = -ui_k[mu]/dk
                // d(dk)/dR[k_atom,mu]/dk = +ui_k[mu]/dk
                // d(di)/dR[j_atom,mu]/di = -ukj[mu]/di   (ujk[mu]/di... see above:
                // d(di)/dR[j]=-ukj) d(di)/dR[k_atom,mu]/di = +ukj[mu]/di
                const double inv_dj = 1.0 / dj;
                const double inv_dk = 1.0 / dk;
                const double inv_di_val = inv_di;  // already computed above
                gksi3[0][mu] -= atm * beta3_ksi3_inv * (-ui_j[mu] * inv_dj + -ui_k[mu] * inv_dk);
                gksi3[1][mu] -=
                    atm * beta3_ksi3_inv * (ui_j[mu] * inv_dj + (-ukj[mu]) * inv_di_val);
                gksi3[2][mu] -=
                    atm * beta3_ksi3_inv * (ui_k[mu] * inv_dk + (+ukj[mu]) * inv_di_val);

                // Contribution from cut_prod/dijk_p * d(atm)/dR (only when use_atm):
                if (use_atm) {
                    const double scale = cut_prod / dijk_p;
                    for (int la = 0; la < 3; ++la) {
                        gksi3[la][mu] += scale * (datm_dcos_i * dcos_i_dR[la][mu] +
                                                  datm_dcos_j * dcos_j_dR[la][mu] +
                                                  datm_dcos_k * dcos_k_dR[la][mu]);
                    }
                }
            }

            // Now compute d(cosp/sinp) / dR for this triplet (j,k)
            // cosp[pj, m, j] += cos_m  where cos_m = (cos(m*theta) - cos(m*(theta+pi))) * ksi3
            // sinp[pj, m, j] += sin_m  where sin_m = (sin(m*theta) - sin(m*(theta+pi))) * ksi3
            //
            // d(cos_m)/dR = d(cos_m_factor)/dR * ksi3 + cos_m_factor * d(ksi3)/dR
            // where cos_m_factor = cos(m*theta) - cos(m*(theta+pi))
            //
            // d(theta)/d(cos_i) = -1/sin(theta)  (for sin_theta != 0)
            // d(cos_i)/dR: computed above in dcos_i_dR

            const int zk = static_cast<int>(z_chan[k]);
            const int zj = static_cast<int>(z_chan[j]);
            if (zk <= 0 || zk >= 256 || zj <= 0 || zj >= 256) continue;
            const int pj = z_to_idx[zk];  // compact index for element at k goes to slot j
            const int pk = z_to_idx[zj];  // compact index for element at j goes to slot k
            if (pj < 0 || pk < 0) continue;

            const int global_atoms[3] = {centre_atom_idx, j_atom, k_atom};

            for (int m = 0; m < order; m += 2) {
                const double mf = static_cast<double>(m + 1);
                const double mth = mf * theta;
                const double c = std::cos(mth);
                const double s = std::sin(mth);

                // Odd n=m+1: cos(nθ)-cos(n(θ+π)) = 2 cos(nθ); even n identically 0.
                const double cos_mfac = 2.0 * c;
                const double sin_mfac = 2.0 * s;

                const double cos_m = cos_mfac * ksi3;
                const double sin_m = sin_mfac * ksi3;

                // d(cos_m_factor)/d(theta) = -2*mf*sin(mth)
                const double dcos_mfac_dtheta = -2.0 * mf * s;
                const double dsin_mfac_dtheta = 2.0 * mf * c;

                // d(theta)/dR[la,mu] = d(theta)/d(cos_i) * d(cos_i)/dR[la,mu]
                //                   = (-1/sin_theta) * dcos_i_dR[la][mu]
                //   (sin_theta >= 0; if 0, cos_i = ±1, derivative is 0 by clamping)

                // Accumulate into the 4 slots: (pj,m,j), (pk,m,k)  [each from ksi3]
                const std::size_t fidx_j = flat_idx(pj, m, j);
                const std::size_t fidx_k = flat_idx(pk, m, k);

                // Forward value (already done in the forward pass; accumulate here too)
                cosp[fidx_j] += cos_m;
                sinp[fidx_j] += sin_m;
                cosp[fidx_k] += cos_m;
                sinp[fidx_k] += sin_m;

                // Gradient
                for (int la = 0; la < 3; ++la) {
                    const int alpha = global_atoms[la];
                    for (int mu = 0; mu < 3; ++mu) {
                        // d(cos_m)/dR[alpha,mu] = dcos_mfac_dtheta * d(theta)/dR * ksi3
                        //                       + cos_mfac * d(ksi3)/dR
                        double dtheta_dR = 0.0;
                        if (sin_theta > 1e-10) {
                            dtheta_dR = (-1.0 / sin_theta) * dcos_i_dR[la][mu];
                        }
                        const double dcos_m =
                            dcos_mfac_dtheta * dtheta_dR * ksi3 + cos_mfac * gksi3[la][mu];
                        const double dsin_m =
                            dsin_mfac_dtheta * dtheta_dR * ksi3 + sin_mfac * gksi3[la][mu];

                        dcosp[grad_offset(fidx_j, alpha, mu)] += dcos_m;
                        dsinp[grad_offset(fidx_j, alpha, mu)] += dsin_m;
                        dcosp[grad_offset(fidx_k, alpha, mu)] += dcos_m;
                        dsinp[grad_offset(fidx_k, alpha, mu)] += dsin_m;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Per-atom precomputed data WITH gradients for molecule A.
// =============================================================================
struct AtomDataGrad {
    int n_neigh;
    std::vector<double> ksi;
    std::vector<double> cosp;
    std::vector<double> sinp;
    // Gradients: d(ksi[k])/dR[alpha,mu]  layout: [k * na3 + alpha*3 + mu]
    std::vector<double> dksi;
    // d(cosp[p,m,neigh])/dR[alpha,mu]    layout: [flat_idx * na3 + alpha*3 + mu]
    std::vector<double> dcosp;
    std::vector<double> dsinp;
    // Neighbour slot k -> atom index in molecule A (built once, reused for dri).
    std::vector<int> nbr_atom_idx;
};

// Build neighbour atom index map for one atom i in molecule A.
// This maps neighbour slot k -> original atom index in molecule A.
// The representation stores neighbours sorted by distance; we need to recover
// which atom in A corresponds to each neighbour slot.
//
// We do this by matching: for each neighbour slot k, the displacement (dx,dy,dz)
// and nuclear charge Z must match exactly one atom j in A.
static std::vector<int> build_nbr_atom_idx(
    const double *atom_chan,  // (5, max_size) for atom i
    int max_size, int n_neigh,
    const std::vector<double> &coords_A,  // (n_atoms_A * 3)
    const std::vector<int> &z_A, int centre_idx, int n_atoms_A
) {
    std::vector<int> result(n_neigh, -1);
    result[0] = centre_idx;  // k=0 is always self

    const double *xc = atom_chan + 2 * max_size;
    const double *yc = atom_chan + 3 * max_size;
    const double *zc = atom_chan + 4 * max_size;
    const double *zc1 = atom_chan + 1 * max_size;  // nuclear charges

    for (int k = 1; k < n_neigh; ++k) {
        const double dx = xc[k], dy = yc[k], dz = zc[k];
        const int zval = static_cast<int>(zc1[k]);

        // Find which atom in A has this displacement and nuclear charge
        for (int j = 0; j < n_atoms_A; ++j) {
            if (z_A[j] != zval) continue;
            const double ex = coords_A[j * 3 + 0] - coords_A[centre_idx * 3 + 0];
            const double ey = coords_A[j * 3 + 1] - coords_A[centre_idx * 3 + 1];
            const double ez = coords_A[j * 3 + 2] - coords_A[centre_idx * 3 + 2];
            if (std::abs(ex - dx) < 1e-10 && std::abs(ey - dy) < 1e-10 &&
                std::abs(ez - dz) < 1e-10) {
                result[k] = j;
                break;
            }
        }
    }
    return result;
}

// Precompute all per-atom data WITH gradients for molecule A.
// parallel_atoms: OpenMP over atoms when not already in a parallel region
// (avoids nested OpenMP; caller disables for large nm2 where B-parallelism dominates).
static std::vector<AtomDataGrad> precompute_atom_data_grad(
    const std::vector<double> &x,     // (max_size, 5, max_size) for one molecule
    const std::vector<int> &n_vec,    // {n_atoms_A}
    const std::vector<int> &nn_flat,  // (max_size) neighbour counts
    int n_atoms_A, int max_size, const std::vector<double> &coords_A, const std::vector<int> &z_A,
    double two_body_power, double cut_start, double cut_distance, double three_body_power,
    int order, int pmax, const std::vector<int> &z_to_idx, bool use_atm, bool parallel_atoms
) {
    std::vector<AtomDataGrad> data(n_atoms_A);
    const std::size_t atom_stride = static_cast<std::size_t>(5) * max_size;
    (void)n_vec;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) \
    if (parallel_atoms && n_atoms_A >= 4 && !omp_in_parallel())
#endif
    for (int i = 0; i < n_atoms_A; ++i) {
        AtomDataGrad &ad = data[static_cast<std::size_t>(i)];
        const int n_neigh = nn_flat[static_cast<std::size_t>(i)];
        ad.n_neigh = n_neigh;

        const double *atom_chan = x.data() + static_cast<std::size_t>(i) * atom_stride;

        // Build neighbour -> atom-index map
        ad.nbr_atom_idx =
            build_nbr_atom_idx(atom_chan, max_size, n_neigh, coords_A, z_A, i, n_atoms_A);
        const std::vector<int> &nbr_idx = ad.nbr_atom_idx;

        compute_ksi_and_grad(
            atom_chan,
            max_size,
            n_neigh,
            two_body_power,
            cut_start,
            cut_distance,
            i,
            n_atoms_A,
            nbr_idx,
            ad.ksi,
            ad.dksi
        );

        compute_threebody_fourier_and_grad(
            atom_chan,
            max_size,
            n_neigh,
            three_body_power,
            cut_start,
            cut_distance,
            order,
            pmax,
            z_to_idx,
            use_atm,
            i,
            n_atoms_A,
            nbr_idx,
            ad.cosp,
            ad.sinp,
            ad.dcosp,
            ad.dsinp
        );
    }
    return data;
}

// =============================================================================
// Gradient of scalar_noalchemy w.r.t. R_A.
//
// Computes s(i,j) AND d(s(i,j))/dR_A[alpha,mu].
//
// Atom i is from molecule A (side 1, carries gradients).
// Atom j is from training set B (side 2, fixed).
//
// d(s)/dR_A[alpha,mu] = sum_{p,q: Zp=Zq}  d(G(r_p,r_q))/dR * W(p,q)
//                     + G(r_p,r_q) * d(W(p,q))/dR
//
// where G = exp(-(r_p-r_q)^2/(4w^2)),  W = ksi_i[p]*ksi_j[q]*dist_scale + angular*ang_scale
//
// d(G)/dR comes through r_p (from atom i's neighbourhood), not r_q (fixed).
// d(W)/dR comes through dksi_i[p]/dR and d(angular_i)/dR.
//
// Output: ds_dR[alpha*3 + mu]  size n_atoms_A*3
// =============================================================================
static double scalar_noalchemy_and_grad(
    const double *x1_chan, int max_size1, int n1, const double *ksi1,
    const double *dksi1,  // ksi1[k], dksi1[k*na3+alpha*3+mu]
    const std::vector<double> &cos1, const std::vector<double> &sin1,
    const std::vector<double> &dcos1, const std::vector<double> &dsin1, const double *x2_chan,
    int max_size2, int n2, const double *ksi2, const std::vector<double> &cos2,
    const std::vector<double> &sin2, double d_width, int order, int pmax, const double *s_prefactor,
    double distance_scale, double angular_scale,
    int n_atoms_A,              // for gradient output size
    std::vector<double> &ds_dR  // output: size n_atoms_A*3
) {
    // Early exit: central atoms must have the same nuclear charge (matches Fortran
    // scalar_noalchemy)
    const int Z1 = static_cast<int>(x1_chan[1 * max_size1 + 0]);
    const int Z2 = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    ds_dR.assign(static_cast<std::size_t>(n_atoms_A) * 3, 0.0);

    if (Z1 != Z2) return 0.0;

    const double inv_width = -1.0 / (4.0 * d_width * d_width);
    const double maxgausdist2 = (8.0 * d_width) * (8.0 * d_width);

    double aadist = 1.0;
    const int na3 = n_atoms_A * 3;

    auto cos1_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size1 +
               static_cast<std::size_t>(m) * max_size1 + neigh;
    };
    auto cos2_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size2 +
               static_cast<std::size_t>(m) * max_size2 + neigh;
    };

    for (int i = 1; i < n1; ++i) {
        const int Zi = static_cast<int>(x1_chan[1 * max_size1 + i]);
        const double ri = x1_chan[i];  // channel 0

        for (int j = 1; j < n2; ++j) {
            const int Zj = static_cast<int>(x2_chan[1 * max_size2 + j]);
            if (Zi != Zj) continue;

            const double rj = x2_chan[j];
            const double dr = ri - rj;
            const double r2 = dr * dr;
            if (r2 >= maxgausdist2) continue;

            const double G = std::exp(r2 * inv_width);
            // dG/d(ri) = G * 2*(ri-rj) * inv_width = G * 2*dr * inv_width
            const double dG_dri = G * 2.0 * dr * inv_width;

            // Angular term (even Fourier modes are identically zero).
            double angular = 0.0;
            for (int m = 0; m < order; m += 2) {
                double ang_m = 0.0;
                for (int p = 0; p < pmax; ++p) {
                    ang_m += cos1[cos1_idx(p, m, i)] * cos2[cos2_idx(p, m, j)] +
                             sin1[cos1_idx(p, m, i)] * sin2[cos2_idx(p, m, j)];
                }
                angular += ang_m * s_prefactor[m];
            }

            const double W_ksi = ksi1[i] * ksi2[j];
            const double W = W_ksi * distance_scale + angular * angular_scale;

            aadist += G * W;

            // Gradient accumulation
            // d(G*W)/dR[alpha,mu] = dG/dR[alpha,mu] * W + G * dW/dR[alpha,mu]
            //
            // dG/dR[alpha,mu] = dG/d(ri) * d(ri)/dR[alpha,mu]
            //   d(ri)/dR[alpha,mu] = dksi-like... but ri is the distance for the
            //   RADIAL Gaussian, not ksi. ri = x1_chan[0][i] = distance of neighbour i
            //   from centre atom of side 1.
            //   d(ri)/dR[alpha,mu]: from dksi computation we know:
            //   d(r_{i_neigh})/dR[alpha,mu]: same formula as ksi derivative but using
            //   just the distance piece. Let's get it from dksi.
            //   Actually: d(ksi[i])/dR = (fc'(r) - fc(r)*power/r) / r^power * d(r)/dR
            //   We need d(r)/dR independently. We stored d(ksi)/dR in dksi.
            //   We can recompute d(ri)/dR from the representation geometry.
            //
            // For this, we need the Jacobian d(ri)/dR separately.
            // We'll get it from the same displacement stored in x1_chan:
            //   d(xc[i])/dR[i_atom, mu]: +delta_mu  (i_atom is the neighbour of centre)
            //   d(xc[i])/dR[centre, mu]: -delta_mu
            //   d(ri)/dR[...] = (xc[i][mu] / ri) for the relevant atoms
            //
            // Rather than re-derive, we pass the gradient of ksi in dksi1 and use:
            //   dksi1[i*na3+alpha*3+mu] = d(ksi[i])/dR = d(f_cut(ri)/ri^p)/dR
            //                           = (fc'(ri) - fc(ri)*p/ri) / ri^p * d(ri)/dR
            //
            // We can recover d(ri)/dR from dksi[i] if we know the scalar factor, but
            // it's cleaner to store d(ri)/dR separately. However, to avoid storing
            // extra data, we observe:
            //
            //   d(G)/d(ri) * d(ri)/dR[alpha,mu]
            //
            // And d(ri)/dR has the same support pattern as d(ksi1[i])/dR.
            // We compute d(ri)/dR directly from the geometry.
            //
            // From the representation: x1_chan[0][i] = ri,  x1_chan[2..4][i] = (dx,dy,dz)
            // These are stored in atom_chan for atom i (centre), neighbour slot i.
            // d(ri)/dR[nbr_atom_i, mu] = +dx[i][mu] / ri
            // d(ri)/dR[centre_atom, mu] = -dx[i][mu] / ri
            //
            // We don't have direct access to those here; they're embedded in dksi1.
            // But we can express: d(ri)/dR[alpha,mu] in terms of dksi1:
            //   If ksi[i] = f_cut(ri)/ri^p, then dksi1[i*na3+alpha*3+mu] = dksi_dr * d(ri)/dR
            //   where dksi_dr = (fc'(ri) - fc(ri)*p/ri) / ri^p
            //
            // However dksi_dr might be 0 (if ri is at plateau of cutoff), while d(ri)/dR != 0.
            // To be safe, we recompute d(ri)/dR from dksi by dividing out — but that's unstable.
            //
            // Better: pass d(ri)/dR separately. We add a new array dri_dR alongside dksi.
            // (See revised calling convention in kernel_gaussian_gradient below.)
            //
            // For now, the gradient of the Gaussian radial factor via dri_dR is handled
            // by the caller who passes it in; we use a separate parameter here.
            // Actually let's just recompute it inline from x1_chan geometry:
            //
            // x1_chan layout: channel 0 = distances, 2 = dx, 3 = dy, 4 = dz
            // So x1_chan[0*max_size1 + i] = ri, x1_chan[2*max_size1+i] = dxk, etc.

            // We need access to the displacement for neighbour slot i of atom 1.
            // The displacement is stored in x1_chan[2..4][i].
            // We'll compute d(ri)/dR inline:
            const double dxi = x1_chan[2 * max_size1 + i];
            const double dyi = x1_chan[3 * max_size1 + i];
            const double dzi = x1_chan[4 * max_size1 + i];

            // d(ri)/dR: we need to know which atom in A corresponds to neighbour slot i.
            // This information is NOT stored here; dksi1 already carries the correct
            // gradient pattern. We compute d(ri)/dR as:
            //   d(ri) = (dxi/ri) * d(R[i_atom]) - (dxi/ri) * d(R[centre])
            //
            // But we don't know i_atom index here. We must get it from the caller.
            // Solution: the caller will pass a dri_dR array (same layout as dksi),
            // or alternatively, we accept d(ri)/dR packed alongside dksi.
            //
            // To keep the interface clean, we'll rearrange: pass dri_dR as a separate
            // vector of size n1 * na3, pre-computed alongside dksi.
            // For now we leave this as a stub and fix the calling convention below.
            //
            // ACTUAL IMPLEMENTATION: We pre-compute dri_dR for each neighbour slot alongside
            // dksi in compute_ksi_and_grad. We'll pass it separately below.
            (void)dxi;
            (void)dyi;
            (void)dzi;  // will be used via dri1 below

            // Accumulate gradient from dW/dR (via d(ksi1[i])/dR and d(angular)/dR)
            for (int alpha = 0; alpha < n_atoms_A; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const std::size_t base = static_cast<std::size_t>(i) * na3 + alpha * 3 + mu;
                    // d(W_ksi)/dR = d(ksi1[i])/dR * ksi2[j]
                    const double dW_ksi = dksi1[base] * ksi2[j];

                    // d(angular)/dR (even Fourier modes are identically zero)
                    double d_angular = 0.0;
                    for (int m = 0; m < order; m += 2) {
                        double dam = 0.0;
                        for (int p = 0; p < pmax; ++p) {
                            const std::size_t fi = cos1_idx(p, m, i);
                            dam += dcos1[fi * na3 + alpha * 3 + mu] * cos2[cos2_idx(p, m, j)] +
                                   dsin1[fi * na3 + alpha * 3 + mu] * sin2[cos2_idx(p, m, j)];
                        }
                        d_angular += dam * s_prefactor[m];
                    }

                    const double dW = dW_ksi * distance_scale + d_angular * angular_scale;
                    // Note: dG/dR term is missing here — it comes from dri1 passed separately.
                    ds_dR[alpha * 3 + mu] += G * dW;
                }
            }
        }
    }

    return aadist;
}

// Overload that also accepts dri1 (d(ri)/dR for each neighbour slot i of atom 1)
// and adds the dG/dR contribution.
// dri1 layout: [neigh_slot * na3 + alpha*3 + mu]  (same as dksi1)
static double scalar_noalchemy_and_grad_full(
    const double *x1_chan, int max_size1, int n1, const double *ksi1, const double *dksi1,
    const double *dri1, const std::vector<double> &cos1, const std::vector<double> &sin1,
    const std::vector<double> &dcos1, const std::vector<double> &dsin1, const double *x2_chan,
    int max_size2, int n2, const double *ksi2, const std::vector<double> &cos2,
    const std::vector<double> &sin2, double d_width, int order, int pmax, const double *s_prefactor,
    double distance_scale, double angular_scale, int n_atoms_A, std::vector<double> &ds_dR
) {
    // Early exit: central atoms must have the same nuclear charge (matches Fortran
    // scalar_noalchemy). Caller may also fast-path this case.
    const int Z1 = static_cast<int>(x1_chan[1 * max_size1 + 0]);
    const int Z2 = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    if (Z1 != Z2) {
        ds_dR.assign(static_cast<std::size_t>(n_atoms_A) * 3, 0.0);
        return 0.0;
    }

    const std::size_t na3 = static_cast<std::size_t>(n_atoms_A) * 3;
    ds_dR.assign(na3, 0.0);

    const double inv_width = -1.0 / (4.0 * d_width * d_width);
    const double maxgausdist2 = (8.0 * d_width) * (8.0 * d_width);

    double aadist = 1.0;

    // Layout helpers (order and max_size are invariant in the hot loops).
    const std::size_t ord = static_cast<std::size_t>(order);
    const std::size_t ms1 = static_cast<std::size_t>(max_size1);
    const std::size_t ms2 = static_cast<std::size_t>(max_size2);

    for (int i = 1; i < n1; ++i) {
        const int Zi = static_cast<int>(x1_chan[1 * max_size1 + i]);
        const double ri = x1_chan[i];
        const std::size_t i_na3 = static_cast<std::size_t>(i) * na3;

        for (int j = 1; j < n2; ++j) {
            const int Zj = static_cast<int>(x2_chan[1 * max_size2 + j]);
            if (Zi != Zj) continue;

            const double rj = x2_chan[j];
            const double dr = ri - rj;
            const double r2 = dr * dr;
            if (r2 >= maxgausdist2) continue;

            const double G = std::exp(r2 * inv_width);
            const double dG_dri = G * 2.0 * dr * inv_width;

            // Angular scalar. order==2 reduces to a single active mode (m=0, n=1);
            // even n are identically zero. order==1 is the same math.
            double angular = 0.0;
            if (order == 1 || order == 2) {
                for (int p = 0; p < pmax; ++p) {
                    const std::size_t base1 = static_cast<std::size_t>(p) * ord * ms1 + i;
                    const std::size_t base2 = static_cast<std::size_t>(p) * ord * ms2 + j;
                    angular += (cos1[base1] * cos2[base2] + sin1[base1] * sin2[base2]) * s_prefactor[0];
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    double ang_m = 0.0;
                    for (int p = 0; p < pmax; ++p) {
                        const std::size_t i1 =
                            static_cast<std::size_t>(p) * ord * ms1 + static_cast<std::size_t>(m) * ms1 +
                            static_cast<std::size_t>(i);
                        const std::size_t i2 =
                            static_cast<std::size_t>(p) * ord * ms2 + static_cast<std::size_t>(m) * ms2 +
                            static_cast<std::size_t>(j);
                        ang_m += cos1[i1] * cos2[i2] + sin1[i1] * sin2[i2];
                    }
                    angular += ang_m * s_prefactor[m];
                }
            }

            const double W_ksi = ksi1[i] * ksi2[j];
            const double W = W_ksi * distance_scale + angular * angular_scale;
            aadist += G * W;

            const double G_dist = G * distance_scale;
            const double G_ang = G * angular_scale;
            const double ksi2_j = ksi2[j];

            // Gradient: loop (p,m) outer so cos2/sin2 load once; amu inner.
            // Two-body (ksi + distance) contribution.
#ifdef _OPENMP
#pragma omp simd
#endif
            for (std::size_t amu = 0; amu < na3; ++amu) {
                const double dG_dR = dG_dri * dri1[i_na3 + amu];
                const double dW_ksi = dksi1[i_na3 + amu] * ksi2_j;
                ds_dR[amu] += dG_dR * W + G_dist * dW_ksi;
            }

            // Angular gradient contribution (active Fourier modes only).
            if (order == 1 || order == 2) {
                const double sp0 = s_prefactor[0];
                for (int p = 0; p < pmax; ++p) {
                    const std::size_t base1 = static_cast<std::size_t>(p) * ord * ms1 + i;
                    const std::size_t base2 = static_cast<std::size_t>(p) * ord * ms2 + j;
                    const double c2_0 = cos2[base2];
                    const double s2_0 = sin2[base2];
                    const std::size_t fi0 = base1 * na3;
#ifdef _OPENMP
#pragma omp simd
#endif
                    for (std::size_t amu = 0; amu < na3; ++amu) {
                        ds_dR[amu] +=
                            G_ang * sp0 * (dcos1[fi0 + amu] * c2_0 + dsin1[fi0 + amu] * s2_0);
                    }
                }
            } else {
                for (int m = 0; m < order; m += 2) {
                    const double spf = s_prefactor[m];
                    for (int p = 0; p < pmax; ++p) {
                        const std::size_t i1 =
                            static_cast<std::size_t>(p) * ord * ms1 + static_cast<std::size_t>(m) * ms1 +
                            static_cast<std::size_t>(i);
                        const std::size_t i2 =
                            static_cast<std::size_t>(p) * ord * ms2 + static_cast<std::size_t>(m) * ms2 +
                            static_cast<std::size_t>(j);
                        const double c2 = cos2[i2];
                        const double s2 = sin2[i2];
                        const std::size_t fi = i1 * na3;
                        for (std::size_t amu = 0; amu < na3; ++amu) {
                            ds_dR[amu] +=
                                G_ang * spf * (dcos1[fi + amu] * c2 + dsin1[fi + amu] * s2);
                        }
                    }
                }
            }
        }
    }

    return aadist;
}

// =============================================================================
// Extended ksi computation that also returns d(ri)/dR separately.
// Adds dri1 alongside dksi in the AtomDataGrad.
// We extend AtomDataGrad to hold dri.
// Since modifying the struct is inconvenient at this stage, we compute dri
// as an auxiliary in kernel_gaussian_gradient directly from the geometry.
// =============================================================================

// =============================================================================
// Shared B-side cache for gradient / jacobian (hoisted across many A molecules).
// =============================================================================
struct GradBSide {
    std::vector<int> z_to_idx;
    int pmax = 0;
    std::vector<AtomData> adB;
    std::vector<double> ss_B;
    std::vector<double> s_prefactor;
    double true_distance_scale = 0.0;
    double true_angular_scale = 0.0;
    double inv_sigma2 = 0.0;
};

static GradBSide build_grad_b_side(
    const std::vector<double> &x2, const std::vector<int> &n2, const std::vector<int> &nn2,
    int nm2, int max_size2, std::vector<int> z_to_idx, int pmax, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm
) {
    GradBSide B;
    B.z_to_idx = std::move(z_to_idx);
    B.pmax = pmax;
    B.true_distance_scale = two_body_scaling / 16.0;
    B.true_angular_scale = three_body_scaling / std::sqrt(8.0);
    B.inv_sigma2 = 0.5 / (sigma * sigma);

    const double ang_norm2 = get_angular_norm2(three_body_width);
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    B.s_prefactor.resize(static_cast<std::size_t>(fourier_order));
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        B.s_prefactor[static_cast<std::size_t>(m)] =
            g1 * std::exp(-(three_body_width * mf) * (three_body_width * mf) / 2.0);
    }

    if (pmax == 0) return B;

    B.adB = precompute_atom_data(
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
        two_body_power,
        cut_start,
        cut_distance,
        three_body_power,
        fourier_order,
        pmax,
        B.z_to_idx,
        use_atm
    );
    B.ss_B = compute_self_scalars(
        x2,
        n2,
        B.adB,
        nm2,
        max_size2,
        fourier_order,
        pmax,
        three_body_width,
        two_body_width,
        ang_norm2,
        B.true_distance_scale,
        B.true_angular_scale
    );
    return B;
}

// Element map from nuclear-charge lists (A) + B representation neighbour scan.
// Neighbours are atoms of the same molecule, so A Z-lists cover all A-side elements.
static int build_element_map_zlists_and_repr(
    const std::vector<std::vector<int>> &z_lists, const std::vector<double> &x2,
    const std::vector<int> &n2, const std::vector<int> &nn2, int nm2, int max_size2,
    std::vector<int> &z_to_idx
) {
    z_to_idx.assign(256, -1);
    std::vector<int> present;

    auto add_z = [&](int z) {
        if (z > 0 && z < 256 && z_to_idx[static_cast<std::size_t>(z)] < 0) {
            z_to_idx[static_cast<std::size_t>(z)] = static_cast<int>(present.size());
            present.push_back(z);
        }
    };

    for (const auto &zl : z_lists)
        for (int z : zl)
            add_z(z);

    const std::size_t mol_s = static_cast<std::size_t>(max_size2) * 5 * max_size2;
    const std::size_t atom_s = static_cast<std::size_t>(5) * max_size2;
    for (int a = 0; a < nm2; ++a) {
        for (int i = 0; i < n2[static_cast<std::size_t>(a)]; ++i) {
            const double *chan = x2.data() + static_cast<std::size_t>(a) * mol_s +
                                 static_cast<std::size_t>(i) * atom_s;
            const int n_neigh = nn2[static_cast<std::size_t>(a) * max_size2 + i];
            for (int k = 0; k < n_neigh; ++k)
                add_z(static_cast<int>(chan[max_size2 + k]));
        }
    }
    return static_cast<int>(present.size());
}

// Core: one query molecule A against precomputed B.
// parallel_over_B: enable OpenMP over training molecules (disable when outer A-parallel).
static void gradient_one_A_against_B(
    const std::vector<double> &coords_A, const std::vector<int> &z_A, int n_atoms_A,
    const std::vector<double> &x2, const std::vector<int> &n2, const std::vector<int> &nn2,
    int nm2, int max_size2, const GradBSide &B, double two_body_width, double two_body_power,
    double three_body_power, double cut_start, double cut_distance, int fourier_order,
    bool use_atm, bool parallel_over_B, double *grad_out
) {
    std::memset(grad_out, 0, sizeof(double) * static_cast<std::size_t>(n_atoms_A) * 3 * nm2);
    if (B.pmax == 0) return;

    static const bool kProfile = [] {
        const char *e = std::getenv("KF_FCHL18_PROFILE");
        return e != nullptr && e[0] == '1';
    }();
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    const int max_size_A = n_atoms_A;
    const int pmax = B.pmax;
    const double true_distance_scale = B.true_distance_scale;
    const double true_angular_scale = B.true_angular_scale;
    const double inv_sigma2 = B.inv_sigma2;
    const double *s_prefactor = B.s_prefactor.data();

    std::vector<double> x_A_mol;
    std::vector<int> nn_A_mol;
    kf::fchl18::generate_fchl18(coords_A, z_A, max_size_A, cut_distance, x_A_mol, nn_A_mol);
    const auto t_repr = clock::now();

    const std::vector<int> n_A_vec = {n_atoms_A};
    const std::vector<int> &nn_A_flat = nn_A_mol;

    // Parallelise A-side atom precompute when B-parallelism owns the team.
    // Skip when already inside an outer A-parallel region (nested OpenMP off).
    // Always worth it for typical n_atoms_A >= 4; large-nm2 fork cost is small
    // vs. O(n_atoms^2) three-body grad work.
#ifdef _OPENMP
    const bool parallel_ad_atoms = parallel_over_B;
#else
    const bool parallel_ad_atoms = false;
#endif
    const std::vector<AtomDataGrad> adA = precompute_atom_data_grad(
        x_A_mol,
        n_A_vec,
        nn_A_flat,
        n_atoms_A,
        max_size_A,
        coords_A,
        z_A,
        two_body_power,
        cut_start,
        cut_distance,
        three_body_power,
        fourier_order,
        pmax,
        B.z_to_idx,
        use_atm,
        parallel_ad_atoms
    );
    const auto t_ad = clock::now();

    const int na3 = n_atoms_A * 3;
    const std::size_t atom_stride_A = static_cast<std::size_t>(5) * max_size_A;

    // dri from stored neighbour indices (no O(n^2) coordinate matching).
    std::vector<std::vector<double>> dri_A(static_cast<std::size_t>(n_atoms_A));
    for (int i = 0; i < n_atoms_A; ++i) {
        const AtomDataGrad &adi = adA[static_cast<std::size_t>(i)];
        const int n_neigh = adi.n_neigh;
        dri_A[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(n_neigh) * na3, 0.0);

        const double *atom_chan = x_A_mol.data() + static_cast<std::size_t>(i) * atom_stride_A;
        const double *dist_chan = atom_chan;
        const double *xc = atom_chan + 2 * max_size_A;
        const double *yc = atom_chan + 3 * max_size_A;
        const double *zc = atom_chan + 4 * max_size_A;

        for (int k = 1; k < n_neigh; ++k) {
            const double r = dist_chan[k];
            if (r < 1e-14) continue;
            const int k_atom = adi.nbr_atom_idx[static_cast<std::size_t>(k)];
            if (k_atom < 0) continue;
            const double inv_r = 1.0 / r;
            const double dxk = xc[k], dyk = yc[k], dzk = zc[k];

            const std::size_t base_k = static_cast<std::size_t>(k) * na3 + k_atom * 3;
            const std::size_t base_i = static_cast<std::size_t>(k) * na3 + i * 3;
            dri_A[static_cast<std::size_t>(i)][base_k + 0] = dxk * inv_r;
            dri_A[static_cast<std::size_t>(i)][base_k + 1] = dyk * inv_r;
            dri_A[static_cast<std::size_t>(i)][base_k + 2] = dzk * inv_r;
            dri_A[static_cast<std::size_t>(i)][base_i + 0] = -dxk * inv_r;
            dri_A[static_cast<std::size_t>(i)][base_i + 1] = -dyk * inv_r;
            dri_A[static_cast<std::size_t>(i)][base_i + 2] = -dzk * inv_r;
        }
    }
    const auto t_dri = clock::now();

    std::vector<double> ss_A(static_cast<std::size_t>(n_atoms_A), 0.0);
    std::vector<std::vector<double>> dss_A(static_cast<std::size_t>(n_atoms_A));

    for (int i = 0; i < n_atoms_A; ++i) {
        const AtomDataGrad &adi = adA[static_cast<std::size_t>(i)];
        const double *x1_chan = x_A_mol.data() + static_cast<std::size_t>(i) * atom_stride_A;

        std::vector<double> ds_tmp;
        ss_A[static_cast<std::size_t>(i)] = scalar_noalchemy_and_grad_full(
            x1_chan,
            max_size_A,
            adi.n_neigh,
            adi.ksi.data(),
            adi.dksi.data(),
            dri_A[static_cast<std::size_t>(i)].data(),
            adi.cosp,
            adi.sinp,
            adi.dcosp,
            adi.dsinp,
            x1_chan,
            max_size_A,
            adi.n_neigh,
            adi.ksi.data(),
            adi.cosp,
            adi.sinp,
            two_body_width,
            fourier_order,
            pmax,
            s_prefactor,
            true_distance_scale,
            true_angular_scale,
            n_atoms_A,
            ds_tmp
        );

        dss_A[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_atoms_A) * 3);
        for (std::size_t q = 0; q < dss_A[static_cast<std::size_t>(i)].size(); ++q)
            dss_A[static_cast<std::size_t>(i)][q] = 2.0 * ds_tmp[q];
    }
    const auto t_ss = clock::now();

    const std::size_t mol2_stride = static_cast<std::size_t>(max_size2) * 5 * max_size2;
    const std::size_t at2_stride = static_cast<std::size_t>(5) * max_size2;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (parallel_over_B)
#endif
    for (int b = 0; b < nm2; ++b) {
        const int nb = n2[static_cast<std::size_t>(b)];
        std::vector<double> dsij_dR(static_cast<std::size_t>(n_atoms_A) * 3);

        for (int i = 0; i < n_atoms_A; ++i) {
            const AtomDataGrad &adi = adA[static_cast<std::size_t>(i)];
            const double *x1_chan = x_A_mol.data() + static_cast<std::size_t>(i) * atom_stride_A;
            const double sii = ss_A[static_cast<std::size_t>(i)];
            const double *dsii = dss_A[static_cast<std::size_t>(i)].data();
            const int Zi = static_cast<int>(x1_chan[max_size_A]);  // channel 1, self

            for (int j = 0; j < nb; ++j) {
                const AtomData &adj = B.adB[static_cast<std::size_t>(b) * max_size2 + j];
                const double *x2_chan = x2.data() + static_cast<std::size_t>(b) * mol2_stride +
                                       static_cast<std::size_t>(j) * at2_stride;
                const double sjj = B.ss_B[static_cast<std::size_t>(b) * max_size2 + j];
                const int Zj = static_cast<int>(x2_chan[max_size2]);

                double sij = 0.0;
                if (Zi == Zj) {
                    sij = scalar_noalchemy_and_grad_full(
                        x1_chan,
                        max_size_A,
                        adi.n_neigh,
                        adi.ksi.data(),
                        adi.dksi.data(),
                        dri_A[static_cast<std::size_t>(i)].data(),
                        adi.cosp,
                        adi.sinp,
                        adi.dcosp,
                        adi.dsinp,
                        x2_chan,
                        max_size2,
                        adj.n_neigh,
                        adj.ksi.data(),
                        adj.cosp,
                        adj.sinp,
                        two_body_width,
                        fourier_order,
                        pmax,
                        s_prefactor,
                        true_distance_scale,
                        true_angular_scale,
                        n_atoms_A,
                        dsij_dR
                    );
                } else {
                    // sij = 0, dsij = 0 — only -ds_ii contributes (matches scalar early-exit).
                    std::fill(dsij_dR.begin(), dsij_dR.end(), 0.0);
                }

                const double kij = std::exp(-(sii + sjj - 2.0 * sij) * inv_sigma2);
                const double coeff = kij * inv_sigma2;

                for (int alpha = 0; alpha < n_atoms_A; ++alpha) {
                    for (int mu = 0; mu < 3; ++mu) {
                        const double dval =
                            -dsii[alpha * 3 + mu] +
                            2.0 * dsij_dR[static_cast<std::size_t>(alpha) * 3 + mu];
                        grad_out[static_cast<std::size_t>(alpha) * 3 * nm2 + mu * nm2 + b] +=
                            coeff * dval;
                    }
                }
            }
        }
    }

    if (kProfile) {
        const double t_repr_ms = std::chrono::duration<double, std::milli>(t_repr - t0).count();
        const double t_ad_ms = std::chrono::duration<double, std::milli>(t_ad - t_repr).count();
        const double t_dri_ms = std::chrono::duration<double, std::milli>(t_dri - t_ad).count();
        const double t_ss_ms = std::chrono::duration<double, std::milli>(t_ss - t_dri).count();
        const double t_main_ms = std::chrono::duration<double, std::milli>(clock::now() - t_ss).count();
        const double total = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        std::fprintf(
            stderr,
            "[fchl18-grad] na=%d nm2=%d  repr=%.2f  ad_grad=%.2f  dri=%.2f  ss_A=%.2f  "
            "main=%.2f  total=%.2f ms\n",
            n_atoms_A,
            nm2,
            t_repr_ms,
            t_ad_ms,
            t_dri_ms,
            t_ss_ms,
            t_main_ms,
            total
        );
    }
}

// =============================================================================
// Public API: kernel_gaussian_gradient
//
// Computes G[alpha, mu, b] = dK[A,b] / dR_A[alpha, mu]
// for one query molecule A (given as raw coordinates) against a training set B
// (given as pre-computed representations x2, n2, nn2).
//
// Output grad_out: (n_atoms_A, 3, nm2) row-major.
// =============================================================================
void kernel_gaussian_gradient(
    const std::vector<double> &coords_A,  // (n_atoms_A * 3) row-major
    const std::vector<int> &z_A,          // (n_atoms_A)
    const std::vector<double> &x2, const std::vector<int> &n2, const std::vector<int> &nn2,
    int n_atoms_A, int nm2, int max_size2, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *grad_out  // (n_atoms_A, 3, nm2) row-major OUT
) {
    if (!grad_out) throw std::invalid_argument("grad_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (n_atoms_A <= 0) throw std::invalid_argument("n_atoms_A must be > 0");

    std::vector<int> z_to_idx;
    const std::vector<std::vector<int>> z_lists = {z_A};
    const int pmax =
        build_element_map_zlists_and_repr(z_lists, x2, n2, nn2, nm2, max_size2, z_to_idx);

    GradBSide B = build_grad_b_side(
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
        std::move(z_to_idx),
        pmax,
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

    gradient_one_A_against_B(
        coords_A,
        z_A,
        n_atoms_A,
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
        B,
        two_body_width,
        two_body_power,
        three_body_power,
        cut_start,
        cut_distance,
        fourier_order,
        use_atm,
        /*parallel_over_B=*/true,
        grad_out
    );
}

// =============================================================================
// Public API: kernel_gaussian_jacobian
//
// Batch form of kernel_gaussian_gradient over many query molecules A.
// Hoists B-side precompute (element map, AtomData, self-scalars) once.
// =============================================================================
enum class JacStoreLayout { RowsAreCoords, ColsAreCoords };

static void kernel_gaussian_jacobian_impl(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list, const std::vector<double> &x2,
    const std::vector<int> &n2, const std::vector<int> &nn2, int nm2, int max_size2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm, JacStoreLayout layout, double *out
) {
    if (!out) throw std::invalid_argument("jacobian output is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    const int N_A = static_cast<int>(coords_A_list.size());
    if (N_A == 0) throw std::invalid_argument("kernel_gaussian_jacobian: empty query set");
    if (static_cast<int>(z_A_list.size()) != N_A)
        throw std::invalid_argument("coords_A_list and z_A_list size mismatch");

    std::vector<int> atom_offset(static_cast<std::size_t>(N_A) + 1, 0);
    for (int i = 0; i < N_A; ++i) {
        const int na = static_cast<int>(z_A_list[static_cast<std::size_t>(i)].size());
        if (na <= 0) throw std::invalid_argument("n_atoms_A must be > 0");
        if (static_cast<int>(coords_A_list[static_cast<std::size_t>(i)].size()) != na * 3)
            throw std::invalid_argument("coords_A length must be n_atoms*3");
        atom_offset[static_cast<std::size_t>(i) + 1] =
            atom_offset[static_cast<std::size_t>(i)] + na * 3;
    }
    const int D_A = atom_offset[static_cast<std::size_t>(N_A)];
    const std::size_t out_n =
        (layout == JacStoreLayout::RowsAreCoords)
            ? static_cast<std::size_t>(D_A) * nm2
            : static_cast<std::size_t>(nm2) * D_A;
    std::memset(out, 0, sizeof(double) * out_n);

    std::vector<int> z_to_idx;
    const int pmax =
        build_element_map_zlists_and_repr(z_A_list, x2, n2, nn2, nm2, max_size2, z_to_idx);

    GradBSide B = build_grad_b_side(
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
        std::move(z_to_idx),
        pmax,
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

#ifdef _OPENMP
    const bool parallel_over_A = (N_A >= omp_get_max_threads());
#else
    const bool parallel_over_A = false;
#endif
    const bool parallel_over_B = !parallel_over_A;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) if (parallel_over_A)
#endif
    for (int i = 0; i < N_A; ++i) {
        const int na_i = static_cast<int>(z_A_list[static_cast<std::size_t>(i)].size());
        std::vector<double> grad(static_cast<std::size_t>(na_i) * 3 * nm2, 0.0);

        gradient_one_A_against_B(
            coords_A_list[static_cast<std::size_t>(i)],
            z_A_list[static_cast<std::size_t>(i)],
            na_i,
            x2,
            n2,
            nn2,
            nm2,
            max_size2,
            B,
            two_body_width,
            two_body_power,
            three_body_power,
            cut_start,
            cut_distance,
            fourier_order,
            use_atm,
            parallel_over_B,
            grad.data()
        );

        const int flat0 = atom_offset[static_cast<std::size_t>(i)];
        if (layout == JacStoreLayout::RowsAreCoords) {
            // J[flat, b]
            for (int alpha = 0; alpha < na_i; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const int row = flat0 + alpha * 3 + mu;
                    for (int b = 0; b < nm2; ++b) {
                        out[static_cast<std::size_t>(row) * nm2 + b] =
                            grad[static_cast<std::size_t>(alpha) * 3 * nm2 + mu * nm2 + b];
                    }
                }
            }
        } else {
            // K_jt[b, flat]
            for (int alpha = 0; alpha < na_i; ++alpha) {
                for (int mu = 0; mu < 3; ++mu) {
                    const int col = flat0 + alpha * 3 + mu;
                    for (int b = 0; b < nm2; ++b) {
                        out[static_cast<std::size_t>(b) * D_A + col] =
                            grad[static_cast<std::size_t>(alpha) * 3 * nm2 + mu * nm2 + b];
                    }
                }
            }
        }
    }
}

void kernel_gaussian_jacobian(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list, const std::vector<double> &x2,
    const std::vector<int> &n2, const std::vector<int> &nn2, int nm2, int max_size2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm, double *J_out
) {
    kernel_gaussian_jacobian_impl(
        coords_A_list,
        z_A_list,
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
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
        use_atm,
        JacStoreLayout::RowsAreCoords,
        J_out
    );
}

void kernel_gaussian_jacobian_t(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list, const std::vector<double> &x2,
    const std::vector<int> &n2, const std::vector<int> &nn2, int nm2, int max_size2, double sigma,
    double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm, double *K_jt_out
) {
    kernel_gaussian_jacobian_impl(
        coords_A_list,
        z_A_list,
        x2,
        n2,
        nn2,
        nm2,
        max_size2,
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
        use_atm,
        JacStoreLayout::ColsAreCoords,
        K_jt_out
    );
}

// =============================================================================
// Cross-Hessian of the FCHL18 scalar product s(i,j) w.r.t. R_A and R_B.
//
// Computes H[α*3+μ, β*3+ν] = d²s(i,j) / dR_A[α,μ] dR_B[β,ν]
//
// Restriction: cut_function == 1 everywhere (cut_distance=1e6 or cut_start>=1)
// and use_atm == false.  Under these conditions V(p,q) = ksi_A[p]*ksi_B[q]*d_scale
// + angular_A[p]*angular_B[q]*a_scale, so d²V/dR_A dR_B = 0 and only three
// terms survive (see plan):
//
//   Term 1: dG/dr_p * d(r_p)/dR_A * dV/dR_B
//   Term 2: dG/dr_q * d(r_q)/dR_B * dV/dR_A
//   Term 3: d²G/dr_p dr_q * d(r_p)/dR_A * d(r_q)/dR_B * V
//
// where G = exp(γ*(r_p-r_q)²), γ = -1/(4*w²).
//
// Inputs:
//   x1_chan : (5, max_size1) atom-channel block for atom i (A-side)
//   x2_chan : (5, max_size2) atom-channel block for atom j (B-side)
//   adi     : AtomDataGrad for atom i (ksi, cosp, sinp, dksi, dcosp, dsinp)
//   adj     : AtomDataGrad for atom j (same)
//   dri_i   : d(r_ik)/dR_A[α,μ]  layout [k*na3_A + α*3 + μ]
//   dri_j   : d(r_jq)/dR_B[β,ν]  layout [q*na3_B + β*3 + ν]
//   hess    : output (n_atoms_A*3, n_atoms_B*3), accumulated (not zeroed here)
// =============================================================================
static void scalar_noalchemy_cross_hessian(
    const double *x1_chan, int max_size1, int n1, const double *x2_chan, int max_size2, int n2,
    const AtomDataGrad &adi, const double *dri_i, const AtomDataGrad &adj, const double *dri_j,
    double d_width, int order, int pmax, const double *s_prefactor, double distance_scale,
    double angular_scale, int n_atoms_A, int n_atoms_B,
    std::vector<double> &hess  // (n_atoms_A*3, n_atoms_B*3) accumulated
) {
    // Early exit: central atoms must have the same nuclear charge (matches Fortran
    // scalar_noalchemy)
    const int Z1 = static_cast<int>(x1_chan[1 * max_size1 + 0]);
    const int Z2 = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    if (Z1 != Z2) return;

    const double inv_width = -1.0 / (4.0 * d_width * d_width);  // = γ
    const double maxgausdist2 = (8.0 * d_width) * (8.0 * d_width);

    const int na3A = n_atoms_A * 3;
    const int na3B = n_atoms_B * 3;

    auto cos1_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size1 +
               static_cast<std::size_t>(m) * max_size1 + neigh;
    };
    auto cos2_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size2 +
               static_cast<std::size_t>(m) * max_size2 + neigh;
    };

    // Reused across (p,q) neighbour pairs.
    std::vector<double> dV_dRA(static_cast<std::size_t>(na3A));
    std::vector<double> dV_dRB(static_cast<std::size_t>(na3B));

    for (int p = 1; p < n1; ++p) {
        const int Zp = static_cast<int>(x1_chan[1 * max_size1 + p]);
        const double rp = x1_chan[p];  // channel 0

        for (int q = 1; q < n2; ++q) {
            const int Zq = static_cast<int>(x2_chan[1 * max_size2 + q]);
            if (Zp != Zq) continue;

            const double rq = x2_chan[q];
            const double dr = rp - rq;
            const double r2 = dr * dr;
            if (r2 >= maxgausdist2) continue;

            const double G = std::exp(r2 * inv_width);
            const double dG_drp = G * 2.0 * dr * inv_width;   //  G * 2γδ
            const double dG_drq = -G * 2.0 * dr * inv_width;  // -G * 2γδ
            // d²G / drp drq = G * 2γ * (2γδ² - 1)
            const double d2G_drpq = G * 2.0 * inv_width * (-2.0 * inv_width * r2 - 1.0);

            // Angular contribution (even Fourier modes identically zero).
            double angular = 0.0;
            for (int m = 0; m < order; m += 2) {
                double ang_m = 0.0;
                for (int pp = 0; pp < pmax; ++pp) {
                    ang_m += adi.cosp[cos1_idx(pp, m, p)] * adj.cosp[cos2_idx(pp, m, q)] +
                             adi.sinp[cos1_idx(pp, m, p)] * adj.sinp[cos2_idx(pp, m, q)];
                }
                angular += ang_m * s_prefactor[m];
            }

            const double W_ksi = adi.ksi[p] * adj.ksi[q];
            const double V = W_ksi * distance_scale + angular * angular_scale;

            // ---- Precompute dV/dR_A[α,μ] for this (p,q) pair ----
            for (int amu = 0; amu < na3A; ++amu) {
                const std::size_t base_ksi = static_cast<std::size_t>(p) * na3A + amu;
                double dang_A = 0.0;
                for (int m = 0; m < order; m += 2) {
                    double dam = 0.0;
                    for (int pp = 0; pp < pmax; ++pp) {
                        const std::size_t fi = cos1_idx(pp, m, p);
                        dam += adi.dcosp[fi * na3A + amu] * adj.cosp[cos2_idx(pp, m, q)] +
                               adi.dsinp[fi * na3A + amu] * adj.sinp[cos2_idx(pp, m, q)];
                    }
                    dang_A += dam * s_prefactor[m];
                }
                dV_dRA[static_cast<std::size_t>(amu)] =
                    adi.dksi[base_ksi] * adj.ksi[q] * distance_scale + dang_A * angular_scale;
            }

            // ---- Precompute dV/dR_B[β,ν] for this (p,q) pair ----
            for (int bnu = 0; bnu < na3B; ++bnu) {
                const std::size_t base_ksi = static_cast<std::size_t>(q) * na3B + bnu;
                double dang_B = 0.0;
                for (int m = 0; m < order; m += 2) {
                    double dam = 0.0;
                    for (int pp = 0; pp < pmax; ++pp) {
                        const std::size_t fi = cos2_idx(pp, m, q);
                        dam += adj.dcosp[fi * na3B + bnu] * adi.cosp[cos1_idx(pp, m, p)] +
                               adj.dsinp[fi * na3B + bnu] * adi.sinp[cos1_idx(pp, m, p)];
                    }
                    dang_B += dam * s_prefactor[m];
                }
                dV_dRB[static_cast<std::size_t>(bnu)] =
                    adj.dksi[base_ksi] * adi.ksi[p] * distance_scale + dang_B * angular_scale;
            }

            // ---- Accumulate the four terms into hess[amu, bnu] ----
            for (int amu = 0; amu < na3A; ++amu) {
                const double drp_dRA = dri_i[static_cast<std::size_t>(p) * na3A + amu];
                const double dG_dRA = dG_drp * drp_dRA;
                const double dksi_A_amu = adi.dksi[static_cast<std::size_t>(p) * na3A + amu];

                for (int bnu = 0; bnu < na3B; ++bnu) {
                    const double drq_dRB = dri_j[static_cast<std::size_t>(q) * na3B + bnu];
                    const double dG_dRB = dG_drq * drq_dRB;

                    double val = dG_dRA * dV_dRB[static_cast<std::size_t>(bnu)];
                    val += dG_dRB * dV_dRA[static_cast<std::size_t>(amu)];
                    val += d2G_drpq * drp_dRA * drq_dRB * V;

                    const double dksi_B_bnu = adj.dksi[static_cast<std::size_t>(q) * na3B + bnu];
                    double d2V = dksi_A_amu * dksi_B_bnu * distance_scale;
                    for (int m = 0; m < order; m += 2) {
                        double dam4 = 0.0;
                        for (int pp = 0; pp < pmax; ++pp) {
                            const std::size_t fiA = cos1_idx(pp, m, p);
                            const std::size_t fiB = cos2_idx(pp, m, q);
                            dam4 += adi.dcosp[fiA * na3A + amu] * adj.dcosp[fiB * na3B + bnu] +
                                    adi.dsinp[fiA * na3A + amu] * adj.dsinp[fiB * na3B + bnu];
                        }
                        d2V += dam4 * s_prefactor[m] * angular_scale;
                    }
                    val += G * d2V;

                    hess[static_cast<std::size_t>(amu) * na3B + bnu] += val;
                }
            }
        }
    }
}

// =============================================================================
// Hessian helpers: per-molecule cache + pair kernel (hoisted across many pairs).
// =============================================================================
struct HessMolSide {
    int n_atoms = 0;
    int max_size = 0;
    std::vector<double> x_mol;
    std::vector<int> nn_mol;
    std::vector<AtomDataGrad> ad;
    std::vector<std::vector<double>> dri;
    std::vector<double> ss;
    std::vector<std::vector<double>> dss;
};

struct HessSharedParams {
    std::vector<int> z_to_idx;
    int pmax = 0;
    std::vector<double> s_prefactor;
    double true_distance_scale = 0.0;
    double true_angular_scale = 0.0;
    double inv_sigma2 = 0.0;
    double two_body_width = 0.0;
    int fourier_order = 0;
};

static void build_dri_from_ad(
    const std::vector<AtomDataGrad> &ad, const std::vector<double> &x_mol, int n_atoms,
    int max_size, std::vector<std::vector<double>> &dri_out
) {
    const int na3 = n_atoms * 3;
    const std::size_t atom_stride = static_cast<std::size_t>(5) * max_size;
    dri_out.resize(static_cast<std::size_t>(n_atoms));
    for (int i = 0; i < n_atoms; ++i) {
        const AtomDataGrad &adi = ad[static_cast<std::size_t>(i)];
        const int n_neigh = adi.n_neigh;
        dri_out[static_cast<std::size_t>(i)].assign(static_cast<std::size_t>(n_neigh) * na3, 0.0);

        const double *atom_chan = x_mol.data() + static_cast<std::size_t>(i) * atom_stride;
        const double *dist_chan = atom_chan;
        const double *xc = atom_chan + 2 * max_size;
        const double *yc = atom_chan + 3 * max_size;
        const double *zc = atom_chan + 4 * max_size;

        for (int k = 1; k < n_neigh; ++k) {
            const double r = dist_chan[k];
            if (r < 1e-14) continue;
            const int k_atom = adi.nbr_atom_idx[static_cast<std::size_t>(k)];
            if (k_atom < 0) continue;
            const double inv_r = 1.0 / r;
            const double dxk = xc[k], dyk = yc[k], dzk = zc[k];

            const std::size_t base_k = static_cast<std::size_t>(k) * na3 + k_atom * 3;
            const std::size_t base_i = static_cast<std::size_t>(k) * na3 + i * 3;
            auto &dri = dri_out[static_cast<std::size_t>(i)];
            dri[base_k + 0] = dxk * inv_r;
            dri[base_k + 1] = dyk * inv_r;
            dri[base_k + 2] = dzk * inv_r;
            dri[base_i + 0] = -dxk * inv_r;
            dri[base_i + 1] = -dyk * inv_r;
            dri[base_i + 2] = -dzk * inv_r;
        }
    }
}

static HessMolSide build_hess_mol_side(
    const std::vector<double> &coords, const std::vector<int> &z, int n_atoms,
    const HessSharedParams &P, double two_body_power, double three_body_power, double cut_start,
    double cut_distance, bool use_atm, bool parallel_atoms
) {
    HessMolSide S;
    S.n_atoms = n_atoms;
    S.max_size = n_atoms;
    kf::fchl18::generate_fchl18(coords, z, S.max_size, cut_distance, S.x_mol, S.nn_mol);

    const std::vector<int> n_vec = {n_atoms};
    S.ad = precompute_atom_data_grad(
        S.x_mol,
        n_vec,
        S.nn_mol,
        n_atoms,
        S.max_size,
        coords,
        z,
        two_body_power,
        cut_start,
        cut_distance,
        three_body_power,
        P.fourier_order,
        P.pmax,
        P.z_to_idx,
        use_atm,
        parallel_atoms
    );

    build_dri_from_ad(S.ad, S.x_mol, n_atoms, S.max_size, S.dri);

    const int na3 = n_atoms * 3;
    const std::size_t atom_stride = static_cast<std::size_t>(5) * S.max_size;
    S.ss.assign(static_cast<std::size_t>(n_atoms), 0.0);
    S.dss.resize(static_cast<std::size_t>(n_atoms));

    for (int i = 0; i < n_atoms; ++i) {
        const AtomDataGrad &adi = S.ad[static_cast<std::size_t>(i)];
        const double *x1_chan = S.x_mol.data() + static_cast<std::size_t>(i) * atom_stride;
        std::vector<double> ds_tmp;
        S.ss[static_cast<std::size_t>(i)] = scalar_noalchemy_and_grad_full(
            x1_chan,
            S.max_size,
            adi.n_neigh,
            adi.ksi.data(),
            adi.dksi.data(),
            S.dri[static_cast<std::size_t>(i)].data(),
            adi.cosp,
            adi.sinp,
            adi.dcosp,
            adi.dsinp,
            x1_chan,
            S.max_size,
            adi.n_neigh,
            adi.ksi.data(),
            adi.cosp,
            adi.sinp,
            P.two_body_width,
            P.fourier_order,
            P.pmax,
            P.s_prefactor.data(),
            P.true_distance_scale,
            P.true_angular_scale,
            n_atoms,
            ds_tmp
        );
        S.dss[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(na3));
        for (int q = 0; q < na3; ++q)
            S.dss[static_cast<std::size_t>(i)][static_cast<std::size_t>(q)] = 2.0 * ds_tmp[static_cast<std::size_t>(q)];
    }
    return S;
}

static HessSharedParams build_hess_shared(
    const std::vector<std::vector<int>> &z_lists, double sigma, double two_body_scaling,
    double three_body_scaling, double three_body_width, double two_body_width, int fourier_order
) {
    HessSharedParams P;
    P.true_distance_scale = two_body_scaling / 16.0;
    P.true_angular_scale = three_body_scaling / std::sqrt(8.0);
    P.inv_sigma2 = 0.5 / (sigma * sigma);
    P.two_body_width = two_body_width;
    P.fourier_order = fourier_order;

    // Element map from nuclear charges only (neighbours are same-molecule atoms).
    P.z_to_idx.assign(256, -1);
    std::vector<int> present;
    for (const auto &zl : z_lists) {
        for (int z : zl) {
            if (z > 0 && z < 256 && P.z_to_idx[static_cast<std::size_t>(z)] < 0) {
                P.z_to_idx[static_cast<std::size_t>(z)] = static_cast<int>(present.size());
                present.push_back(z);
            }
        }
    }
    P.pmax = static_cast<int>(present.size());

    const double ang_norm2 = get_angular_norm2(three_body_width);
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    P.s_prefactor.resize(static_cast<std::size_t>(fourier_order));
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        P.s_prefactor[static_cast<std::size_t>(m)] =
            g1 * std::exp(-(three_body_width * mf) * (three_body_width * mf) / 2.0);
    }
    return P;
}

// Core pair kernel using precomputed molecule sides.
static void hessian_pair_from_sides(
    const HessMolSide &A, const HessMolSide &B, const HessSharedParams &P, double *hess_out
) {
    const int na3A = A.n_atoms * 3;
    const int na3B = B.n_atoms * 3;
    std::memset(hess_out, 0, sizeof(double) * static_cast<std::size_t>(na3A) * na3B);
    if (P.pmax == 0) return;

    const std::size_t atom_stride_A = static_cast<std::size_t>(5) * A.max_size;
    const std::size_t atom_stride_B = static_cast<std::size_t>(5) * B.max_size;
    const std::size_t hess_size = static_cast<std::size_t>(na3A) * na3B;

    std::vector<double> dsij_dRA(static_cast<std::size_t>(na3A));
    std::vector<double> dsij_dRB(static_cast<std::size_t>(na3B));
    std::vector<double> Hij(hess_size);

    for (int i = 0; i < A.n_atoms; ++i) {
        const AtomDataGrad &adi = A.ad[static_cast<std::size_t>(i)];
        const double *x1_chan = A.x_mol.data() + static_cast<std::size_t>(i) * atom_stride_A;
        const int Zi = static_cast<int>(x1_chan[A.max_size]);

        for (int j = 0; j < B.n_atoms; ++j) {
            const AtomDataGrad &adj = B.ad[static_cast<std::size_t>(j)];
            const double *x2_chan = B.x_mol.data() + static_cast<std::size_t>(j) * atom_stride_B;
            const int Zj = static_cast<int>(x2_chan[B.max_size]);

            double sij = 0.0;
            if (Zi == Zj) {
                sij = scalar_noalchemy_and_grad_full(
                    x1_chan,
                    A.max_size,
                    adi.n_neigh,
                    adi.ksi.data(),
                    adi.dksi.data(),
                    A.dri[static_cast<std::size_t>(i)].data(),
                    adi.cosp,
                    adi.sinp,
                    adi.dcosp,
                    adi.dsinp,
                    x2_chan,
                    B.max_size,
                    adj.n_neigh,
                    adj.ksi.data(),
                    adj.cosp,
                    adj.sinp,
                    P.two_body_width,
                    P.fourier_order,
                    P.pmax,
                    P.s_prefactor.data(),
                    P.true_distance_scale,
                    P.true_angular_scale,
                    A.n_atoms,
                    dsij_dRA
                );
                scalar_noalchemy_and_grad_full(
                    x2_chan,
                    B.max_size,
                    adj.n_neigh,
                    adj.ksi.data(),
                    adj.dksi.data(),
                    B.dri[static_cast<std::size_t>(j)].data(),
                    adj.cosp,
                    adj.sinp,
                    adj.dcosp,
                    adj.dsinp,
                    x1_chan,
                    A.max_size,
                    adi.n_neigh,
                    adi.ksi.data(),
                    adi.cosp,
                    adi.sinp,
                    P.two_body_width,
                    P.fourier_order,
                    P.pmax,
                    P.s_prefactor.data(),
                    P.true_distance_scale,
                    P.true_angular_scale,
                    B.n_atoms,
                    dsij_dRB
                );
                std::fill(Hij.begin(), Hij.end(), 0.0);
                scalar_noalchemy_cross_hessian(
                    x1_chan,
                    A.max_size,
                    adi.n_neigh,
                    x2_chan,
                    B.max_size,
                    adj.n_neigh,
                    adi,
                    A.dri[static_cast<std::size_t>(i)].data(),
                    adj,
                    B.dri[static_cast<std::size_t>(j)].data(),
                    P.two_body_width,
                    P.fourier_order,
                    P.pmax,
                    P.s_prefactor.data(),
                    P.true_distance_scale,
                    P.true_angular_scale,
                    A.n_atoms,
                    B.n_atoms,
                    Hij
                );
            } else {
                std::fill(dsij_dRA.begin(), dsij_dRA.end(), 0.0);
                std::fill(dsij_dRB.begin(), dsij_dRB.end(), 0.0);
                std::fill(Hij.begin(), Hij.end(), 0.0);
            }

            const double kij =
                std::exp(-(A.ss[static_cast<std::size_t>(i)] + B.ss[static_cast<std::size_t>(j)] -
                           2.0 * sij) *
                         P.inv_sigma2);
            const double coeff = kij * P.inv_sigma2;

            for (int amu = 0; amu < na3A; ++amu) {
                const double gA =
                    -A.dss[static_cast<std::size_t>(i)][static_cast<std::size_t>(amu)] +
                    2.0 * dsij_dRA[static_cast<std::size_t>(amu)];
                for (int bnu = 0; bnu < na3B; ++bnu) {
                    const double gB =
                        -B.dss[static_cast<std::size_t>(j)][static_cast<std::size_t>(bnu)] +
                        2.0 * dsij_dRB[static_cast<std::size_t>(bnu)];
                    hess_out[static_cast<std::size_t>(amu) * na3B + bnu] +=
                        coeff * (P.inv_sigma2 * gA * gB +
                                 2.0 * Hij[static_cast<std::size_t>(amu) * na3B + bnu]);
                }
            }
        }
    }
}

// =============================================================================
// Public API: kernel_gaussian_hessian
// =============================================================================
void kernel_gaussian_hessian(
    const std::vector<double> &coords_A, const std::vector<int> &z_A,
    const std::vector<double> &coords_B, const std::vector<int> &z_B, int n_atoms_A, int n_atoms_B,
    double sigma, double two_body_scaling, double two_body_width, double two_body_power,
    double three_body_scaling, double three_body_width, double three_body_power, double cut_start,
    double cut_distance, int fourier_order, bool use_atm,
    double *hess_out  // (n_atoms_A*3, n_atoms_B*3) row-major OUT
) {
    if (!hess_out) throw std::invalid_argument("hess_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    if (n_atoms_A <= 0 || n_atoms_B <= 0)
        throw std::invalid_argument("n_atoms must be > 0");

    const std::vector<std::vector<int>> z_lists = {z_A, z_B};
    HessSharedParams P = build_hess_shared(
        z_lists, sigma, two_body_scaling, three_body_scaling, three_body_width, two_body_width,
        fourier_order
    );

    // Single-pair call: parallelise atom precompute on each side.
    HessMolSide sideA = build_hess_mol_side(
        coords_A, z_A, n_atoms_A, P, two_body_power, three_body_power, cut_start, cut_distance,
        use_atm, /*parallel_atoms=*/true
    );
    HessMolSide sideB = build_hess_mol_side(
        coords_B, z_B, n_atoms_B, P, two_body_power, three_body_power, cut_start, cut_distance,
        use_atm, /*parallel_atoms=*/true
    );
    hessian_pair_from_sides(sideA, sideB, P, hess_out);
}

// =============================================================================
// Public API: kernel_gaussian_hessian_rect
//
// Batch Hessian over many A × B molecule pairs.
// H_out layout (D_A, D_B) with molecule blocks:
//   rows [atom_offset_A[i], atom_offset_A[i+1]) × cols [atom_offset_B[j], ...)
// Hoists per-molecule AtomDataGrad / self-scalars once.
// =============================================================================
void kernel_gaussian_hessian_rect(
    const std::vector<std::vector<double>> &coords_A_list,
    const std::vector<std::vector<int>> &z_A_list,
    const std::vector<std::vector<double>> &coords_B_list,
    const std::vector<std::vector<int>> &z_B_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *H_out  // (D_A, D_B) row-major OUT
) {
    if (!H_out) throw std::invalid_argument("H_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    const int N_A = static_cast<int>(coords_A_list.size());
    const int N_B = static_cast<int>(coords_B_list.size());
    if (N_A == 0 || N_B == 0) throw std::invalid_argument("hessian_rect: empty molecule list");
    if (static_cast<int>(z_A_list.size()) != N_A || static_cast<int>(z_B_list.size()) != N_B)
        throw std::invalid_argument("coords/z list size mismatch");

    std::vector<int> off_A(static_cast<std::size_t>(N_A) + 1, 0);
    std::vector<int> off_B(static_cast<std::size_t>(N_B) + 1, 0);
    for (int i = 0; i < N_A; ++i) {
        const int na = static_cast<int>(z_A_list[static_cast<std::size_t>(i)].size());
        if (na <= 0) throw std::invalid_argument("n_atoms must be > 0");
        off_A[static_cast<std::size_t>(i) + 1] = off_A[static_cast<std::size_t>(i)] + na * 3;
    }
    for (int j = 0; j < N_B; ++j) {
        const int nb = static_cast<int>(z_B_list[static_cast<std::size_t>(j)].size());
        if (nb <= 0) throw std::invalid_argument("n_atoms must be > 0");
        off_B[static_cast<std::size_t>(j) + 1] = off_B[static_cast<std::size_t>(j)] + nb * 3;
    }
    const int D_A = off_A[static_cast<std::size_t>(N_A)];
    const int D_B = off_B[static_cast<std::size_t>(N_B)];
    std::memset(H_out, 0, sizeof(double) * static_cast<std::size_t>(D_A) * D_B);

    std::vector<std::vector<int>> z_all = z_A_list;
    z_all.insert(z_all.end(), z_B_list.begin(), z_B_list.end());
    HessSharedParams P = build_hess_shared(
        z_all, sigma, two_body_scaling, three_body_scaling, three_body_width, two_body_width,
        fourier_order
    );
    if (P.pmax == 0) return;

    std::vector<HessMolSide> sides_A(static_cast<std::size_t>(N_A));
    std::vector<HessMolSide> sides_B(static_cast<std::size_t>(N_B));

    // Precompute all molecules once (parallel over molecules).
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < N_A + N_B; ++i) {
        if (i < N_A) {
            sides_A[static_cast<std::size_t>(i)] = build_hess_mol_side(
                coords_A_list[static_cast<std::size_t>(i)],
                z_A_list[static_cast<std::size_t>(i)],
                static_cast<int>(z_A_list[static_cast<std::size_t>(i)].size()),
                P,
                two_body_power,
                three_body_power,
                cut_start,
                cut_distance,
                use_atm,
                /*parallel_atoms=*/false
            );
        } else {
            const int j = i - N_A;
            sides_B[static_cast<std::size_t>(j)] = build_hess_mol_side(
                coords_B_list[static_cast<std::size_t>(j)],
                z_B_list[static_cast<std::size_t>(j)],
                static_cast<int>(z_B_list[static_cast<std::size_t>(j)].size()),
                P,
                two_body_power,
                three_body_power,
                cut_start,
                cut_distance,
                use_atm,
                /*parallel_atoms=*/false
            );
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int i = 0; i < N_A; ++i) {
        for (int j = 0; j < N_B; ++j) {
            const int na3A = sides_A[static_cast<std::size_t>(i)].n_atoms * 3;
            const int na3B = sides_B[static_cast<std::size_t>(j)].n_atoms * 3;
            std::vector<double> block(static_cast<std::size_t>(na3A) * na3B, 0.0);
            hessian_pair_from_sides(
                sides_A[static_cast<std::size_t>(i)],
                sides_B[static_cast<std::size_t>(j)],
                P,
                block.data()
            );
            const int r0 = off_A[static_cast<std::size_t>(i)];
            const int c0 = off_B[static_cast<std::size_t>(j)];
            for (int amu = 0; amu < na3A; ++amu) {
                for (int bnu = 0; bnu < na3B; ++bnu) {
                    H_out[static_cast<std::size_t>(r0 + amu) * D_B + (c0 + bnu)] =
                        block[static_cast<std::size_t>(amu) * na3B + bnu];
                }
            }
        }
    }
}

// =============================================================================
// Public API: kernel_gaussian_hessian_symm_blocks
//
// Symmetric batch: fills (D, D) using lower-triangle pairs + mirror.
// Diagonal blocks are symmetrised.
// =============================================================================
void kernel_gaussian_hessian_symm_blocks(
    const std::vector<std::vector<double>> &coords_list,
    const std::vector<std::vector<int>> &z_list, double sigma, double two_body_scaling,
    double two_body_width, double two_body_power, double three_body_scaling,
    double three_body_width, double three_body_power, double cut_start, double cut_distance,
    int fourier_order, bool use_atm,
    double *H_out  // (D, D) row-major OUT
) {
    if (!H_out) throw std::invalid_argument("H_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");
    const int nm = static_cast<int>(coords_list.size());
    if (nm == 0) throw std::invalid_argument("hessian_symm_blocks: empty molecule list");
    if (static_cast<int>(z_list.size()) != nm)
        throw std::invalid_argument("coords/z list size mismatch");

    std::vector<int> offset(static_cast<std::size_t>(nm) + 1, 0);
    for (int a = 0; a < nm; ++a) {
        const int na = static_cast<int>(z_list[static_cast<std::size_t>(a)].size());
        if (na <= 0) throw std::invalid_argument("n_atoms must be > 0");
        offset[static_cast<std::size_t>(a) + 1] = offset[static_cast<std::size_t>(a)] + na * 3;
    }
    const int D = offset[static_cast<std::size_t>(nm)];
    std::memset(H_out, 0, sizeof(double) * static_cast<std::size_t>(D) * D);

    HessSharedParams P = build_hess_shared(
        z_list, sigma, two_body_scaling, three_body_scaling, three_body_width, two_body_width,
        fourier_order
    );
    if (P.pmax == 0) return;

    std::vector<HessMolSide> sides(static_cast<std::size_t>(nm));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int a = 0; a < nm; ++a) {
        sides[static_cast<std::size_t>(a)] = build_hess_mol_side(
            coords_list[static_cast<std::size_t>(a)],
            z_list[static_cast<std::size_t>(a)],
            static_cast<int>(z_list[static_cast<std::size_t>(a)].size()),
            P,
            two_body_power,
            three_body_power,
            cut_start,
            cut_distance,
            use_atm,
            /*parallel_atoms=*/false
        );
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int a = 0; a < nm; ++a) {
        for (int b = 0; b < nm; ++b) {
            if (b > a) continue;

            const int na3A = sides[static_cast<std::size_t>(a)].n_atoms * 3;
            const int na3B = sides[static_cast<std::size_t>(b)].n_atoms * 3;
            std::vector<double> block(static_cast<std::size_t>(na3A) * na3B, 0.0);
            hessian_pair_from_sides(
                sides[static_cast<std::size_t>(a)],
                sides[static_cast<std::size_t>(b)],
                P,
                block.data()
            );

            const int r0 = offset[static_cast<std::size_t>(a)];
            const int c0 = offset[static_cast<std::size_t>(b)];

            if (a == b) {
                for (int amu = 0; amu < na3A; ++amu) {
                    for (int bnu = 0; bnu < na3B; ++bnu) {
                        double v;
                        if (amu == bnu) {
                            v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                        } else {
                            v = 0.5 * (block[static_cast<std::size_t>(amu) * na3B + bnu] +
                                       block[static_cast<std::size_t>(bnu) * na3B + amu]);
                        }
                        H_out[static_cast<std::size_t>(r0 + amu) * D + (c0 + bnu)] = v;
                        H_out[static_cast<std::size_t>(c0 + bnu) * D + (r0 + amu)] = v;
                    }
                }
            } else {
                for (int amu = 0; amu < na3A; ++amu) {
                    for (int bnu = 0; bnu < na3B; ++bnu) {
                        const double v = block[static_cast<std::size_t>(amu) * na3B + bnu];
                        H_out[static_cast<std::size_t>(r0 + amu) * D + (c0 + bnu)] = v;
                        H_out[static_cast<std::size_t>(c0 + bnu) * D + (r0 + amu)] = v;
                    }
                }
            }
        }
    }
}

}  // namespace fchl18
}  // namespace kf
