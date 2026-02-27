// Own header
#include "fchl18_kernel.hpp"

// C++ standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
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
    for (; n > 0; --n) r *= x;
    return r;
}

// Generalised power: uses ipow when exponent is a small integer, std::pow otherwise.
static inline double fast_pow(double x, double p) {
    const int ip = static_cast<int>(p);
    if (static_cast<double>(ip) == p && ip >= 0 && ip <= 16)
        return ipow(x, ip);
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
    const int    limit = 10000;
    double       ang_norm2 = 0.0;

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
    const double *atom_chan,   // (5, max_size) slice for one atom
    int max_size, int n_neigh,
    double power, double cut_start, double cut_distance,
    std::vector<double> &ksi   // size n_neigh, output
) {
    ksi.assign(n_neigh, 0.0);
    for (int k = 1; k < n_neigh; ++k) {   // k=0 is self, skip
        const double r = atom_chan[k];     // channel 0
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
// Output: cosp/sinp with layout (pmax, order, max_size) flat.
static void compute_threebody_fourier(
    const double *atom_chan,   // (5, max_size) slice for one atom
    int max_size, int n_neigh,
    double three_body_power, double cut_start, double cut_distance,
    int order, int pmax,
    const std::vector<int> &z_to_idx,  // Z -> compact index (size 256)
    std::vector<double> &cosp, // (pmax, order, max_size) flat
    std::vector<double> &sinp  // (pmax, order, max_size) flat
) {
    const double pi = 4.0 * std::atan(1.0);
    const std::size_t sz = static_cast<std::size_t>(pmax) * order * max_size;
    cosp.assign(sz, 0.0);
    sinp.assign(sz, 0.0);

    auto idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size
             + static_cast<std::size_t>(m) * max_size
             + neigh;
    };

    // Channel pointers
    const double *dist_chan = atom_chan;                    // channel 0: distances
    const double *z_chan    = atom_chan + max_size;          // channel 1: nuclear charges
    const double *xc        = atom_chan + 2 * max_size;     // channel 2: dx
    const double *yc        = atom_chan + 3 * max_size;     // channel 3: dy
    const double *zc        = atom_chan + 4 * max_size;     // channel 4: dz

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
        const double dj = dist_chan[j];   // dist(centre, j)
        if (dj < 1e-14) continue;
        const double cutj = cut_k[j];
        if (cutj == 0.0) continue;

        for (int k = j + 1; k < n_neigh; ++k) {
            const double dk = dist_chan[k];   // dist(centre, k)
            if (dk < 1e-14) continue;
            const double cutk = cut_k[k];
            if (cutk == 0.0) continue;

            // Distance between neighbours j and k
            const double dxjk = xc[j] - xc[k];
            const double dyjk = yc[j] - yc[k];
            const double dzjk = zc[j] - zc[k];
            const double di2  = dxjk*dxjk + dyjk*dyjk + dzjk*dzjk;
            const double di   = std::sqrt(di2);
            if (di < 1e-14) continue;

            const double cut_jk = cut_function(di, cut_start, cut_distance);
            if (cut_jk == 0.0) continue;

            // All three cosines use precomputed unit vectors — no sqrt needed
            // cos_i = angle at centre between j and k: dot(unit_j, unit_k)
            const double cos_i = clamp11(ux[j]*ux[k] + uy[j]*uy[k] + uz[j]*uz[k]);

            // cos_j = angle at j between k and centre: dot(unit_k_from_j, unit_centre_from_j)
            // unit_k_from_j = (xc[k]-xc[j], ...) / di
            const double inv_di = 1.0 / di;
            const double ukj_x = -dxjk * inv_di, ukj_y = -dyjk * inv_di, ukj_z = -dzjk * inv_di;
            // unit_centre_from_j = -unit_j (centre is at origin, j is at xc[j])
            const double cos_j = clamp11(ukj_x * (-ux[j]) + ukj_y * (-uy[j]) + ukj_z * (-uz[j]));

            // cos_k = angle at k between j and centre: dot(unit_j_from_k, unit_centre_from_k)
            const double ujk_x =  dxjk * inv_di, ujk_y =  dyjk * inv_di, ujk_z =  dzjk * inv_di;
            const double cos_k = clamp11(ujk_x * (-ux[k]) + ujk_y * (-uy[k]) + ujk_z * (-uz[k]));

            const double atm = 1.0 + 3.0 * cos_i * cos_j * cos_k;
            if (atm == 0.0) continue;

            const double dijk  = di * dj * dk;
            const double denom = fast_pow(dijk, three_body_power);
            const double cut   = cutj * cutk * cut_jk;
            const double ksi3  = cut * atm / denom;
            if (ksi3 == 0.0) continue;

            // theta = angle at centre (same as cos_i, just one acos)
            const double theta = std::acos(cos_i);

            const int zk = static_cast<int>(z_chan[k]);
            const int zj = static_cast<int>(z_chan[j]);
            if (zk <= 0 || zk >= 256 || zj <= 0 || zj >= 256) continue;
            const int pj = z_to_idx[zk];  // compact index for Z of k-th neighbour
            const int pk = z_to_idx[zj];  // compact index for Z of j-th neighbour
            if (pj < 0 || pk < 0) continue;

            for (int m = 0; m < order; ++m) {
                const double mf    = static_cast<double>(m + 1);
                const double mth   = mf * theta;
                const double mthpi = mf * (theta + pi);
                const double cos_m = (std::cos(mth) - std::cos(mthpi)) * ksi3;
                const double sin_m = (std::sin(mth) - std::sin(mthpi)) * ksi3;

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
    const double *x1_chan, int max_size1, int n1,
    const double *ksi1,
    const std::vector<double> &cos1, // (pmax, order, max_size1)
    const std::vector<double> &sin1,
    // Atom j data: (5, max_size_2) channel-major
    const double *x2_chan, int max_size2, int n2,
    const double *ksi2,
    const std::vector<double> &cos2, // (pmax, order, max_size2)
    const std::vector<double> &sin2,
    // Kernel hyperparameters
    double d_width, int order, int pmax,
    const double *s_prefactor,  // (order) precomputed Fourier prefactors
    double distance_scale, double angular_scale
) {
    // Early exit: central atoms must have the same nuclear charge
    const int Z1 = static_cast<int>(x1_chan[1 * max_size1 + 0]); // channel 1, neighbour 0 = self
    const int Z2 = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    if (Z1 != Z2) return 0.0;

    const double inv_width    = -1.0 / (4.0 * d_width * d_width);
    const double maxgausdist2 = (8.0 * d_width) * (8.0 * d_width);

    // Scalar product starts at 1 (self-contribution)
    double aadist = 1.0;

    // Sum over neighbour pairs (i from atom1, j from atom2) with same Z
    auto cos1_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size1
             + static_cast<std::size_t>(m) * max_size1
             + neigh;
    };
    auto cos2_idx = [&](int p, int m, int neigh) -> std::size_t {
        return static_cast<std::size_t>(p) * order * max_size2
             + static_cast<std::size_t>(m) * max_size2
             + neigh;
    };

    for (int i = 1; i < n1; ++i) {
        const int Zi = static_cast<int>(x1_chan[1 * max_size1 + i]);
        const double ri = x1_chan[i]; // channel 0, neighbour i

        for (int j = 1; j < n2; ++j) {
            const int Zj = static_cast<int>(x2_chan[1 * max_size2 + j]);
            if (Zi != Zj) continue;

            const double rj = x2_chan[j]; // channel 0
            const double r2 = (rj - ri) * (rj - ri);
            if (r2 >= maxgausdist2) continue;

            const double d = std::exp(r2 * inv_width);

            // Angular contribution: sum over element types p and Fourier orders m
            double angular = 0.0;
            for (int m = 0; m < order; ++m) {
                double ang_m = 0.0;
                // We sum over element types p that exist in both atoms' neighbourhoods.
                // In the no-alchemy path elements must match (same p used for both sides).
                for (int p = 0; p < pmax; ++p) {
                    ang_m += cos1[cos1_idx(p, m, i)] * cos2[cos2_idx(p, m, j)]
                           + sin1[cos1_idx(p, m, i)] * sin2[cos2_idx(p, m, j)];
                }
                angular += ang_m * s_prefactor[m];
            }

            aadist += d * (ksi1[i] * ksi2[j] * distance_scale
                          + angular * angular_scale);
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
    const std::vector<int>    &n,  // (nm) atom counts
    const std::vector<int>    &nn, // (nm * max_size) neighbour counts
    int nm, int max_size,
    double two_body_power, double cut_start, double cut_distance,
    double three_body_power, int order, int pmax,
    const std::vector<int>    &z_to_idx  // compact element map (size 256)
) {
    std::vector<AtomData> data(static_cast<std::size_t>(nm) * max_size);

    // Build a flat list of (mol, atom) pairs for all real atoms so that OpenMP
    // gets one task per atom instead of one task per molecule — much better load
    // balance when molecules have different sizes.
    std::vector<std::pair<int,int>> atom_list;
    atom_list.reserve(static_cast<std::size_t>(nm) * max_size);
    for (int a = 0; a < nm; ++a)
        for (int i = 0; i < n[a]; ++i)
            atom_list.emplace_back(a, i);
    const int total_atoms = static_cast<int>(atom_list.size());

    const std::size_t mol_stride  = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t atom_stride = static_cast<std::size_t>(5) * max_size;

#pragma omp parallel for schedule(dynamic, 4)
    for (int t = 0; t < total_atoms; ++t) {
        const int a = atom_list[t].first;
        const int i = atom_list[t].second;

        AtomData &ad = data[static_cast<std::size_t>(a) * max_size + i];
        const int n_neigh = nn[static_cast<std::size_t>(a) * max_size + i];
        ad.n_neigh = n_neigh;

        const double *atom_chan = x.data()
            + static_cast<std::size_t>(a) * mol_stride
            + static_cast<std::size_t>(i) * atom_stride;

        compute_ksi(atom_chan, max_size, n_neigh,
                    two_body_power, cut_start, cut_distance,
                    ad.ksi);

        compute_threebody_fourier(atom_chan, max_size, n_neigh,
                                  three_body_power, cut_start, cut_distance,
                                  order, pmax, z_to_idx,
                                  ad.cosp, ad.sinp);
    }
    return data;
}

// =============================================================================
// Compute self-scalar for every real atom in a set.
// self_scalar[a * max_size + i] = scalar(atom_i from mol_a, atom_i from mol_a)
// =============================================================================
static std::vector<double> compute_self_scalars(
    const std::vector<double> &x,  // (nm, max_size, 5, max_size)
    const std::vector<int>    &n,  // (nm)
    const std::vector<AtomData> &ad,
    int nm, int max_size, int order, int pmax,
    double t_width, double d_width, double ang_norm2,
    double distance_scale, double angular_scale
) {
    const std::size_t mol_stride  = static_cast<std::size_t>(max_size) * 5 * max_size;
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
    std::vector<std::pair<int,int>> atom_list;
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
        const double *atom_chan = x.data()
            + static_cast<std::size_t>(a) * mol_stride
            + static_cast<std::size_t>(i) * atom_stride;

        ss[static_cast<std::size_t>(a) * max_size + i] = scalar_noalchemy(
            atom_chan, max_size, adi.n_neigh,
            adi.ksi.data(),
            adi.cosp, adi.sinp,
            atom_chan, max_size, adi.n_neigh,
            adi.ksi.data(),
            adi.cosp, adi.sinp,
            d_width, order, pmax, s_prefactor.data(),
            distance_scale, angular_scale
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
    const std::vector<double> &x1, const std::vector<int> &n1,
    const std::vector<int>    &nn1, int nm1, int max_size1,
    const std::vector<double> &x2, const std::vector<int> &n2,
    const std::vector<int>    &nn2, int nm2, int max_size2,
    std::vector<int>          &z_to_idx  // size 256, output
) {
    z_to_idx.assign(256, -1);
    std::vector<int> present;  // Z values seen so far

    auto scan = [&](const std::vector<double> &x, const std::vector<int> &nv,
                    const std::vector<int> &nn, int nm, int ms) {
        const std::size_t mol_s  = static_cast<std::size_t>(ms) * 5 * ms;
        const std::size_t atom_s = static_cast<std::size_t>(5) * ms;
        for (int a = 0; a < nm; ++a) {
            for (int i = 0; i < nv[a]; ++i) {
                const double *chan = x.data() + a * mol_s + i * atom_s;
                const int n_neigh  = nn[static_cast<std::size_t>(a) * ms + i];
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
    const std::vector<double> &x1,
    const std::vector<double> &x2,
    const std::vector<int>    &n1,
    const std::vector<int>    &n2,
    const std::vector<int>    &nn1,
    const std::vector<int>    &nn2,
    int nm1, int nm2, int max_size1, int max_size2,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int    fourier_order,
    double *kernel_out
) {
    if (!kernel_out) throw std::invalid_argument("kernel_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

    // Apply the Fortran convention scaling correction
    const double true_distance_scale = two_body_scaling  / 16.0;
    const double true_angular_scale  = three_body_scaling / std::sqrt(8.0);

    const double ang_norm2 = get_angular_norm2(three_body_width);

    // Build compact element map: only distinct Z values seen in real neighbour slots.
    // This reduces pmax from max(Z) to the number of distinct elements (e.g. 4 for H/C/N/O).
    std::vector<int> z_to_idx;
    const int pmax = build_element_map(x1, n1, nn1, nm1, max_size1,
                                       x2, n2, nn2, nm2, max_size2, z_to_idx);

    if (pmax == 0) {
        std::memset(kernel_out, 0, sizeof(double) * nm1 * nm2);
        return;
    }

    // Precompute per-atom data
    auto ad1 = precompute_atom_data(x1, n1, nn1, nm1, max_size1,
                                    two_body_power, cut_start, cut_distance,
                                    three_body_power, fourier_order, pmax, z_to_idx);
    auto ad2 = precompute_atom_data(x2, n2, nn2, nm2, max_size2,
                                    two_body_power, cut_start, cut_distance,
                                    three_body_power, fourier_order, pmax, z_to_idx);

    // Self-scalars
    auto ss1 = compute_self_scalars(x1, n1, ad1, nm1, max_size1,
                                    fourier_order, pmax,
                                    three_body_width, two_body_width, ang_norm2,
                                    true_distance_scale, true_angular_scale);
    auto ss2 = compute_self_scalars(x2, n2, ad2, nm2, max_size2,
                                    fourier_order, pmax,
                                    three_body_width, two_body_width, ang_norm2,
                                    true_distance_scale, true_angular_scale);

    // Zero output
    std::memset(kernel_out, 0, sizeof(double) * nm1 * nm2);

    const double inv_sigma2 = -1.0 / (sigma * sigma);

    // Precompute Fourier prefactors once for the entire kernel call
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    std::vector<double> s_prefactor(fourier_order);
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        s_prefactor[m] = g1 * std::exp(-(three_body_width * mf) * (three_body_width * mf) / 2.0);
    }

    const std::size_t mol1 = static_cast<std::size_t>(max_size1) * 5 * max_size1;
    const std::size_t at1  = static_cast<std::size_t>(5) * max_size1;
    const std::size_t mol2 = static_cast<std::size_t>(max_size2) * 5 * max_size2;
    const std::size_t at2  = static_cast<std::size_t>(5) * max_size2;

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
                const int Zi = static_cast<int>(x1_chan[max_size1]); // Z of centre atom i

                const double sii = ss1[static_cast<std::size_t>(a) * max_size1 + i];

                for (int j = 0; j < nb; ++j) {
                    const AtomData &adj = ad2[static_cast<std::size_t>(b) * max_size2 + j];
                    const double *x2_chan = x2.data() + b * mol2 + j * at2;
                    const int Zj = static_cast<int>(x2_chan[max_size2]); // Z of centre atom j
                    if (Zi != Zj) continue;

                    const double sjj = ss2[static_cast<std::size_t>(b) * max_size2 + j];

                    const double s12 = scalar_noalchemy(
                        x1_chan, max_size1, adi.n_neigh,
                        adi.ksi.data(), adi.cosp, adi.sinp,
                        x2_chan, max_size2, adj.n_neigh,
                        adj.ksi.data(), adj.cosp, adj.sinp,
                        two_body_width, fourier_order, pmax, s_prefactor.data(),
                        true_distance_scale, true_angular_scale
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
    const std::vector<double> &x,
    const std::vector<int>    &n,
    const std::vector<int>    &nn,
    int nm, int max_size,
    double sigma,
    double two_body_scaling,
    double two_body_width,
    double two_body_power,
    double three_body_scaling,
    double three_body_width,
    double three_body_power,
    double cut_start,
    double cut_distance,
    int    fourier_order,
    double *kernel_out
) {
    if (!kernel_out) throw std::invalid_argument("kernel_out is null");
    if (sigma <= 0.0) throw std::invalid_argument("sigma must be > 0");

    const double true_distance_scale = two_body_scaling  / 16.0;
    const double true_angular_scale  = three_body_scaling / std::sqrt(8.0);

    const double ang_norm2 = get_angular_norm2(three_body_width);

    std::vector<int> z_to_idx;
    const int pmax = build_element_map(x, n, nn, nm, max_size,
                                       x, n, nn, nm, max_size, z_to_idx);

    if (pmax == 0) {
        std::memset(kernel_out, 0, sizeof(double) * nm * nm);
        return;
    }

    auto ad = precompute_atom_data(x, n, nn, nm, max_size,
                                   two_body_power, cut_start, cut_distance,
                                   three_body_power, fourier_order, pmax, z_to_idx);

    auto ss = compute_self_scalars(x, n, ad, nm, max_size,
                                   fourier_order, pmax,
                                   three_body_width, two_body_width, ang_norm2,
                                   true_distance_scale, true_angular_scale);

    // Zero output
    std::memset(kernel_out, 0, sizeof(double) * nm * nm);

    const double inv_sigma2 = -1.0 / (sigma * sigma);

    // Precompute Fourier prefactors once for the entire kernel call
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    std::vector<double> s_prefactor(fourier_order);
    for (int m = 0; m < fourier_order; ++m) {
        const double mf = static_cast<double>(m + 1);
        s_prefactor[m] = g1 * std::exp(-(three_body_width * mf) * (three_body_width * mf) / 2.0);
    }

    const std::size_t mol_s = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t at_s  = static_cast<std::size_t>(5) * max_size;

    // Build flat list of all upper-triangle (a, b) pairs for load-balanced parallelism.
    // Previously parallelised over 'a' only: the last thread gets work proportional to
    // just 1 pair while the first gets nm pairs — terrible balance for large nm.
    std::vector<std::pair<int,int>> pairs;
    pairs.reserve(static_cast<std::size_t>(nm) * (nm + 1) / 2);
    for (int a = 0; a < nm; ++a)
        for (int b = a; b < nm; ++b)
            pairs.emplace_back(a, b);
    const int npairs = static_cast<int>(pairs.size());

#pragma omp parallel for schedule(dynamic, 4)
    for (int p = 0; p < npairs; ++p) {
        const int a  = pairs[p].first;
        const int b  = pairs[p].second;
        const int na = n[a];
        const int nb = n[b];
        double kab = 0.0;

        for (int i = 0; i < na; ++i) {
            const AtomData &adi = ad[static_cast<std::size_t>(a) * max_size + i];
            const double *xa_chan = x.data() + a * mol_s + i * at_s;
            const int Zi = static_cast<int>(xa_chan[max_size]);

            const double sii = ss[static_cast<std::size_t>(a) * max_size + i];

            for (int j = 0; j < nb; ++j) {
                const AtomData &adj = ad[static_cast<std::size_t>(b) * max_size + j];
                const double *xb_chan = x.data() + b * mol_s + j * at_s;
                const int Zj = static_cast<int>(xb_chan[max_size]);
                if (Zi != Zj) continue;

                const double sjj = ss[static_cast<std::size_t>(b) * max_size + j];

                const double s12 = scalar_noalchemy(
                    xa_chan, max_size, adi.n_neigh,
                    adi.ksi.data(), adi.cosp, adi.sinp,
                    xb_chan, max_size, adj.n_neigh,
                    adj.ksi.data(), adj.cosp, adj.sinp,
                    two_body_width, fourier_order, pmax, s_prefactor.data(),
                    true_distance_scale, true_angular_scale
                );

                kab += std::exp((sii + sjj - 2.0 * s12) * inv_sigma2);
            }
        }

        kernel_out[static_cast<std::size_t>(a) * nm + b] = kab;
        kernel_out[static_cast<std::size_t>(b) * nm + a] = kab;
    }
}

}  // namespace fchl18
}  // namespace kf
