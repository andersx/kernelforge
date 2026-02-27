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

// Compute cosine of angle at vertex b, formed by vectors (a-b) and (c-b).
static inline double calc_cos_angle(
    const double *a, const double *b, const double *c
) {
    double v1[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    double v2[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};

    const double n1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
    const double n2 = std::sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);

    if (n1 < 1e-14 || n2 < 1e-14) return 0.0;

    v1[0] /= n1; v1[1] /= n1; v1[2] /= n1;
    v2[0] /= n2; v2[1] /= n2; v2[2] /= n2;

    double c_ang = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    if (c_ang >  1.0) c_ang =  1.0;
    if (c_ang < -1.0) c_ang = -1.0;
    return c_ang;
}

// Compute angle at vertex b.
static inline double calc_angle(
    const double *a, const double *b, const double *c
) {
    return std::acos(calc_cos_angle(a, b, c));
}

// ATM three-body weight for triplet (centre=pos[0], j-th neighbour, k-th neighbour).
// ksi3 = cut_j*cut_k*cut_jk * (1 + 3*cos_i*cos_j*cos_k) / (di*dj*dk)^power
static inline double calc_ksi3(
    // Full (5, max_size) atom data — passed as a flat pointer, channel-major.
    // pos_chan[c * max_size + idx] = data for channel c, index idx.
    const double *pos_chan, int max_size,
    int j_idx, int k_idx,
    double power, double cut_start, double cut_distance
) {
    // Cartesian positions:  centre = channel2/3/4 of index 0
    //                        j     = channel2/3/4 of index j_idx
    //                        k     = channel2/3/4 of index k_idx
    const double *centre = pos_chan + 2 * max_size;  // channels 2,3,4 start here
    const double  cx = centre[0], cy = centre[max_size], cz = centre[2 * max_size];
    const double  jx = centre[j_idx], jy = centre[max_size + j_idx], jz = centre[2*max_size + j_idx];
    const double  kx = centre[k_idx], ky = centre[max_size + k_idx], kz = centre[2*max_size + k_idx];

    const double c[3] = {cx, cy, cz};
    const double vj[3] = {jx, jy, jz};
    const double vk[3] = {kx, ky, kz};

    const double cos_i = calc_cos_angle(vk, c,  vj);   // angle at centre
    const double cos_j = calc_cos_angle(vj, vk, c );   // angle at k-atom
    const double cos_k = calc_cos_angle(c,  vj, vk);   // angle at j-atom

    // distances: dk = dist(centre, j), dj = dist(centre, k), di = dist(j, k)
    const double dk = pos_chan[j_idx];   // channel 0
    const double dj = pos_chan[k_idx];

    const double dxjk = vj[0] - vk[0];
    const double dyjk = vj[1] - vk[1];
    const double dzjk = vj[2] - vk[2];
    const double di = std::sqrt(dxjk*dxjk + dyjk*dyjk + dzjk*dzjk);

    if (di < 1e-14 || dj < 1e-14 || dk < 1e-14) return 0.0;

    const double cut = cut_function(dk, cut_start, cut_distance)
                     * cut_function(dj, cut_start, cut_distance)
                     * cut_function(di, cut_start, cut_distance);

    const double didj = di * dj;
    const double didjdk = didj * dk;
    const double denom = std::pow(didjdk, power);

    return cut * (1.0 + 3.0 * cos_i * cos_j * cos_k) / denom;
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
        ksi[k] = cut_function(r, cut_start, cut_distance) / std::pow(r, power);
    }
}

// Compute three-body Fourier terms for one atom.
//
// Output: cosp/sinp with layout (pmax, order, n_neigh) — but we only care about
// which element-type (pj/pk) and which fourier index (m) and which neighbour.
// We return flat vectors indexed as [p * order * max_size + m * max_size + neigh].
// pmax = max nuclear charge seen across the dataset (passed in).
static void compute_threebody_fourier(
    const double *atom_chan,   // (5, max_size) slice for one atom
    int max_size, int n_neigh,
    double three_body_power, double cut_start, double cut_distance,
    int order, int pmax,
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

    for (int j = 1; j < n_neigh; ++j) {       // j: first neighbour
        for (int k = j + 1; k < n_neigh; ++k) { // k: second neighbour (k>j)

            const double ksi3 = calc_ksi3(atom_chan, max_size, j, k,
                                           three_body_power, cut_start, cut_distance);
            if (ksi3 == 0.0) continue;

            // angle at the centre atom, between neighbours j and k
            const double *centre = atom_chan + 2 * max_size;
            const double c[3]  = {centre[0], centre[max_size], centre[2*max_size]};
            const double vj[3] = {centre[j], centre[max_size+j], centre[2*max_size+j]};
            const double vk[3] = {centre[k], centre[max_size+k], centre[2*max_size+k]};
            const double theta = calc_angle(vj, c, vk);

            // element indices (0-based for array, but nuclear charge is 1-based)
            const int pj = static_cast<int>(atom_chan[1 * max_size + k]) - 1;  // Z of k-th neighbour
            const int pk = static_cast<int>(atom_chan[1 * max_size + j]) - 1;  // Z of j-th neighbour
            if (pj < 0 || pj >= pmax) continue;
            if (pk < 0 || pk >= pmax) continue;

            for (int m = 0; m < order; ++m) {
                const double mf = static_cast<double>(m + 1);  // 1-indexed in Fortran
                const double cos_m = (std::cos(mf * theta) - std::cos((theta + pi) * mf)) * ksi3;
                const double sin_m = (std::sin(mf * theta) - std::sin((theta + pi) * mf)) * ksi3;

                // Fortran: fourier(1,pj,m,j) and fourier(1,pk,m,k) both get cos_m
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
    double t_width, double d_width, int order, int pmax,
    double ang_norm2,
    double distance_scale, double angular_scale
) {
    // Early exit: central atoms must have the same nuclear charge
    const int Z1 = static_cast<int>(x1_chan[1 * max_size1 + 0]); // channel 1, neighbour 0 = self
    const int Z2 = static_cast<int>(x2_chan[1 * max_size2 + 0]);
    if (Z1 != Z2) return 0.0;

    // Pre-computed constants
    const double pi = 4.0 * std::atan(1.0);
    const double g1 = std::sqrt(2.0 * pi) / ang_norm2;
    // For the no-alchemy path we only use fourier_order=1 (the original had a sum
    // over m but the non-alchemy path in the Fortran uses only m=1 cached in cos1c/cos2c).
    // We generalise to arbitrary order here.
    const double inv_width    = -1.0 / (4.0 * d_width * d_width);
    const double maxgausdist2 = (8.0 * d_width) * (8.0 * d_width);

    // Scalar product starts at 1 (self-contribution)
    double aadist = 1.0;

    const int order_use = std::min(order, 1); // noalchemy path originally uses order=1
    (void)order_use; // suppress unused warning — we use 'order' directly below

    // Fourier prefactors: s[m] = g1 * exp(-(t_width*(m+1))^2 / 2)
    // For the no-alchemy path the original Fortran computes only m=1 (fourier_order=1
    // for the cached cos1c/cos2c). We honour the full 'order' here.
    std::vector<double> s(order);
    for (int m = 0; m < order; ++m) {
        const double mf = static_cast<double>(m + 1);
        s[m] = g1 * std::exp(-(t_width * mf) * (t_width * mf) / 2.0);
    }

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
                angular += ang_m * s[m];
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
static std::vector<AtomData> precompute_atom_data(
    const std::vector<double> &x,  // (nm, max_size, 5, max_size) row-major
    const std::vector<int>    &n,  // (nm) atom counts
    const std::vector<int>    &nn, // (nm * max_size) neighbour counts
    int nm, int max_size,
    double two_body_power, double cut_start, double cut_distance,
    double three_body_power, int order, int pmax
) {
    std::vector<AtomData> data(static_cast<std::size_t>(nm) * max_size);

    // Access helper: x[(mol, atom, chan, neigh)] in (nm, max_size, 5, max_size) layout
    // = x[mol*max_size*5*max_size + atom*5*max_size + chan*max_size + neigh]
    const std::size_t mol_stride  = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t atom_stride = static_cast<std::size_t>(5) * max_size;

#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < nm; ++a) {
        const int na = n[a];
        for (int i = 0; i < na; ++i) {
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
                                      order, pmax,
                                      ad.cosp, ad.sinp);
        }
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

    std::vector<double> ss(static_cast<std::size_t>(nm) * max_size, 0.0);

#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < nm; ++a) {
        const int na = n[a];
        for (int i = 0; i < na; ++i) {
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
                t_width, d_width, order, pmax, ang_norm2,
                distance_scale, angular_scale
            );
        }
    }
    return ss;
}

// =============================================================================
// Determine pmax: maximum nuclear charge across all atoms in both sets
// =============================================================================
static int get_pmax(
    const std::vector<double> &x1, const std::vector<int> &n1, int nm1, int max_size1,
    const std::vector<double> &x2, const std::vector<int> &n2, int nm2, int max_size2
) {
    int pmax = 0;
    const std::size_t mol1 = static_cast<std::size_t>(max_size1) * 5 * max_size1;
    const std::size_t at1  = static_cast<std::size_t>(5) * max_size1;
    for (int a = 0; a < nm1; ++a) {
        for (int i = 0; i < n1[a]; ++i) {
            const double *chan = x1.data() + a * mol1 + i * at1;
            for (int k = 0; k < max_size1; ++k) {
                const int z = static_cast<int>(chan[max_size1 + k]);
                if (z > pmax) pmax = z;
            }
        }
    }
    const std::size_t mol2 = static_cast<std::size_t>(max_size2) * 5 * max_size2;
    const std::size_t at2  = static_cast<std::size_t>(5) * max_size2;
    for (int b = 0; b < nm2; ++b) {
        for (int j = 0; j < n2[b]; ++j) {
            const double *chan = x2.data() + b * mol2 + j * at2;
            for (int k = 0; k < max_size2; ++k) {
                const int z = static_cast<int>(chan[max_size2 + k]);
                if (z > pmax) pmax = z;
            }
        }
    }
    return pmax;
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
    const int    pmax      = get_pmax(x1, n1, nm1, max_size1, x2, n2, nm2, max_size2);

    if (pmax == 0) {
        std::memset(kernel_out, 0, sizeof(double) * nm1 * nm2);
        return;
    }

    // Precompute per-atom data
    auto ad1 = precompute_atom_data(x1, n1, nn1, nm1, max_size1,
                                    two_body_power, cut_start, cut_distance,
                                    three_body_power, fourier_order, pmax);
    auto ad2 = precompute_atom_data(x2, n2, nn2, nm2, max_size2,
                                    two_body_power, cut_start, cut_distance,
                                    three_body_power, fourier_order, pmax);

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
                        three_body_width, two_body_width,
                        fourier_order, pmax, ang_norm2,
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
    const int    pmax      = get_pmax(x, n, nm, max_size, x, n, nm, max_size);

    if (pmax == 0) {
        std::memset(kernel_out, 0, sizeof(double) * nm * nm);
        return;
    }

    auto ad = precompute_atom_data(x, n, nn, nm, max_size,
                                   two_body_power, cut_start, cut_distance,
                                   three_body_power, fourier_order, pmax);

    auto ss = compute_self_scalars(x, n, ad, nm, max_size,
                                   fourier_order, pmax,
                                   three_body_width, two_body_width, ang_norm2,
                                   true_distance_scale, true_angular_scale);

    // Zero output
    std::memset(kernel_out, 0, sizeof(double) * nm * nm);

    const double inv_sigma2 = -1.0 / (sigma * sigma);

    const std::size_t mol_s = static_cast<std::size_t>(max_size) * 5 * max_size;
    const std::size_t at_s  = static_cast<std::size_t>(5) * max_size;

#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < nm; ++a) {
        const int na = n[a];
        for (int b = a; b < nm; ++b) {
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
                        three_body_width, two_body_width,
                        fourier_order, pmax, ang_norm2,
                        true_distance_scale, true_angular_scale
                    );

                    kab += std::exp((sii + sjj - 2.0 * s12) * inv_sigma2);
                }
            }

            kernel_out[static_cast<std::size_t>(a) * nm + b] = kab;
            kernel_out[static_cast<std::size_t>(b) * nm + a] = kab;
        }
    }
}

}  // namespace fchl18
}  // namespace kf
