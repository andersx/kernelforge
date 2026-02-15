#include <algorithm>
#include <array>
#include <cmath>
#if defined(__APPLE__)
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <omp.h>
#include <iostream>
#include "aligned_alloc64.hpp"
#include "fchl19_representation.hpp"

namespace fchl19 {

// Compute the expected representation size so the caller doesn't have to.
std::size_t compute_rep_size(size_t nelements, size_t nbasis2, size_t nbasis3, size_t nabasis) {
    const size_t two_body = nelements * nbasis2;
    const size_t n_pairs_symmetric = nelements * (nelements + 1) / 2; // unordered pairs (p<=q)
    const size_t three_body = n_pairs_symmetric * nbasis3 * nabasis;
    return two_body + three_body;
}


// constexpr double PI = 3.141592653589793238462643383279502884;
// constexpr double SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383; // sqrt(2*pi)
// constexpr double EPS = 1e-12;

// -------------------- small 3D helpers --------------------
double dot3(const double a[3], const double b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
void sub3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}
double norm3(const double v[3]) {
    return std::sqrt(std::max(dot3(v,v), 0.0));
}
void normalize3(double v[3]) {
    double n = norm3(v);
    if (n < EPS) { v[0] = v[1] = v[2] = 0.0; return; }
    v[0] /= n; v[1] /= n; v[2] /= n;
}

double calc_angle(const double a[3], const double b[3], const double c[3]) {
    double v1[3], v2[3];
    sub3(a, b, v1);
    sub3(c, b, v2);
    normalize3(v1);
    normalize3(v2);
    double cosang = dot3(v1, v2);
    if (cosang > 1.0) cosang = 1.0;
    if (cosang < -1.0) cosang = -1.0;
    return std::acos(cosang);
}

double calc_cos_angle(const double a[3], const double b[3], const double c[3]) {
    double v1[3], v2[3];
    sub3(a, b, v1);
    sub3(c, b, v2);
    normalize3(v1);
    normalize3(v2);
    return dot3(v1, v2);
}

// --------------- indexing helpers for flat matrices ---------------
size_t idx2(size_t i, size_t j, size_t ncols) { return i * ncols + j; }

// Half-cosine cutoff-like decay used in the original Fortran: f = 0.5*(cos(pi*r*invrc)+1)
// rmat: natoms x natoms distance matrix (row-major)
void decay_matrix(const std::vector<double>& rmat, double invrc,
                         size_t natoms, std::vector<double>& out) {
    out.resize(natoms * natoms);
    const double f = PI * invrc;
    for (size_t i = 0; i < natoms * natoms; ++i) {
        out[i] = 0.5 * (std::cos(f * rmat[i]) + 1.0);
    }
}

// Compute full pairwise distance matrix (natoms x natoms, row-major)
std::vector<double> pairwise_distances(const std::vector<double>& coords, size_t natoms) {
    if (coords.size() != natoms * 3) throw std::invalid_argument("coords.size() must equal natoms*3");
    std::vector<double> D(natoms * natoms, 0.0);
    for (size_t i = 0; i < natoms; ++i) {
        const double* ri = &coords[3*i];
        for (size_t j = i + 1; j < natoms; ++j) {
            const double* rj = &coords[3*j];
            double dx = rj[0] - ri[0];
            double dy = rj[1] - ri[1];
            double dz = rj[2] - ri[2];
            double d = std::sqrt(dx*dx + dy*dy + dz*dz);
            D[idx2(i,j,natoms)] = d;
            D[idx2(j,i,natoms)] = d;
        }
    }
    return D;
}

inline size_t gidx(size_t i, size_t feat, size_t a, size_t d, size_t rep_size, size_t natoms){
    // (i,feat,a,d) -> flat index for grad of shape (natoms, rep_size, natoms, 3)
    return (((i*rep_size + feat)*natoms + a)*3 + d);
}

inline size_t pair_index(size_t nelements, int p, int q){
    if (p>q) std::swap(p,q);
    long long llp=p, llq=q, llN=static_cast<long long>(nelements);
    long long idx = -llp*(llp+1)/2 + llq + llN*llp;
    return static_cast<size_t>(idx);
}


// #########################
// # FCHL19 REPRESENTATION #
// #########################

// Main generator. This reproduces the Fortran logic closely but in 0-based indexing.
// Inputs:
//  - coords:   length natoms*3 (x,y,z per atom), row-major
//  - nuclear_z: length natoms, atomic numbers
//  - elements: unique element types used (length nelements); order defines channel layout
//  - Rs2, Rs3: radial basis centers for 2-body and 3-body, respectively
//  - Ts:       angular basis placeholder (only its size is used, must be even)
//  - eta2, eta3, zeta, rcut, acut, two_body_decay, three_body_decay, three_body_weight: as in Fortran
// Output:
//  - rep: resized to natoms * rep_size and filled
void generate_fchl_acsf(
    const std::vector<double>& coords,
    const std::vector<int>& nuclear_z,
    const std::vector<int>& elements,
    const std::vector<double>& Rs2,
    const std::vector<double>& Rs3,
    const std::vector<double>& Ts,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight,
    std::vector<double>& rep
) {
    const size_t natoms = nuclear_z.size();
    if (coords.size() != natoms * 3) throw std::invalid_argument("coords size must be natoms*3");

    const size_t nelements = elements.size();
    const size_t nbasis2 = Rs2.size();
    const size_t nbasis3 = Rs3.size();
    const size_t nabasis  = Ts.size();
    if (nabasis % 2 != 0) throw std::invalid_argument("Ts.size() (nabasis) must be even");

    const size_t rep_size = compute_rep_size(nelements, nbasis2, nbasis3, nabasis);
    rep.assign(natoms * rep_size, 0.0);

    // Map Z -> element channel index [0..nelements)
    std::unordered_map<int, int> z2idx; z2idx.reserve(nelements * 2);
    for (size_t j = 0; j < nelements; ++j) z2idx[elements[j]] = static_cast<int>(j);

    std::vector<int> elem_of_atom(natoms, -1);
    for (size_t i = 0; i < natoms; ++i) {
        auto it = z2idx.find(nuclear_z[i]);
        if (it == z2idx.end()) throw std::runtime_error("nuclear_z contains an element not present in elements");
        elem_of_atom[i] = it->second; // 0-based
    }

    // Distances and decays
    const std::vector<double> D = pairwise_distances(coords, natoms);

    std::vector<double> rdecay2, rdecay3;
    decay_matrix(D, (rcut > 0 ? 1.0/rcut : 0.0), natoms, rdecay2);
    decay_matrix(D, (acut > 0 ? 1.0/acut : 0.0), natoms, rdecay3);

    // Precompute angular weights for harmonics o = 1,3,5,... (length nabasis/2)
    const size_t n_harm = nabasis / 2;
    std::vector<double> ang_w(n_harm); // 2*exp(-0.5*(zeta*o)^2)
    std::vector<int>    ang_o(n_harm); // o values
    for (size_t l = 0; l < n_harm; ++l) {
        int o = static_cast<int>(2*l + 1); // 1,3,5,...
        ang_o[l] = o;
        double t = zeta * static_cast<double>(o);
        ang_w[l] = 2.0 * std::exp(-0.5 * t * t);
    }

    // Precompute log(Rs2) and 1/Rs2
    std::vector<double> log_Rs2(nbasis2, 0.0);
    std::vector<double> inv_Rs2(nbasis2, 0.0);
    for (size_t k = 0; k < nbasis2; ++k) {
        if (Rs2[k] <= 0.0) throw std::invalid_argument("All Rs2 must be > 0");
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
                const double rij = D[idx2(i,j,natoms)];
                if (rij > rcut) continue;

                const double rij2 = rij * rij;
                const double t = eta2 / std::max(rij2, EPS);
                const double log1pt = std::log1p(t);
                const double sigma = std::sqrt(std::max(log1pt, 0.0));
                if (sigma < EPS) continue;
                const double mu = std::log(rij) - 0.5 * log1pt;
                const double decay_ij = rdecay2[idx2(i,j,natoms)];
                const double inv_pref = decay_ij / (sigma * SQRT_2PI * std::pow(rij, two_body_decay));
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

        const double* rb = &coords[3*i];
        for (size_t j = 0; j + 1 < natoms; ++j) {
            if (j == i) continue;
            const double rij = D[idx2(i,j,natoms)];
            if (rij > acut) continue;
            const int elem_j = elem_of_atom[j];
            const double* ra = &coords[3*j];

            // Precompute normalized edge vector eij = (A-B)/rij
            const double inv_rij = 1.0 / std::max(rij, EPS);
            const double eij0 = (ra[0] - rb[0]) * inv_rij;
            const double eij1 = (ra[1] - rb[1]) * inv_rij;
            const double eij2 = (ra[2] - rb[2]) * inv_rij;

            for (size_t k = j + 1; k < natoms; ++k) {
                if (k == i) continue;
                const double rik = D[idx2(i,k,natoms)];
                if (rik > acut) continue;
                const int elem_k = elem_of_atom[k];
                const double* rc = &coords[3*k];

                // Normalized edge vector eik = (C-B)/rik
                const double inv_rik = 1.0 / std::max(rik, EPS);
                const double eik0 = (rc[0] - rb[0]) * inv_rik;
                const double eik1 = (rc[1] - rb[1]) * inv_rik;
                const double eik2 = (rc[2] - rb[2]) * inv_rik;

                // Normalized edge vector ejk = (C-A)/rjk
                const double rjk = D[idx2(j,k,natoms)];
                const double inv_rjk = 1.0 / std::max(rjk, EPS);
                const double ejk0 = (rc[0] - ra[0]) * inv_rjk;
                const double ejk1 = (rc[1] - ra[1]) * inv_rjk;
                const double ejk2 = (rc[2] - ra[2]) * inv_rjk;

                // Cosines from dot products of normalized vectors (no acos needed)
                double cos_i = eij0*eik0 + eij1*eik1 + eij2*eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0*ejk0 + eij1*ejk1 + eij2*ejk2);  // dot(-eij, ejk)
                const double cos_k = eik0*ejk0 + eik1*ejk1 + eik2*ejk2;     // dot(eik, ejk) = dot(-eik, -ejk)

                // ATM factor
                const double denom = std::pow(std::max(rik*rij*rjk, EPS), three_body_decay);
                const double ksi3 = (1.0 + 3.0 * cos_i * cos_j * cos_k) * (three_body_weight / denom);

                // Angular basis via Chebyshev recurrence (no acos/cos/sin calls)
                const double sin_i = std::sqrt(std::max(0.0, 1.0 - cos_i * cos_i));
                // o=1: cos(1*angle) = cos_i, sin(1*angle) = sin_i
                angular[0] = ang_w[0] * cos_i;
                if (nabasis > 1) angular[1] = ang_w[0] * sin_i;

                // Higher harmonics o=3,5,... via Chebyshev recurrence
                if (n_harm > 1) {
                    const double two_cos = 2.0 * cos_i;
                    double cn_2 = 1.0, sn_2 = 0.0;     // cos(0), sin(0)
                    double cn_1 = cos_i, sn_1 = sin_i;  // cos(θ), sin(θ)
                    size_t harm_stored = 1;
                    const int max_o = ang_o[n_harm - 1];
                    for (int n = 2; n <= max_o; ++n) {
                        const double cn = two_cos * cn_1 - cn_2;
                        const double sn = two_cos * sn_1 - sn_2;
                        cn_2 = cn_1; sn_2 = sn_1;
                        cn_1 = cn;   sn_1 = sn;
                        if (n == ang_o[harm_stored]) {
                            angular[2*harm_stored]     = ang_w[harm_stored] * cn;
                            angular[2*harm_stored + 1] = ang_w[harm_stored] * sn;
                            if (++harm_stored >= n_harm) break;
                        }
                    }
                }

                // Element pair index and accumulation
                const size_t pair_idx = pair_index(nelements, elem_j, elem_k);
                const size_t base = three_offset + pair_idx * (nbasis3 * nabasis);

                const double decay_ij = rdecay3[idx2(i,j,natoms)];
                const double decay_ik = rdecay3[idx2(i,k,natoms)];
                const double rbar = 0.5 * (rij + rik);

                for (size_t l = 0; l < nbasis3; ++l) {
                    const double dr = rbar - Rs3[l];
                    const double radial_l = std::exp(-eta3 * dr * dr) * decay_ij * decay_ik;
                    const double scale = radial_l * ksi3;
                    const size_t z = base + l * nabasis;
                    double* dst = &rep[idx2(i, z, rep_size)];
                    for (size_t t = 0; t < nabasis; ++t) dst[t] += angular[t] * scale;
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
    const std::vector<double>& coords,
    const std::vector<int>& nuclear_z,
    const std::vector<int>& elements,
    const std::vector<double>& Rs2,
    const std::vector<double>& Rs3,
    const std::vector<double>& Ts,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight,
    std::vector<double>& rep,
    std::vector<double>& grad
){
    const size_t natoms = nuclear_z.size();
    if (coords.size() != natoms*3) throw std::invalid_argument("coords size must be natoms*3");
    const size_t nelements = elements.size();
    const size_t nbasis2 = Rs2.size();
    const size_t nbasis3 = Rs3.size();
    const size_t nabasis  = Ts.size();
    const size_t rep_size = compute_rep_size(nelements, nbasis2, nbasis3, nabasis);

    rep.assign(natoms*rep_size, 0.0);
    grad.assign(natoms*rep_size*natoms*3, 0.0);

    // Map Z->element index
    std::unordered_map<int,int> z2idx; z2idx.reserve(nelements*2);
    for(size_t j=0;j<nelements;++j) z2idx[elements[j]]=(int)j;
    std::vector<int> elem_of_atom(natoms, -1);
    for(size_t i=0;i<natoms;++i){ auto it=z2idx.find(nuclear_z[i]); if(it==z2idx.end()) throw std::runtime_error("Unknown element in nuclear_z"); elem_of_atom[i]=it->second; }

    // Distances and powers
    const std::vector<double> D = pairwise_distances(coords, natoms);
    std::vector<double> D2(natoms*natoms,0.0), invD(natoms*natoms,0.0), invD2(natoms*natoms,0.0);
    for(size_t i=0;i<natoms;++i){
        for(size_t j=i+1;j<natoms;++j){
            double rij = D[idx2(i,j,natoms)];
            double rij2 = std::max(rij*rij, EPS);
            double invr = 1.0/rij;
            double invr2 = 1.0/rij2;
            D2[idx2(i,j,natoms)] = D2[idx2(j,i,natoms)] = rij2;
            invD[idx2(i,j,natoms)] = invD[idx2(j,i,natoms)] = invr;
            invD2[idx2(i,j,natoms)] = invD2[idx2(j,i,natoms)] = invr2;
        }
    }

    // Decays
    std::vector<double> rdecay2, rdecay3;
    if (rcut>0) decay_matrix(D, 1.0/rcut, natoms, rdecay2); else rdecay2.assign(natoms*natoms, 1.0);
    if (acut>0) decay_matrix(D, 1.0/acut, natoms, rdecay3); else rdecay3.assign(natoms*natoms, 1.0);

    // Precompute log(Rs2)
    std::vector<double> log_Rs2(nbasis2);
    for(size_t k=0;k<nbasis2;++k){ if(Rs2[k]<=0.0) throw std::invalid_argument("Rs2 must be >0"); log_Rs2[k]=std::log(Rs2[k]); }

    // ---------------- Two-body: values + gradients ----------------
    for(size_t i=0;i<natoms;++i){
        const int elem_i = elem_of_atom[i];
        for(size_t j=i+1;j<natoms;++j){
            const int elem_j = elem_of_atom[j];
            const double rij = D[idx2(i,j,natoms)];
            if (rij>rcut) continue;
            const double invr = invD[idx2(i,j,natoms)];
            const double invr2 = invD2[idx2(i,j,natoms)];
            const double s2 = std::log1p(eta2*invr2); // sigma^2
            const double sigma = std::sqrt(std::max(s2,0.0));
            if (sigma<EPS) continue;
            const double mu = std::log(rij) - 0.5*s2;
            const double decay_ij = rdecay2[idx2(i,j,natoms)];
            const double scaling = std::pow(rij, -two_body_decay);
            const double inv_pref_common = 1.0/(sigma * std::sqrt(2.0*PI));

            // radial_base[k] and radial[k]
            std::vector<double> radial_base(nbasis2), radial(nbasis2), exp_ln(nbasis2);
            for(size_t k=0;k<nbasis2;++k){
                const double dlog = log_Rs2[k] - mu;
                const double g = std::exp(-0.5 * dlog*dlog / s2);
                exp_ln[k] = g * std::sqrt(2.0); // as in Fortran (exp_ln used with sqrt(2))
                radial_base[k] = (inv_pref_common / Rs2[k]) * g;
                radial[k] = radial_base[k] * scaling * decay_ij;
            }

            // accumulate values to rows i and j, channels of counterpart element
            const size_t ch_j = (size_t)elem_j * nbasis2;
            const size_t ch_i = (size_t)elem_i * nbasis2;
            for(size_t k=0;k<nbasis2;++k){
                rep[idx2(i, ch_j + k, rep_size)] += radial[k];
                rep[idx2(j, ch_i + k, rep_size)] += radial[k];
            }

            // gradient contributions wrt coordinates of i and j
            const double exp_s2 = std::exp(s2);
            const double sqrt_exp_s2 = std::sqrt(exp_s2);
            for(int t=0;t<3;++t){
                const double dx = -(coords[3*i+t] - coords[3*j+t]); // -(ri - rj)
                // dscal = d(1/rij^p)/d x_i_t (with dx = rj_t - ri_t)
                const double dscal = two_body_decay * dx * std::pow(rij, -(two_body_decay+2.0));
                const double ddecay = dx * 0.5 * PI * std::sin(PI*rij*(rcut>0?1.0/rcut:0.0)) * (rcut>0?1.0/rcut:0.0) * invr;

                for(size_t k=0;k<nbasis2;++k){
                    const double L = log_Rs2[k] - mu;
                    // term inside Fortran's big bracket
                    const double term1 = L * (-dx * (rij*rij*exp_s2 + eta2) / std::pow(rij * sqrt_exp_s2, 3)) * (sqrt_exp_s2 / (s2 * rij));
                    const double term2 = (L*L) * eta2 * dx / ( (s2*s2) * std::pow(rij,4) * exp_s2 );
                    const double A = (term1 + term2) * (exp_ln[k] / (Rs2[k] * sigma * std::sqrt(PI) * 2.0))
                                   - (exp_ln[k] * eta2 * dx) / (Rs2[k] * (s2*std::sqrt(PI)) * sigma * std::pow(rij,4) * exp_s2 * 2.0);
                    double part = A * scaling * decay_ij + radial_base[k] * dscal * decay_ij + radial_base[k] * scaling * ddecay;

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
    const double tbd_over_w_pre = (three_body_weight != 0.0) ? (three_body_decay / three_body_weight) : 0.0;

    #pragma omp parallel for schedule(dynamic)
    for(size_t i=0;i<natoms;++i){
        // per-center buffers for the 3-body block
        std::vector<double> atom_rep(three_block_size, 0.0);
        std::vector<double> atom_grad(three_block_size * natoms * 3, 0.0);
        // Pre-allocated scratch buffers (avoid heap alloc per triplet)
        std::vector<double> radial(nbasis3), d_radial(nbasis3);
        std::vector<double> angular(nabasis), d_angular(nabasis);

        const double* rb = &coords[3*i];
        const double Bx = rb[0], By = rb[1], Bz = rb[2];
        for(size_t j=0;j+1<natoms;++j){
            if (j==i) continue;
            const double rij = D[idx2(i,j,natoms)]; if (rij>acut) continue;
            const double rij2 = D2[idx2(i,j,natoms)];
            const double invrij = invD[idx2(i,j,natoms)];
            const double invrij2 = invD2[idx2(i,j,natoms)];
            const int elem_j = elem_of_atom[j];
            const double* ra = &coords[3*j];
            const double Ax = ra[0], Ay = ra[1], Az = ra[2];

            // Precompute normalized edge vector eij = (A-B)/rij
            const double eij0 = (Ax - Bx) * invrij;
            const double eij1 = (Ay - By) * invrij;
            const double eij2 = (Az - Bz) * invrij;

            // Precompute decay derivative scalar for ij
            const double s_ij = -PI * std::sin(PI * rij * inv_acut) * 0.5 * invrij * inv_acut;

            for(size_t k=j+1;k<natoms;++k){
                if (k==i) continue;
                const double rik = D[idx2(i,k,natoms)]; if (rik>acut) continue;
                const double rik2 = D2[idx2(i,k,natoms)];
                const double invrik = invD[idx2(i,k,natoms)];
                const double invrik2 = invD2[idx2(i,k,natoms)];
                const double invrjk = invD[idx2(j,k,natoms)];
                const int elem_k = elem_of_atom[k];
                const double* rc = &coords[3*k];
                const double Cx = rc[0], Cy = rc[1], Cz = rc[2];

                // Normalized edge vectors
                const double eik0 = (Cx - Bx) * invrik;
                const double eik1 = (Cy - By) * invrik;
                const double eik2 = (Cz - Bz) * invrik;
                const double ejk0 = (Cx - Ax) * invrjk;
                const double ejk1 = (Cy - Ay) * invrjk;
                const double ejk2 = (Cz - Az) * invrjk;

                // Cosines from dot products (no acos/cos/sin needed)
                double cos_i = eij0*eik0 + eij1*eik1 + eij2*eik2;
                cos_i = std::max(-1.0, std::min(1.0, cos_i));
                const double cos_j = -(eij0*ejk0 + eij1*ejk1 + eij2*ejk2);
                const double cos_k = eik0*ejk0 + eik1*ejk1 + eik2*ejk2;

                // Raw dot product (A-B)·(C-B) — needed for angle gradient
                const double dot = (Ax-Bx)*(Cx-Bx) + (Ay-By)*(Cy-By) + (Az-Bz)*(Cz-Bz);

                // Radial parts per Rs3 center (reuse pre-allocated buffer)
                for(size_t l=0;l<nbasis3;++l){
                    const double rbar = 0.5*(rij+rik) - Rs3[l];
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
                const double denom = std::sqrt(std::max(1e-10, rij2*rik2 - dot*dot));
                // direction vectors multiplying d_angular
                const double d_ang_d_j0 = Cx-Bx + dot*((Bx-Ax)*invrij2);
                const double d_ang_d_j1 = Cy-By + dot*((By-Ay)*invrij2);
                const double d_ang_d_j2 = Cz-Bz + dot*((Bz-Az)*invrij2);
                const double d_ang_d_k0 = Ax-Bx + dot*((Bx-Cx)*invrik2);
                const double d_ang_d_k1 = Ay-By + dot*((By-Cy)*invrik2);
                const double d_ang_d_k2 = Az-Bz + dot*((Bz-Cz)*invrik2);
                const double d_ang_d_i0 = -(d_ang_d_j0 + d_ang_d_k0);
                const double d_ang_d_i1 = -(d_ang_d_j1 + d_ang_d_k1);
                const double d_ang_d_i2 = -(d_ang_d_j2 + d_ang_d_k2);

                // Decay and decay derivatives
                const double decay_ij = rdecay3[idx2(i,j,natoms)];
                const double decay_ik = rdecay3[idx2(i,k,natoms)];
                // d_ijdecay[t] = s_ij * (B[t] - A[t])
                const double d_ijd0 = s_ij * (Bx-Ax), d_ijd1 = s_ij * (By-Ay), d_ijd2 = s_ij * (Bz-Az);
                const double s_ik = -PI * std::sin(PI * rik * inv_acut) * 0.5 * invrik * inv_acut;
                const double d_ikd0 = s_ik * (Bx-Cx), d_ikd1 = s_ik * (By-Cy), d_ikd2 = s_ik * (Bz-Cz);

                // ATM factor and its pieces
                const double invr_atm = std::pow(invrij * invrjk * invrik, three_body_decay);
                const double atm = (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight;
                const double atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik;
                const double atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk;
                const double atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik;

                const double vi = dot; // (A-B)·(C-B)
                const double vj = (Cx-Ax)*(Bx-Ax) + (Cy-Ay)*(By-Ay) + (Cz-Az)*(Bz-Az);
                const double vk = (Bx-Cx)*(Ax-Cx) + (By-Cy)*(Ay-Cy) + (Bz-Cz)*(Az-Cz);

                // ATM derivative direction vectors (3 atoms x 3 derivative sources)
                const double d_atm_ii0 = 2*Bx-Ax-Cx - vi*((Bx-Ax)*invrij2 + (Bx-Cx)*invrik2);
                const double d_atm_ii1 = 2*By-Ay-Cy - vi*((By-Ay)*invrij2 + (By-Cy)*invrik2);
                const double d_atm_ii2 = 2*Bz-Az-Cz - vi*((Bz-Az)*invrij2 + (Bz-Cz)*invrik2);
                const double d_atm_ij0 = Cx-Ax - vj*(Bx-Ax)*invrij2;
                const double d_atm_ij1 = Cy-Ay - vj*(By-Ay)*invrij2;
                const double d_atm_ij2 = Cz-Az - vj*(Bz-Az)*invrij2;
                const double d_atm_ik0 = Ax-Cx - vk*(Bx-Cx)*invrik2;
                const double d_atm_ik1 = Ay-Cy - vk*(By-Cy)*invrik2;
                const double d_atm_ik2 = Az-Cz - vk*(Bz-Cz)*invrik2;

                const double d_atm_ji0 = Cx-Bx - vi*(Ax-Bx)*invrij2;
                const double d_atm_ji1 = Cy-By - vi*(Ay-By)*invrij2;
                const double d_atm_ji2 = Cz-Bz - vi*(Az-Bz)*invrij2;
                const double d_atm_jj0 = 2*Ax-Bx-Cx - vj*((Ax-Bx)*invrij2 + (Ax-Cx)*invrjk*invrjk);
                const double d_atm_jj1 = 2*Ay-By-Cy - vj*((Ay-By)*invrij2 + (Ay-Cy)*invrjk*invrjk);
                const double d_atm_jj2 = 2*Az-Bz-Cz - vj*((Az-Bz)*invrij2 + (Az-Cz)*invrjk*invrjk);
                const double d_atm_jk0 = Bx-Cx - vk*(Ax-Cx)*invrjk*invrjk;
                const double d_atm_jk1 = By-Cy - vk*(Ay-Cy)*invrjk*invrjk;
                const double d_atm_jk2 = Bz-Cz - vk*(Az-Cz)*invrjk*invrjk;

                const double d_atm_ki0 = Ax-Bx - vi*(Cx-Bx)*invrik2;
                const double d_atm_ki1 = Ay-By - vi*(Cy-By)*invrik2;
                const double d_atm_ki2 = Az-Bz - vi*(Cz-Bz)*invrik2;
                const double d_atm_kj0 = Bx-Ax - vj*(Cx-Ax)*invrjk*invrjk;
                const double d_atm_kj1 = By-Ay - vj*(Cy-Ay)*invrjk*invrjk;
                const double d_atm_kj2 = Bz-Az - vj*(Cz-Az)*invrjk*invrjk;
                const double d_atm_kk0 = 2*Cx-Ax-Bx - vk*((Cx-Ax)*invrjk*invrjk + (Cx-Bx)*invrik2);
                const double d_atm_kk1 = 2*Cy-Ay-By - vk*((Cy-Ay)*invrjk*invrjk + (Cy-By)*invrik2);
                const double d_atm_kk2 = 2*Cz-Az-Bz - vk*((Cz-Az)*invrjk*invrjk + (Cz-Bz)*invrik2);

                const double atm_tbd = atm * tbd_over_w_pre;
                const double d_extra_i0 = ((Ax-Bx)*invrij2 + (Cx-Bx)*invrik2) * atm_tbd;
                const double d_extra_i1 = ((Ay-By)*invrij2 + (Cy-By)*invrik2) * atm_tbd;
                const double d_extra_i2 = ((Az-Bz)*invrij2 + (Cz-Bz)*invrik2) * atm_tbd;
                const double d_extra_j0 = ((Bx-Ax)*invrij2 + (Cx-Ax)*invrjk*invrjk) * atm_tbd;
                const double d_extra_j1 = ((By-Ay)*invrij2 + (Cy-Ay)*invrjk*invrjk) * atm_tbd;
                const double d_extra_j2 = ((Bz-Az)*invrij2 + (Cz-Az)*invrjk*invrjk) * atm_tbd;
                const double d_extra_k0 = ((Ax-Cx)*invrjk*invrjk + (Bx-Cx)*invrik2) * atm_tbd;
                const double d_extra_k1 = ((Ay-Cy)*invrjk*invrjk + (By-Cy)*invrik2) * atm_tbd;
                const double d_extra_k2 = ((Az-Cz)*invrjk*invrjk + (Bz-Cz)*invrik2) * atm_tbd;

                // Element pair index
                const size_t pair_idx0 = pair_index(nelements, elem_j, elem_k);
                const size_t base = pair_idx0 * (nbasis3 * nabasis);

                // Precompute per-dimension gradient building blocks
                const double inv_denom = 1.0 / denom;
                // d_ang direction / denom for each atom and dim
                const double dai0 = d_ang_d_i0 * inv_denom, dai1 = d_ang_d_i1 * inv_denom, dai2 = d_ang_d_i2 * inv_denom;
                const double daj0 = d_ang_d_j0 * inv_denom, daj1 = d_ang_d_j1 * inv_denom, daj2 = d_ang_d_j2 * inv_denom;
                const double dak0 = d_ang_d_k0 * inv_denom, dak1 = d_ang_d_k1 * inv_denom, dak2 = d_ang_d_k2 * inv_denom;

                // ATM derivatives (full, including three_body_weight)
                const double atmi0 = (atm_i*d_atm_ii0 + atm_j*d_atm_ij0 + atm_k*d_atm_ik0 + d_extra_i0) * three_body_weight;
                const double atmi1 = (atm_i*d_atm_ii1 + atm_j*d_atm_ij1 + atm_k*d_atm_ik1 + d_extra_i1) * three_body_weight;
                const double atmi2 = (atm_i*d_atm_ii2 + atm_j*d_atm_ij2 + atm_k*d_atm_ik2 + d_extra_i2) * three_body_weight;
                const double atmj0 = (atm_i*d_atm_ji0 + atm_j*d_atm_jj0 + atm_k*d_atm_jk0 + d_extra_j0) * three_body_weight;
                const double atmj1 = (atm_i*d_atm_ji1 + atm_j*d_atm_jj1 + atm_k*d_atm_jk1 + d_extra_j1) * three_body_weight;
                const double atmj2 = (atm_i*d_atm_ji2 + atm_j*d_atm_jj2 + atm_k*d_atm_jk2 + d_extra_j2) * three_body_weight;
                const double atmk0 = (atm_i*d_atm_ki0 + atm_j*d_atm_kj0 + atm_k*d_atm_kk0 + d_extra_k0) * three_body_weight;
                const double atmk1 = (atm_i*d_atm_ki1 + atm_j*d_atm_kj1 + atm_k*d_atm_kk1 + d_extra_k1) * three_body_weight;
                const double atmk2 = (atm_i*d_atm_ki2 + atm_j*d_atm_kj2 + atm_k*d_atm_kk2 + d_extra_k2) * three_body_weight;

                // Decay combination derivatives per dim
                const double dec_i0 = d_ijd0*decay_ik + decay_ij*d_ikd0;
                const double dec_i1 = d_ijd1*decay_ik + decay_ij*d_ikd1;
                const double dec_i2 = d_ijd2*decay_ik + decay_ij*d_ikd2;
                const double dec_j0 = -d_ijd0*decay_ik;
                const double dec_j1 = -d_ijd1*decay_ik;
                const double dec_j2 = -d_ijd2*decay_ik;
                const double dec_k0 = -decay_ij*d_ikd0;
                const double dec_k1 = -decay_ij*d_ikd1;
                const double dec_k2 = -decay_ij*d_ikd2;

                // Radial derivative directional parts (precomputed per dim)
                const double BmA0 = (Bx-Ax)*invrij, BmA1 = (By-Ay)*invrij, BmA2 = (Bz-Az)*invrij;
                const double BmC0 = (Bx-Cx)*invrik, BmC1 = (By-Cy)*invrik, BmC2 = (Bz-Cz)*invrik;

                const double decay_prod = decay_ij * decay_ik;

                for(size_t l=0;l<nbasis3;++l){
                    const double scale_val = radial[l] * atm * decay_prod;
                    const double scale_ang = decay_prod * radial[l];
                    const double d_rad_l = d_radial[l];
                    const size_t z0 = base + l * nabasis;

                    // values
                    for(size_t aidx=0;aidx<nabasis;++aidx){
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

                    for(size_t aidx=0;aidx<nabasis;++aidx){
                        const double ang = angular[aidx];
                        const double dang = d_angular[aidx];

                        // dim 0
                        {
                            const double gi = dang*dai0*scale_ang*atm + ang*dri0*atm*decay_prod + ang*rad_l*atmi0*decay_prod + ang*rad_l*dec_i0*atm;
                            const double gj = dang*daj0*scale_ang*atm + ang*drj0*atm*decay_prod + ang*rad_l*atmj0*decay_prod + ang*rad_l*dec_j0*atm;
                            const double gk = dang*dak0*scale_ang*atm + ang*drk0*atm*decay_prod + ang*rad_l*atmk0*decay_prod + ang*rad_l*dec_k0*atm;
                            atom_grad[((z0+aidx)*natoms+i)*3+0] += gi;
                            atom_grad[((z0+aidx)*natoms+j)*3+0] += gj;
                            atom_grad[((z0+aidx)*natoms+k)*3+0] += gk;
                        }
                        // dim 1
                        {
                            const double gi = dang*dai1*scale_ang*atm + ang*dri1*atm*decay_prod + ang*rad_l*atmi1*decay_prod + ang*rad_l*dec_i1*atm;
                            const double gj = dang*daj1*scale_ang*atm + ang*drj1*atm*decay_prod + ang*rad_l*atmj1*decay_prod + ang*rad_l*dec_j1*atm;
                            const double gk = dang*dak1*scale_ang*atm + ang*drk1*atm*decay_prod + ang*rad_l*atmk1*decay_prod + ang*rad_l*dec_k1*atm;
                            atom_grad[((z0+aidx)*natoms+i)*3+1] += gi;
                            atom_grad[((z0+aidx)*natoms+j)*3+1] += gj;
                            atom_grad[((z0+aidx)*natoms+k)*3+1] += gk;
                        }
                        // dim 2
                        {
                            const double gi = dang*dai2*scale_ang*atm + ang*dri2*atm*decay_prod + ang*rad_l*atmi2*decay_prod + ang*rad_l*dec_i2*atm;
                            const double gj = dang*daj2*scale_ang*atm + ang*drj2*atm*decay_prod + ang*rad_l*atmj2*decay_prod + ang*rad_l*dec_j2*atm;
                            const double gk = dang*dak2*scale_ang*atm + ang*drk2*atm*decay_prod + ang*rad_l*atmk2*decay_prod + ang*rad_l*dec_k2*atm;
                            atom_grad[((z0+aidx)*natoms+i)*3+2] += gi;
                            atom_grad[((z0+aidx)*natoms+j)*3+2] += gj;
                            atom_grad[((z0+aidx)*natoms+k)*3+2] += gk;
                        }
                    }
                }
            }
        }

        // scatter into global rep/grad for center i (3-body block only)
        for(size_t off=0; off<three_block_size; ++off){
            rep[idx2(i, three_offset + off, rep_size)] += atom_rep[off];
        }
        for(size_t off=0; off<three_block_size; ++off){
            for(size_t a=0;a<natoms;++a){
                for(size_t t=0;t<3;++t){
                    const double v = atom_grad[(off*natoms + a)*3 + t];
                    if (v != 0.0) grad[gidx(i, three_offset + off, a, t, rep_size, natoms)] += v;
                }
            }
        }
    }
}


//  #########################
//  # FCHL19 KERNEL HELPERS #
//  #########################

// Distinct labels across both sets, with q1 (nm1,max_atoms1) and q2 (nm2,max_atoms2)
static inline void collect_distinct_labels_T(
    const std::vector<int>& q1, int nm1, int max_atoms1, const std::vector<int>& n1,
    const std::vector<int>& q2, int nm2, int max_atoms2, const std::vector<int>& n2,
    std::vector<int>& labels_out
) {
    std::unordered_map<int, char> seen;
    seen.reserve(128);

    for (int a = 0; a < nm1; ++a) {
        int na = std::min(std::max(n1[a], 0), max_atoms1);
        for (int j = 0; j < na; ++j) {
            int lbl = q1[(std::size_t)a * max_atoms1 + j];
            seen.emplace(lbl, 1);
        }
    }
    for (int b = 0; b < nm2; ++b) {
        int nb = std::min(std::max(n2[b], 0), max_atoms2);
        for (int j = 0; j < nb; ++j) {
            int lbl = q2[(std::size_t)b * max_atoms2 + j];
            seen.emplace(lbl, 1);
        }
    }
    labels_out.clear();
    labels_out.reserve(seen.size());
    for (auto& kv : seen) labels_out.push_back(kv.first);
    std::sort(labels_out.begin(), labels_out.end());
}

// Pack for asymmetric case with q1 (nm1,max_atoms1), q2 (nm2,max_atoms2)
struct PackedLabel {
    // Dense blocks
    std::vector<double> A;      // (R x rep_size), rows from set 1
    std::vector<double> B;      // (S x rep_size), rows from set 2
    std::vector<double> row_n2; // (R) for A rows
    std::vector<double> col_n2; // (S) for B rows
    int R = 0, S = 0;

    // Per-molecule mapping to row/col indices
    std::vector<std::vector<int>> rows_per_mol1; // size nm1
    std::vector<std::vector<int>> cols_per_mol2; // size nm2
};

static inline PackedLabel pack_label_block_T(
    int label,
    const std::vector<double>& x1, int nm1, int max_atoms1, int rep_size,
    const std::vector<double>& x2, int nm2, int max_atoms2,
    const std::vector<int>& q1, const std::vector<int>& q2,
    const std::vector<int>& n1, const std::vector<int>& n2
) {
    PackedLabel pk;
    pk.rows_per_mol1.resize(nm1);
    pk.cols_per_mol2.resize(nm2);

    // Count rows/cols
    int R = 0, S = 0;
    for (int a = 0; a < nm1; ++a) {
        const int na = std::min(std::max(n1[a], 0), max_atoms1);
        for (int j = 0; j < na; ++j)
            if (q1[(std::size_t)a * max_atoms1 + j] == label) ++R;
    }
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::min(std::max(n2[b], 0), max_atoms2);
        for (int j = 0; j < nb; ++j)
            if (q2[(std::size_t)b * max_atoms2 + j] == label) ++S;
    }
    pk.R = R; pk.S = S;
    if (R == 0 || S == 0) return pk;

    pk.A.resize((std::size_t)R * rep_size);
    pk.B.resize((std::size_t)S * rep_size);
    pk.row_n2.resize(R);
    pk.col_n2.resize(S);

    // Pack A (set 1)
    int ridx = 0;
    for (int a = 0; a < nm1; ++a) {
        const int na = std::min(std::max(n1[a], 0), max_atoms1);
        for (int j = 0; j < na; ++j) {
            if (q1[(std::size_t)a * max_atoms1 + j] != label) continue;

            const std::size_t base1 = (std::size_t)a * max_atoms1 * rep_size
                                    + (std::size_t)j * rep_size;
            double n2sum = 0.0;
            double* Ai = &pk.A[(std::size_t)ridx * rep_size];
            for (int k = 0; k < rep_size; ++k) {
                const double v = x1[base1 + k];
                Ai[k] = v;
                n2sum += v * v;
            }
            pk.row_n2[ridx] = n2sum;
            pk.rows_per_mol1[a].push_back(ridx);
            ++ridx;
        }
    }

    // Pack B (set 2)
    int cidx = 0;
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::min(std::max(n2[b], 0), max_atoms2);
        for (int j = 0; j < nb; ++j) {
            if (q2[(std::size_t)b * max_atoms2 + j] != label) continue;

            const std::size_t base2 = (std::size_t)b * max_atoms2 * rep_size
                                    + (std::size_t)j * rep_size;
            double n2sum = 0.0;
            double* Bj = &pk.B[(std::size_t)cidx * rep_size];
            for (int k = 0; k < rep_size; ++k) {
                const double v = x2[base2 + k];
                Bj[k] = v;
                n2sum += v * v;
            }
            pk.col_n2[cidx] = n2sum;
            pk.cols_per_mol2[b].push_back(cidx);
            ++cidx;
        }
    }

    return pk;
}

//  #################################
//  # FCHL19 KERNEL IMPLEMENTATION #
//  #################################

void flocal_kernel(
    const std::vector<double>& x1,   // (nm1, max_atoms1, rep_size)
    const std::vector<double>& x2,   // (nm2, max_atoms2, rep_size)
    const std::vector<int>&    q1,   // (nm1, max_atoms1)
    const std::vector<int>&    q2,   // (nm2, max_atoms2)
    const std::vector<int>&    n1,   // (nm1)
    const std::vector<int>&    n2,   // (nm2)
    int nm1,
    int nm2,
    int max_atoms1,
    int max_atoms2,
    int rep_size,
    double sigma,
    double* kernel                 // (nm1, nm2), row-major: kernel[a*nm2 + b]
) {
    if (!kernel) throw std::invalid_argument("kernel_out is null");
    if (!(std::isfinite(sigma)) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be > 0");

    // Zero K (full rectangular matrix)
    std::memset(kernel, 0, sizeof(double) * (size_t)nm1 * nm2);

    const double inv_sigma2 = -1.0 / (2.0 * sigma * sigma);

    // Gather labels shared by the two sets
    std::vector<int> labels;
    collect_distinct_labels_T(q1, nm1, max_atoms1, n1,
                              q2, nm2, max_atoms2, n2,
                              labels);
    if (labels.empty()) return;

    // ---- Tunable tile size ----
    const int B = 8192; // try 1024–4096; 8192 if you have RAM (B×B doubles scratch)

    for (int label : labels) {
        // Pack rows/cols for this label
        auto pk = pack_label_block_T(label,
                                     x1, nm1, max_atoms1, rep_size,
                                     x2, nm2, max_atoms2,
                                     q1, q2, n1, n2);
        const int R = pk.R, S = pk.S;
        if (R == 0 || S == 0) continue;

        const int nblkR = (R + B - 1) / B;
        const int nblkS = (S + B - 1) / B;

        // Bucket molecule rows/cols by block id to avoid inner-range checks
        std::vector<std::vector<std::vector<int>>> bucketsR(nm1, std::vector<std::vector<int>>(nblkR));
        std::vector<std::vector<std::vector<int>>> bucketsS(nm2, std::vector<std::vector<int>>(nblkS));
        for (int a = 0; a < nm1; ++a) {
            for (int gi : pk.rows_per_mol1[a]) bucketsR[a][gi / B].push_back(gi);
        }
        for (int b = 0; b < nm2; ++b) {
            for (int gj : pk.cols_per_mol2[b]) bucketsS[b][gj / B].push_back(gj);
        }

        // Reusable aligned scratch tile (B×B), written anew per dgemm call
        double* Cblk = aligned_alloc_64((size_t)B * B);
        if (!Cblk) throw std::bad_alloc();

        // ---- Tile over (rows, cols) ----
        for (int i0 = 0; i0 < R; i0 += B) {
            const int ib = std::min(B, R - i0);
            const double* Ai0 = &pk.A[(size_t)i0 * rep_size];
            const int bi = i0 / B;

            for (int j0 = 0; j0 < S; j0 += B) {
                const int jb = std::min(B, S - j0);
                const double* Bj0 = &pk.B[(size_t)j0 * rep_size];
                const int bj = j0 / B;

                // Cblk(ib×jb) = -2 * Ai0(ib×rep) * Bj0(jb×rep)^T
                // RowMajor: M=ib, N=jb, K=rep_size, lda=rep_size, ldb=rep_size, ldc=jb
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ib, jb, rep_size,
                            -2.0, Ai0, rep_size, Bj0, rep_size,
                             0.0, Cblk, jb);

                // Accumulate this tile into K
                #pragma omp parallel for schedule(guided)
                for (int a = 0; a < nm1; ++a) {
                    const auto& Ia = bucketsR[a][bi];
                    if (Ia.empty()) continue;

                    for (int b = 0; b < nm2; ++b) {
                        const auto& Jb = bucketsS[b][bj];
                        if (Jb.empty()) continue;

                        double kab = 0.0;

                        for (int gi : Ia) {
                            const int li = gi - i0;                  // 0..ib-1
                            const double rn = pk.row_n2[gi];
                            const double* __restrict Grow = Cblk + (size_t)li * jb;

                            // Unroll over columns in this block
                            const int mJ = (int)Jb.size();
                            int t = 0;
                            for (; t + 3 < mJ; t += 4) {
                                const int j0g = Jb[t+0], j1g = Jb[t+1];
                                const int j2g = Jb[t+2], j3g = Jb[t+3];
                                const int l0 = j0g - j0, l1 = j1g - j0;
                                const int l2 = j2g - j0, l3 = j3g - j0;

                                kab += std::exp((rn + pk.col_n2[j0g] + Grow[l0]) * inv_sigma2)
                                     + std::exp((rn + pk.col_n2[j1g] + Grow[l1]) * inv_sigma2)
                                     + std::exp((rn + pk.col_n2[j2g] + Grow[l2]) * inv_sigma2)
                                     + std::exp((rn + pk.col_n2[j3g] + Grow[l3]) * inv_sigma2);
                            }
                            for (; t < mJ; ++t) {
                                const int jg = Jb[t];
                                kab += std::exp((rn + pk.col_n2[jg] + Grow[jg - j0]) * inv_sigma2);
                            }
                        }

                        // Row-major (nm1, nm2): stride-1 across 'b'
                        kernel[(size_t)a * nm2 + b] += kab;
                    }
                }
            } // j0
        }     // i0

        std::free(Cblk);
    } // labels
}

//  ###################################
//  # FCHL19 SYMMETRIC KERNEL HELPERS #
//  ###################################

// --- helper: collect labels for single set with q shape (nm, max_atoms)
static inline void collect_distinct_labels_single_T(
    const std::vector<int>& q, int nm, int max_atoms, const std::vector<int>& n,
    std::vector<int>& labels_out
) {
    std::unordered_map<int, char> seen;
    seen.reserve(128);
    for (int a = 0; a < nm; ++a) {
        int na = std::min(std::max(n[a], 0), max_atoms);
        for (int j = 0; j < na; ++j) {
            seen.emplace(q[(std::size_t)a * max_atoms + j], 1);
        }
    }
    labels_out.clear();
    labels_out.reserve(seen.size());
    for (auto& kv : seen) labels_out.push_back(kv.first);
    std::sort(labels_out.begin(), labels_out.end());
}

// --- pack for symmetric case (single set), q shape (nm, max_atoms)
struct PackedLabelSym {
    std::vector<double> A;                  // (R x rep_size), row-major
    std::vector<double> row_n2;             // (R)
    std::vector<std::vector<int>> rows_per_mol; // size nm
    int R = 0;
};

static inline PackedLabelSym pack_label_block_sym_T(
    int label,
    const std::vector<double>& x,  int nm, int max_atoms, int rep_size,
    const std::vector<int>& q,     const std::vector<int>& n
) {
    PackedLabelSym pk;
    pk.rows_per_mol.resize(nm);

    // count rows
    int R = 0;
    for (int a = 0; a < nm; ++a) {
        const int na = std::min(std::max(n[a], 0), max_atoms);
        for (int j = 0; j < na; ++j)
            if (q[(std::size_t)a * max_atoms + j] == label) ++R;
    }
    pk.R = R;
    if (R == 0) return pk;

    pk.A.resize((std::size_t)R * rep_size);
    pk.row_n2.resize(R);

    // pack rows
    int ridx = 0;
    for (int a = 0; a < nm; ++a) {
        const int na = std::min(std::max(n[a], 0), max_atoms);
        for (int j = 0; j < na; ++j) {
            if (q[(std::size_t)a * max_atoms + j] != label) continue;

            const std::size_t base = (std::size_t)a * max_atoms * rep_size
                                   + (std::size_t)j * rep_size;

            double n2sum = 0.0;
            double* Ai = &pk.A[(std::size_t)ridx * rep_size];
            for (int k = 0; k < rep_size; ++k) {
                const double v = x[base + k];
                Ai[k] = v;
                n2sum += v * v;
            }
            pk.row_n2[ridx] = n2sum;
            pk.rows_per_mol[a].push_back(ridx);
            ++ridx;
        }
    }
    return pk;
}

// assume you already have:
//   double* aligned_alloc_64(size_t n);  // aligned, uninitialized; free with std::free


//  #######################################
//  # FCHL19 KERNEL PACKED IMPLEMENTATION #
//  #######################################

// Map (i,j) with 0-based indices and i <= j to linear 0-based RFP index.
// Matches your Fortran rfp_index (TRANSR='N', UPLO='U').
static inline size_t rfp_index_upper_N(int n, int i, int j) {
    // Preconditions: 0 <= i <= j < n
    // if (i > j) std::swap(i, j);
    const int k = n / 2;                       // floor(n/2)
    const int stride = (n % 2 == 0) ? (n + 1)  : n;  // even->n+1, odd->n

    // "top" zone if j >= k  (same as your Fortran)
    // top:   idx = (j-k)*stride + i
    // bottom:idx = i*stride + j + k + 1
    if (j >= k) {
        return (size_t)(j - k) * (size_t)stride + (size_t)i;
    } else {
        return (size_t)i * (size_t)stride + (size_t)(j + k + 1);
    }
}


// Tiled, memory-bounded version that writes **directly to RFP** (TRANSR='N', UPLO='U')
void flocal_kernel_symmetric_rfp(
    const std::vector<double>& x,   // (nm, max_atoms, rep_size)
    const std::vector<int>&    q,   // (nm, max_atoms)
    const std::vector<int>&    n,   // (nm)
    int nm,
    int max_atoms,
    int rep_size,
    double sigma,
    double* arf                 // length nt = nm*(nm+1)/2, output RFP (TRANSR='N', UPLO='U')
) {
    if (!arf) throw std::invalid_argument("arf (RFP output) is null");
    if (!(std::isfinite(sigma)) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be > 0");

    // Zero the RFP vector
    const size_t nt = (size_t)nm * (nm + 1ull) / 2ull;
    std::memset(arf, 0, nt * sizeof(double));

    const double inv_sigma2 = -1.0 / (2.0 * sigma * sigma);

    // Collect labels
    std::vector<int> labels;
    collect_distinct_labels_single_T(q, nm, max_atoms, n, labels);
    if (labels.empty()) return;

    // Tile size (tune as before)
    const int B = 8192;

    for (int label : labels) {
        // Pack per-label rows (same as your symmetric path)
        PackedLabelSym pk = pack_label_block_sym_T(label, x, nm, max_atoms, rep_size, q, n);
        const int R = pk.R;
        if (R == 0) continue;

        const int num_blocks = (R + B - 1) / B;

        // Bucket molecule rows by block id to avoid inner checks
        std::vector<std::vector<std::vector<int>>> buckets(nm, std::vector<std::vector<int>>(num_blocks));
        for (int a = 0; a < nm; ++a) {
            const auto& rows = pk.rows_per_mol[a];
            for (int gi : rows) buckets[a][gi / B].push_back(gi);
        }

        // Scratch tile
        double* Cblk = aligned_alloc_64((size_t)B * B);
        if (!Cblk) throw std::bad_alloc();

        for (int i0 = 0; i0 < R; i0 += B) {
            const int ib = std::min(B, R - i0);
            const double* Ai0 = &pk.A[(size_t)i0 * rep_size];
            const int bi = i0 / B;

            // ----- Diagonal tile: DSYRK → upper-tri in Cblk (LDC = ib) -----
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                        ib, rep_size, -2.0, Ai0, rep_size, 0.0, Cblk, ib);

            #pragma omp parallel for schedule(guided)
            for (int a = 0; a < nm; ++a) {
                const auto& Ia = buckets[a][bi];
                if (Ia.empty()) continue;

                for (int b = a; b < nm; ++b) {
                    const auto& Ib = buckets[b][bi];
                    if (Ib.empty()) continue;

                    double kab = 0.0;
                    for (int gi : Ia) {
                        const int li = gi - i0;           // [0..ib)
                        const double rn_i = pk.row_n2[gi];
                        for (int gj : Ib) {
                            const int lj = gj - i0;       // [0..ib)
                            const int r = (li <= lj) ? li : lj;
                            const int c = (li <= lj) ? lj : li;
                            const double cij = Cblk[(size_t)r * ib + c];
                            const double l2  = rn_i + pk.row_n2[gj] + cij;
                            kab += std::exp(l2 * inv_sigma2);
                        }
                    }
                    // Write once to RFP (a<=b)
                    const size_t idx = rfp_index_upper_N(nm, a, b);
                    arf[idx] += kab;
                }
            }

            // ----- Off-diagonal rectangles: DGEMM (ib × jb) -----
            for (int j0 = i0 + B; j0 < R; j0 += B) {
                const int jb = std::min(B, R - j0);
                const double* Aj0 = &pk.A[(size_t)j0 * rep_size];
                const int bj = j0 / B;

                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ib, jb, rep_size, -2.0, Ai0, rep_size, Aj0, rep_size,
                            0.0, Cblk, jb);

                #pragma omp parallel for schedule(guided)
                for (int a = 0; a < nm; ++a) {
                    const auto& Ia = buckets[a][bi];
                    if (Ia.empty()) continue;

                    for (int b = a; b < nm; ++b) {
                        const auto& Jb = buckets[b][bj];
                        if (Jb.empty()) continue;

                        double kab = 0.0;
                        for (int gi : Ia) {
                            const int li = gi - i0;       // [0..ib)
                            const double rn_i = pk.row_n2[gi];
                            for (int gj : Jb) {
                                const int lj = gj - j0;   // [0..jb)
                                const double cij = Cblk[(size_t)li * jb + lj];
                                const double l2  = rn_i + pk.row_n2[gj] + cij;
                                kab += std::exp(l2 * inv_sigma2);
                            }
                        }
                        // Self-kernel cross-block pairs counted twice
                        if (a == b) kab *= 2.0;

                        const size_t idx = rfp_index_upper_N(nm, a, b);
                        arf[idx] += kab;
                    }
                }
            } // j0
        }     // i0

        std::free(Cblk);
    } // labels
}


//  ###########################################
//  # FCHL19 KERNEL SYMMETRIC IMPLEMENTATION #
//  ###########################################


// Tiled, memory-bounded version: only allocates a B×B scratch tile.
void flocal_kernel_symmetric(
    const std::vector<double>& x,   // (nm, max_atoms, rep_size)
    const std::vector<int>&    q,   // (nm, max_atoms)
    const std::vector<int>&    n,   // (nm)
    int nm,
    int max_atoms,
    int rep_size,
    double sigma,
    double* kernel                 // (nm, nm), row-major: kernel[a*nm + b]
) {
    if (!kernel) throw std::invalid_argument("kernel_out is null");
    if (!(std::isfinite(sigma)) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be > 0");

    // Zero once (full matrix; cheap and simple). If you only ever write the lower
    // triangle, you can halve this by zeroing rows up to i inclusive.
    // Initialize kernel matrix to zero
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < (size_t)nm * nm; ++i) {
        kernel[i] = 0.0;
    }

    const double inv_sigma2 = -1.0 / (2.0 * sigma * sigma);


    std::vector<int> labels;
    collect_distinct_labels_single_T(q, nm, max_atoms, n, labels);
    if (labels.empty()) return;

    const int B = 8192; // tile size; tune 512–2048

    for (int label : labels) {
        PackedLabelSym pk = pack_label_block_sym_T(label, x, nm, max_atoms, rep_size, q, n);
        const int R = pk.R;
        if (R == 0) continue;

        const int num_blocks = (R + B - 1) / B;

        // bucket molecule rows by block id
        std::vector<std::vector<std::vector<int>>> buckets(nm, std::vector<std::vector<int>>(num_blocks));
        for (int a = 0; a < nm; ++a) {
            const auto& rows = pk.rows_per_mol[a];
            for (int gi : rows) buckets[a][gi / B].push_back(gi);
        }

        // ---- aligned scratch tile (reused for all tiles of this label) ----
        double* Cblk = aligned_alloc_64((size_t)B * B);
        if (!Cblk) throw std::bad_alloc();

        for (int i0 = 0; i0 < R; i0 += B) {
            const int ib = std::min(B, R - i0);
            const double* Ai0 = &pk.A[(size_t)i0 * rep_size];
            const int bi = i0 / B;

            // Diagonal tile: DSYRK produces upper-tri in Cblk with LDC=ib
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans,
                        ib, rep_size,
                        -2.0, Ai0, rep_size,
                         0.0,  Cblk, ib);

            #pragma omp parallel for schedule(guided)
            for (int a = 0; a < nm; ++a) {
                const auto& Ia = buckets[a][bi];
                if (Ia.empty()) continue;

                for (int b = a; b < nm; ++b) {
                    const auto& Ib = buckets[b][bi];
                    if (Ib.empty()) continue;

                    double kab = 0.0;
                    for (int gi : Ia) {
                        const int li = gi - i0;           // [0..ib)
                        const double rn_i = pk.row_n2[gi];
                        for (int gj : Ib) {
                            const int lj = gj - i0;       // [0..ib)
                            // index with LDC = ib (upper-tri)
                            const int r = (li <= lj) ? li : lj;
                            const int c = (li <= lj) ? lj : li;
                            const double cij = Cblk[(size_t)r * ib + c];
                            const double l2  = rn_i + pk.row_n2[gj] + cij;
                            kab += std::exp(l2 * inv_sigma2);
                        }
                    }
                    kernel[(size_t)a * nm + b] += kab;
                    if (b != a) kernel[(size_t)b * nm + a] += kab;
                }
            }

            // Off-diagonal rectangles: DGEMM to Cblk (LDC=jb)
            for (int j0 = i0 + B; j0 < R; j0 += B) {
                const int jb = std::min(B, R - j0);
                const double* Aj0 = &pk.A[(size_t)j0 * rep_size];
                const int bj = j0 / B;

                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ib, jb, rep_size,
                            -2.0, Ai0, rep_size, Aj0, rep_size,
                             0.0,  Cblk, jb);

                #pragma omp parallel for schedule(guided)
                for (int a = 0; a < nm; ++a) {
                    const auto& Ia = buckets[a][bi];
                    if (Ia.empty()) continue;

                    for (int b = a; b < nm; ++b) {
                        const auto& Jb = buckets[b][bj];
                        if (Jb.empty()) continue;

                        double kab = 0.0;
                        for (int gi : Ia) {
                            const int li = gi - i0;       // [0..ib)
                            const double rn_i = pk.row_n2[gi];
                            for (int gj : Jb) {
                                const int lj = gj - j0;   // [0..jb)
                                // index with LDC = jb (full rectangle)
                                const double cij = Cblk[(size_t)li * jb + lj];
                                const double l2  = rn_i + pk.row_n2[gj] + cij;
                                kab += std::exp(l2 * inv_sigma2);
                            }
                        }
                        // count cross-block self-pairs twice to match full (i,j)+(j,i)
                        if (a == b) kab *= 2.0;

                        kernel[(size_t)a * nm + b] += kab;
                        if (b != a) kernel[(size_t)b * nm + a] += kab;
                    }
                }
            } // j0
        }     // i0

        std::free(Cblk);
    } // labels

}


// helper: flat indices (row-major, last dim fastest)
static inline size_t idx_x1(int a, int j1, int k, int nm1, int max_atoms1, int rep) {
    return (static_cast<size_t>(a) * max_atoms1 + j1) * rep + k;
}
static inline size_t idx_x2(int b, int j2, int k, int nm2, int max_atoms2, int rep) {
    return (static_cast<size_t>(b) * max_atoms2 + j2) * rep + k;
}
static inline size_t idx_q1(int a, int j1, int nm1, int max_atoms1) {
    return static_cast<size_t>(a) * max_atoms1 + j1;
}
static inline size_t idx_q2(int b, int j2, int nm2, int max_atoms2) {
    return static_cast<size_t>(b) * max_atoms2 + j2;
}
// dx1: (nm1, max_atoms1, rep, 3*max_atoms1)
static inline size_t base_dx1(int a, int i1, int nm1, int max_atoms1, int rep) {
    return (((size_t)a * max_atoms1 + i1) * rep) * (3 * (size_t)max_atoms1);
}
// dX2: (nm2, max_atoms2, rep, 3*max_atoms2)
static inline size_t base_dx2(int b, int i2, int nm2, int max_atoms2, int rep) {
    return ( (static_cast<size_t>(b) * max_atoms2 + i2) * rep ) * (3 * static_cast<size_t>(max_atoms2) );
}

// Symmetric helpers
static inline size_t idx_x(int m, int j, int k, int nm, int max_atoms, int rep) {
    return ((size_t)m * max_atoms + j) * rep + (size_t)k;
}
static inline size_t base_dx(int m, int j, int nm, int max_atoms, int rep) {
    return (((size_t)m * max_atoms + j) * rep) * (3 * (size_t)max_atoms);
}


void fatomic_local_gradient_kernel(
    const std::vector<double>& x1,   // (nm1, max_atoms1, rep)
    const std::vector<double>& x2,   // (nm2, max_atoms2, rep)
    const std::vector<double>& dX2,  // (nm2, max_atoms2, rep, 3*max_atoms2)
    const std::vector<int>&    q1,   // (nm1, max_atoms1)
    const std::vector<int>&    q2,   // (nm2, max_atoms2)
    const std::vector<int>&    n1,   // (nm1)
    const std::vector<int>&    n2,   // (nm2)
    int nm1, int nm2,
    int max_atoms1, int max_atoms2,
    int rep_size,
    int naq2,
    double sigma,
    double* kernel_out              // (nm1, naq2)
) {

    // --- validation (unchanged) ---
    if (nm1<=0 || nm2<=0 || max_atoms1<=0 || max_atoms2<=0 || rep_size<=0)
        throw std::invalid_argument("All dims must be positive.");
    if (!std::isfinite(sigma) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive and finite.");
    if (!kernel_out)
        throw std::invalid_argument("kernel_out is null.");

    const size_t x1N = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t dXN = (size_t)nm2 * max_atoms2 * rep_size * (3 * (size_t)max_atoms2);
    const size_t q1N = (size_t)nm1 * max_atoms1;
    const size_t q2N = (size_t)nm2 * max_atoms2;

    if (x1.size() != x1N || x2.size() != x2N) throw std::invalid_argument("x1/x2 size mismatch.");
    if (dX2.size() != dXN) throw std::invalid_argument("dX2 size mismatch.");
    if (q1.size() != q1N || q2.size() != q2N) throw std::invalid_argument("q1/q2 size mismatch.");
    if ((int)n1.size() != nm1 || (int)n2.size() != nm2) throw std::invalid_argument("n1/n2 size mismatch.");

    // per-b offsets and ncols (3 * n2[b])
    std::vector<int> offs2(nm2), ncols_b(nm2);
    int acc = 0;
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::max(0, std::min(n2[b], max_atoms2));
        offs2[b]  = acc;
        ncols_b[b]= 3 * nb;
        acc      += ncols_b[b];
    }
    if (naq2 != acc) throw std::invalid_argument("naq2 != 3*sum(n2)");

    // zero output
    std::fill(kernel_out, kernel_out + (size_t)nm1 * naq2, 0.0);

    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma2  = -1.0 / (      sigma * sigma);

    // ------------------------------------------------------------
    // Build per-label list of (a, j1) only for valid j1 < n1[a]
    // ------------------------------------------------------------
    std::unordered_map<int, std::vector<std::pair<int,int>>> lj1;
    lj1.reserve(128);
    for (int a = 0; a < nm1; ++a) {
        const int na = std::max(0, std::min(n1[a], max_atoms1));
        for (int j1 = 0; j1 < na; ++j1) {
            const int lbl = q1[(size_t)a * max_atoms1 + j1]; // q1(a,j1)
            lj1[lbl].emplace_back(a, j1);
        }
    }

    // Heuristics for batching
    constexpr int BATCH_T      = 8192; // try 256..1024; tune for your machine
    constexpr int T_MIN_GEMM   = 8;   // below this, GEMV often wins
    const     int LDB          = BATCH_T; // row-major leading dimension for D (rep x T)
    const     int LDC          = BATCH_T; // row-major leading dimension for H (ncols x T)

    // ------------------------------------------------------------
    // Parallelize over b: each thread owns a disjoint column block
    // ------------------------------------------------------------
    #pragma omp parallel default(none) shared(x1,x2,dX2,q1,q2,n1,n2, \
                                              nm1,nm2,max_atoms1,max_atoms2,rep_size,naq2, \
                                              kernel_out, lj1, offs2, ncols_b, inv_2sigma2, inv_sigma2)
    {
        // thread-local scratch (aligned, reused)
        double* D_scaled = aligned_alloc_64((size_t)rep_size * LDB);              // (rep_size x LDB)
        double* H        = aligned_alloc_64((size_t)(3*max_atoms2) * LDC);        // (max ncols x LDC)

        #pragma omp for schedule(dynamic)
        for (int b = 0; b < nm2; ++b) {
            const int nb    = ncols_b[b] / 3;
            const int ncols = ncols_b[b];
            if (nb == 0) continue;

            const int out_offset = offs2[b];
            const int lda_rowmaj = 3 * max_atoms2;

            for (int j2 = 0; j2 < nb; ++j2) {
                const int label = q2[(size_t)b * max_atoms2 + j2];
                auto it = lj1.find(label);
                if (it == lj1.end() || it->second.empty()) continue;

                const auto& aj1_list = it->second;

                // dX2 slice for (b,j2): A = dX2(b, j2, :, 0:ncols)
                const double* A = &dX2[ base_dx2(b, j2, nm2, max_atoms2, rep_size) ];

                // Process (a,j1) in tiles
                for (size_t t0 = 0; t0 < aj1_list.size(); t0 += BATCH_T) {
                    const int T = (int)std::min<size_t>(BATCH_T, aj1_list.size() - t0);

                    if (T < T_MIN_GEMM) {
                        // ---- fallback: original GEMV path for tiny batches ----
                        for (int t = 0; t < T; ++t) {
                            const int a  = aj1_list[t0 + t].first;
                            const int j1 = aj1_list[t0 + t].second;

                            // d, l2
                            double l2 = 0.0;
                            // We reuse column t in D_scaled as a temporary buffer for d (no extra alloc)
                            double* dcol = &D_scaled[(size_t)0 * LDB + t]; // start; access [k*LDB + t]
                            for (int k = 0; k < rep_size; ++k) {
                                const double diff =
                                    x1[((size_t)a * max_atoms1 + j1) * rep_size + k] -
                                    x2[((size_t)b * max_atoms2 + j2) * rep_size + k];
                                dcol[(size_t)k * LDB] = diff; // place at k*LDB + t
                                l2 += diff * diff;
                            }
                            const double alpha = std::exp(l2 * inv_2sigma2) * inv_sigma2;

                            cblas_dgemv(CblasRowMajor, CblasTrans,
                                        rep_size, ncols, alpha,
                                        A, lda_rowmaj,
                                        /*x:*/ dcol, LDB,   // stride LDB steps the same column
                                        1.0,
                                        &kernel_out[(size_t)a * naq2 + out_offset], 1);
                        }
                        continue;
                    }

                    // ---- batched DGEMM path ----
                    // 1) Build D_scaled (rep_size x T): column t is alpha_t * d_t
                    for (int t = 0; t < T; ++t) {
                        const int a  = aj1_list[t0 + t].first;
                        const int j1 = aj1_list[t0 + t].second;

                        double l2 = 0.0;
                        for (int k = 0; k < rep_size; ++k) {
                            const double diff =
                                x1[((size_t)a * max_atoms1 + j1) * rep_size + k] -
                                x2[((size_t)b * max_atoms2 + j2) * rep_size + k];
                            D_scaled[(size_t)k * LDB + t] = diff; // D[k,t]
                            l2 += diff * diff;
                        }
                        const double alpha_t = std::exp(l2 * inv_2sigma2) * inv_sigma2;
                        for (int k = 0; k < rep_size; ++k) {
                            D_scaled[(size_t)k * LDB + t] *= alpha_t;
                        }
                    }

                    // 2) GEMM: H = A^T (ncols x rep) * D_scaled (rep x T)  -> H (ncols x T)
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                ncols, T, rep_size,
                                1.0,
                                A, lda_rowmaj,
                                D_scaled, LDB,
                                0.0,
                                H, LDC);

                    // 3) Scatter-add columns of H into kernel_out[a, out_offset : out_offset+ncols]
                    for (int t = 0; t < T; ++t) {
                        const int a  = aj1_list[t0 + t].first;
                        double* kout = &kernel_out[(size_t)a * naq2 + out_offset];
                        const double* hcol = &H[(size_t)t]; // H[r,t] at H[r*LDC + t]
                        // contiguous add
                        for (int r = 0; r < ncols; ++r) {
                            kout[r] += hcol[(size_t)r * LDC];
                        }
                    }
                } // tiles
            } // j2
        } // omp for

        aligned_free_64(D_scaled);
        aligned_free_64(H);
    } // omp parallel
}

// #########################
// # FCHL19 HESSIAN KERNEL #
// #########################
void fgdml_kernel(
    const std::vector<double>& x1,   // (nm1, max_atoms1, rep_size)
    const std::vector<double>& x2,   // (nm2, max_atoms2, rep_size)
    const std::vector<double>& dx1,  // (nm1, max_atoms1, rep_size, 3*max_atoms1)
    const std::vector<double>& dx2,  // (nm2, max_atoms2, rep_size, 3*max_atoms2)
    const std::vector<int>&    q1,   // (nm1, max_atoms1)
    const std::vector<int>&    q2,   // (nm2, max_atoms2)
    const std::vector<int>&    n1,   // (nm1)
    const std::vector<int>&    n2,   // (nm2)
    int nm1, int nm2,
    int max_atoms1, int max_atoms2,
    int rep_size,
    int naq1,                        // must be 3 * sum_a n1[a]
    int naq2,                        // must be 3 * sum_b n2[b]
    double sigma,
    double* kernel_out               // (naq2, naq1), row-major => idx = row * naq1 + col
) {
    // ---- validation ----
    if (nm1<=0 || nm2<=0 || max_atoms1<=0 || max_atoms2<=0 || rep_size<=0)
        throw std::invalid_argument("All dims must be positive.");
    if (!std::isfinite(sigma) || sigma <= 0.0)
        throw std::invalid_argument("sigma must be positive and finite.");
    if (!kernel_out) throw std::invalid_argument("kernel_out is null.");

    const size_t x1N  = (size_t)nm1 * max_atoms1 * rep_size;
    const size_t x2N  = (size_t)nm2 * max_atoms2 * rep_size;
    const size_t dx1N = (size_t)nm1 * max_atoms1 * rep_size * (3 * (size_t)max_atoms1);
    const size_t dx2N = (size_t)nm2 * max_atoms2 * rep_size * (3 * (size_t)max_atoms2);
    const size_t q1N  = (size_t)nm1 * max_atoms1;
    const size_t q2N  = (size_t)nm2 * max_atoms2;

    if (x1.size()!=x1N || x2.size()!=x2N) throw std::invalid_argument("x1/x2 size mismatch.");
    if (dx1.size()!=dx1N || dx2.size()!=dx2N) throw std::invalid_argument("dx1/dx2 size mismatch.");
    if (q1.size()!=q1N || q2.size()!=q2N)     throw std::invalid_argument("q1/q2 size mismatch.");
    if ((int)n1.size()!=nm1 || (int)n2.size()!=nm2) throw std::invalid_argument("n1/n2 size mismatch.");

    // column offsets (set-1) and row offsets (set-2)
    std::vector<int> offs1(nm1), offs2(nm2);
    int sum1 = 0;
    for (int a = 0; a < nm1; ++a) {
        offs1[a] = sum1;
        sum1 += 3 * std::max(0, std::min(n1[a], max_atoms1));
    }
    int sum2 = 0;
    for (int b = 0; b < nm2; ++b) {
        offs2[b] = sum2;
        sum2 += 3 * std::max(0, std::min(n2[b], max_atoms2));
    }
    if (naq1 != sum1) throw std::invalid_argument("naq1 != 3*sum(n1)");
    if (naq2 != sum2) throw std::invalid_argument("naq2 != 3*sum(n2)");

    // zero output
    std::fill(kernel_out, kernel_out + (size_t)naq2 * naq1, 0.0);

    // scalars
    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4  = -1.0 / (      sigma * sigma * sigma * sigma); // (< 0)
    const double sigma2_neg  = - (sigma * sigma);                            // (< 0)

    // ----------------------------------------------------------------
    // Precompute, for each molecule a in set-1, a map: label -> list of i1
    // ----------------------------------------------------------------
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_i1(nm1);
    for (int a = 0; a < nm1; ++a) {
        const int na = std::max(0, std::min(n1[a], max_atoms1));
        auto& m = lab_to_i1[a];
        m.reserve(64);
        for (int i1 = 0; i1 < na; ++i1) {
            m[q1[(size_t)a * max_atoms1 + i1]].push_back(i1);
        }
    }

    // ---- batching parameters (tune) ----
    constexpr int T_MAX    = 512;         // columns per tile for rank-1 batching
    const int     LDT      = T_MAX;       // row-major leading dimension for column tiles

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm2; ++b) {
        const int nb = std::max(0, std::min(n2[b], max_atoms2));
        if (nb == 0) continue;
        const int ncols_b   = 3 * nb;               // rows in K block for this b
        const int lda2      = 3 * max_atoms2;
        const int row_off   = offs2[b];

        // thread-local scratch
        std::vector<double> d(rep_size);
        std::vector<double> Wtile((size_t)ncols_b * LDT); // stores columns: sqrt(expd)*w
        std::vector<double> Vtile((size_t)(3*max_atoms1) * LDT); // over-alloc; we only use first ncols_a rows
        std::vector<double> S1_sum; // allocated per (a) when known

        for (int a = 0; a < nm1; ++a) {
            const int na = std::max(0, std::min(n1[a], max_atoms1));
            if (na == 0) continue;
            const int ncols_a = 3 * na;            // cols in K block for this a
            const int col_off = offs1[a];
            const int lda1    = 3 * max_atoms1;
            if ((int)S1_sum.size() < rep_size * ncols_a) S1_sum.resize((size_t)rep_size * ncols_a);

            // K block pointer (rows for b, cols for a)
            double* Kba = &kernel_out[(size_t)row_off * naq1 + col_off];

            // access map label->i1 once
            const auto& label_i1 = lab_to_i1[a];

            // loop atoms i2 in molecule b
            for (int i2 = 0; i2 < nb; ++i2) {
                const int label = q2[(size_t)b * max_atoms2 + i2];
                auto it = label_i1.find(label);
                if (it == label_i1.end()) continue;
                const auto& i1_list = it->second;
                if (i1_list.empty()) continue;

                // SD2 slice for (b,i2)
                const double* SD2 = &dx2[ base_dx2(b, i2, nm2, max_atoms2, rep_size) ];

                // ------ STATIC TERM: Kba += SD2^T * (sum_i1 expdiag * SD1_i1) ------
                std::fill(S1_sum.begin(), S1_sum.begin() + (size_t)rep_size * ncols_a, 0.0);
                for (int i1 : i1_list) {
                    // distance d = x1(a,i1,:) - x2(b,i2,:)
                    double l2 = 0.0;
                    for (int k = 0; k < rep_size; ++k) {
                        const double diff =
                            x1[idx_x1(a, i1, k, nm1, max_atoms1, rep_size)] -
                            x2[idx_x2(b, i2, k, nm2, max_atoms2, rep_size)];
                        d[k] = diff;
                        l2  += diff * diff;
                    }
                    const double exp_base = std::exp(l2 * inv_2sigma2);
                    const double expd     = inv_sigma4 * exp_base;     // (<0)
                    const double expdiag  = sigma2_neg * expd;         // (>0)

                    // S1_sum += expdiag * SD1(a,i1)
                    const double* SD1 = &dx1[ base_dx1(a, i1, nm1, max_atoms1, rep_size) ];
                    // row-major axpy over matrix: rep_size x ncols_a
                    for (int k = 0; k < rep_size; ++k) {
                        const double* srow = SD1 + (size_t)k * (3 * max_atoms1);
                        double*       trow = S1_sum.data() + (size_t)k * ncols_a;
                        // trow[0:ncols_a] += expdiag * srow[0:ncols_a]
                        cblas_daxpy(ncols_a, expdiag, srow, 1, trow, 1);
                    }
                }
                // One GEMM for the whole static term of this (a,b,i2)
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                            ncols_b, ncols_a, rep_size,
                            1.0,
                            SD2,    lda2,        // SD2^T  via Trans
                            S1_sum.data(), ncols_a,
                            1.0,
                            Kba,    naq1);

                // ------ RANK-1 TERM: sum_i1 expd * (SD2^T d) (SD1^T d)^T ------
                // Batch across i1: build W' and V' columns, each scaled by sqrt(expd)
                for (size_t t0 = 0; t0 < i1_list.size(); t0 += T_MAX) {
                    const int T = (int)std::min<size_t>(T_MAX, i1_list.size() - t0);

                    // build columns
                    for (int t = 0; t < T; ++t) {
                        const int i1 = i1_list[t0 + t];
                        // d and l2
                        double l2 = 0.0;
                        for (int k = 0; k < rep_size; ++k) {
                            const double diff =
                                x1[idx_x1(a, i1, k, nm1, max_atoms1, rep_size)] -
                                x2[idx_x2(b, i2, k, nm2, max_atoms2, rep_size)];
                            d[k] = diff;
                            l2  += diff * diff;
                        }
                        const double exp_base = std::exp(l2 * inv_2sigma2);
                        double expd = inv_sigma4 * exp_base;  // can be negative
                        if (expd < 0.0) expd = -expd;         // sqrt needs non-neg; sign handled via V later
                        const double s = std::sqrt(expd);

                        // w = SD2^T d  -> put into Wtile[:, t]
                        cblas_dgemv(CblasRowMajor, CblasTrans,
                                    rep_size, ncols_b,
                                    s,                  // alpha = sqrt(|expd|)
                                    SD2, lda2,
                                    d.data(), 1,
                                    0.0,
                                    &Wtile[(size_t)t], LDT);  // write with stride LDT

                        // v = SD1^T d  -> put into Vtile[:ncols_a, t]
                        const double* SD1 = &dx1[ base_dx1(a, i1, nm1, max_atoms1, rep_size) ];
                        double alpha_v = s;
                        // If original expd was negative, carry the sign on V
                        if (inv_sigma4 * exp_base < 0.0) alpha_v = -alpha_v;

                        cblas_dgemv(CblasRowMajor, CblasTrans,
                                    rep_size, ncols_a,
                                    alpha_v,
                                    SD1, lda1,
                                    d.data(), 1,
                                    0.0,
                                    &Vtile[(size_t)t], LDT);
                    }

                    // One GEMM for T rank-1s: Kba += Wtile * Vtile^T
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                ncols_b, ncols_a, T,
                                1.0,
                                Wtile.data(), LDT,    // (ncols_b x T)
                                Vtile.data(), LDT,    // (ncols_a x T) as columns ⇒ Trans gives T x ncols_a
                                1.0,
                                Kba, naq1);
                }
            } // i2
        } // a
    } // b
}


void fgdml_kernel_symmetric_lower(
    const std::vector<double>& x,    // (nm, max_atoms, rep_size)
    const std::vector<double>& dx,   // (nm, max_atoms, rep_size, 3*max_atoms)
    const std::vector<int>&    q,    // (nm, max_atoms)
    const std::vector<int>&    n,    // (nm)
    int nm,
    int max_atoms,
    int rep_size,
    int naq,                          // must be 3 * sum_m n[m]
    double sigma,
    double* kernel_out                // (naq, naq) row-major
) {
    // ---- validation ----
    if (nm<=0 || max_atoms<=0 || rep_size<=0) throw std::invalid_argument("dims must be positive");
    if (!std::isfinite(sigma) || sigma<=0.0)   throw std::invalid_argument("sigma must be > 0");
    if (!kernel_out)                            throw std::invalid_argument("kernel_out is null");

    const size_t xN  = (size_t)nm * max_atoms * rep_size;
    const size_t dxN = (size_t)nm * max_atoms * rep_size * (3 * (size_t)max_atoms);
    const size_t qN  = (size_t)nm * max_atoms;

    if (x.size()!=xN)   throw std::invalid_argument("x size mismatch");
    if (dx.size()!=dxN) throw std::invalid_argument("dx size mismatch");
    if (q.size()!=qN)   throw std::invalid_argument("q size mismatch");
    if ((int)n.size()!=nm) throw std::invalid_argument("n size mismatch");

    // Offsets in derivative axis (3 * active atoms per molecule)
    std::vector<int> offs(nm);
    int sum = 0;
    for (int m = 0; m < nm; ++m) {
        offs[m] = sum;
        sum += 3 * std::max(0, std::min(n[m], max_atoms));
    }
    if (naq != sum) throw std::invalid_argument("naq != 3*sum(n)");

    // Zero only lower triangle (optional: zero all for simplicity)
    std::fill(kernel_out, kernel_out + (size_t)naq * naq, 0.0);

    // Scalars
    const double inv_2sigma2 = -1.0 / (2.0 * sigma * sigma);
    const double inv_sigma4  = -1.0 / (      sigma * sigma * sigma * sigma); // < 0
    const double sigma2_neg  = - (sigma * sigma);                            // < 0

    // Label -> indices per molecule
    std::vector<std::unordered_map<int, std::vector<int>>> lab_to_idx(nm);
    for (int a = 0; a < nm; ++a) {
        const int na = std::max(0, std::min(n[a], max_atoms));
        auto& M = lab_to_idx[a];
        M.reserve(64);
        for (int i1 = 0; i1 < na; ++i1) {
            M[q[(size_t)a * max_atoms + i1]].push_back(i1);
        }
    }

    // Scratch memory budget per thread
    constexpr size_t BYTES_BUDGET = 2ull * 1024ull * 1024ull;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < nm; ++b) {
        const int nb = std::max(0, std::min(n[b], max_atoms));
        if (nb == 0) continue;

        const int ncols_b = 3 * nb;
        const int lda_b   = 3 * max_atoms;
        const int row_off = offs[b];

        // worst-case ncols_a across all 'a' is 3*max_atoms
        const int ncols_max = 3 * max_atoms;

        // Choose tile width T within budget:
        // bytes ≈ 8 * [ T*(rep + ncols_b + ncols_max) + rep*ncols_max ] + small
        size_t denom = (size_t)rep_size + (size_t)ncols_b + (size_t)ncols_max;
        size_t bytes_fixed = 8ull * (size_t)rep_size * (size_t)ncols_max; // S_sum
        size_t bytes_left  = (BYTES_BUDGET > bytes_fixed) ? (BYTES_BUDGET - bytes_fixed) : (size_t)1024;
        int T = (int)std::max<size_t>(16, std::min<size_t>(512, bytes_left / (8ull * denom)));
        const int LDT = T;

        // Thread-local aligned scratch
        double* D      = aligned_alloc_64((size_t)rep_size * LDT);        // (rep x T)
        double* W      = aligned_alloc_64((size_t)ncols_b   * LDT);       // (ncols_b x T)
        double* V      = aligned_alloc_64((size_t)ncols_max * LDT);       // (ncols_a x T in first rows)
        double* S_sum  = aligned_alloc_64((size_t)rep_size * ncols_max);  // (rep x ncols_a)
        double* xbv    = aligned_alloc_64((size_t)rep_size);              // x(b,i2,:)

        std::vector<double> sign(T);
        std::vector<double> expdiag(T);

        for (int a = 0; a <= b; ++a) {  // lower triangle only
            const int na = std::max(0, std::min(n[a], max_atoms));
            if (na == 0) continue;

            const int ncols_a = 3 * na;
            const int col_off = offs[a];
            const int lda_a   = 3 * max_atoms;

            // Destination block
            double* Kba = &kernel_out[(size_t)row_off * naq + col_off];

            // For diagonal block, accumulate into a small temp (ncols_b x ncols_a), then scatter lower
            std::vector<double> Cdiag;
            double* Cdst = Kba;
            if (a == b) {
                Cdiag.assign((size_t)ncols_b * ncols_a, 0.0);
                Cdst = Cdiag.data(); // accumulate full square, scatter lower later
            }

            const auto& lab_a = lab_to_idx[a];
            const auto& lab_b = lab_to_idx[b];

            // loop atoms j2 in molecule b
            for (int j2 = 0; j2 < nb; ++j2) {
                const int lbl = q[(size_t)b * max_atoms + j2];
                auto it_a = lab_a.find(lbl);
                if (it_a == lab_a.end() || it_a->second.empty()) continue;
                const auto& i1_list = it_a->second;

                // SD_b and x_b slice (for j2)
                const double* SD_b = &dx[ base_dx(b, j2, nm, max_atoms, rep_size) ];
                for (int k = 0; k < rep_size; ++k)
                    xbv[k] = x[idx_x(b, j2, k, nm, max_atoms, rep_size)];

                // Process i1 in tiles
                for (size_t t0 = 0; t0 < i1_list.size(); t0 += T) {
                    const int Tcur = (int)std::min<size_t>(T, i1_list.size() - t0);

                    // Build D (rep x Tcur): columns d = x(a,i1,:) - x(b,j2,:)
                    for (int t = 0; t < Tcur; ++t) {
                        const int i1 = i1_list[t0 + t];
                        double* dcol = &D[(size_t)0 * LDT + t];
                        double l2 = 0.0;
                        for (int k = 0; k < rep_size; ++k) {
                            const double diff = x[idx_x(a, i1, k, nm, max_atoms, rep_size)] - xbv[k];
                            dcol[(size_t)k * LDT] = diff;
                            l2 += diff * diff;
                        }
                        const double eb = std::exp(l2 * inv_2sigma2);
                        const double e1 = inv_sigma4 * eb;  // may be negative
                        sign[t]    = (e1 >= 0.0) ? 1.0 : -1.0;
                        const double s = std::sqrt(std::abs(e1));
                        expdiag[t] = sigma2_neg * e1;

                        if (s != 1.0) {
                            for (int k = 0; k < rep_size; ++k)
                                dcol[(size_t)k * LDT] *= s;
                        }
                    }

                    // W = SD_b^T * D   (ncols_b x Tcur)
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                ncols_b, Tcur, rep_size,
                                1.0, SD_b, lda_b, D, LDT,
                                0.0, W,    LDT);

                    // V columns: v_t = SD_a(i1)^T * D[:,t], apply sign[t]
                    for (int t = 0; t < Tcur; ++t) {
                        const int i1 = i1_list[t0 + t];
                        const double* SD_a = &dx[ base_dx(a, i1, nm, max_atoms, rep_size) ];

                        cblas_dgemv(CblasRowMajor, CblasTrans,
                                    rep_size, ncols_a,
                                    1.0, SD_a, lda_a,
                                    &D[(size_t)0 * LDT + t], LDT,
                                    0.0, &V[(size_t)0 * LDT + t], LDT);

                        if (sign[t] < 0.0)
                            cblas_dscal(ncols_a, -1.0, &V[(size_t)0 * LDT + t], LDT);
                    }

                    // Rank-1 batch: Cdst += W * V^T
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                ncols_b, ncols_a, Tcur,
                                1.0, W, LDT, V, LDT,
                                1.0, Cdst, (a==b ? ncols_a : naq));

                    // Static term: S_sum = sum_t expdiag[t] * SD_a(i1)
                    for (int k = 0; k < rep_size; ++k) {
                        double* row = &S_sum[(size_t)k * ncols_max];
                        std::fill(row, row + ncols_a, 0.0);
                    }
                    for (int t = 0; t < Tcur; ++t) {
                        const int i1 = i1_list[t0 + t];
                        const double w = expdiag[t];
                        if (w == 0.0) continue;
                        const double* SD_a = &dx[ base_dx(a, i1, nm, max_atoms, rep_size) ];
                        for (int k = 0; k < rep_size; ++k) {
                            const double* srow = SD_a + (size_t)k * (3 * max_atoms);
                            double*       trow = &S_sum[(size_t)k * ncols_max];
                            cblas_daxpy(ncols_a, w, srow, 1, trow, 1);
                        }
                    }
                    // Cdst += SD_b^T * S_sum
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                                ncols_b, ncols_a, rep_size,
                                1.0, SD_b, lda_b,
                                S_sum, ncols_max,
                                1.0, Cdst, (a==b ? ncols_a : naq));
                } // tile
            } // j2

            // Scatter diagonal block's lower triangle only
            if (a == b) {
                for (int r = 0; r < ncols_b; ++r) {
                    double* kout = Kba + (size_t)r * naq;
                    const double* crow = Cdiag.data() + (size_t)r * ncols_a;
                    const int cmax = std::min(r, ncols_a - 1);
                    cblas_daxpy(cmax + 1, 1.0, crow, 1, kout, 1); // add columns 0..cmax
                }
            }
        } // a

        aligned_free_64(D);
        aligned_free_64(W);
        aligned_free_64(V);
        aligned_free_64(S_sum);
        aligned_free_64(xbv);
    } // b
}


} // namespace fchl19
