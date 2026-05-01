// cuda_rff_features.cu — CUDA Random Fourier Features for global descriptors (FP32)

#include "cuda_rff_features.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "curfp.h"

namespace kf::rff_cuda {

namespace {

// Default D-tile size for the gradient path.  Trade-off: smaller = less phase
// memory, more kernel launches.  1024 gives a 4x reduction at D=4096.
static constexpr int D_TILE_DEFAULT = 1024;

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t _err = (call);                                                                  \
        TORCH_CHECK(_err == cudaSuccess, "CUDA error: ", cudaGetErrorString(_err));                \
    } while (0)

#define CUBLAS_CHECK(call)                                                                          \
    do {                                                                                            \
        cublasStatus_t _st = (call);                                                                \
        TORCH_CHECK(_st == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", static_cast<int>(_st));        \
    } while (0)

#define CURFP_CHECK(call)                                                                           \
    do {                                                                                            \
        curfpStatus_t _st = (call);                                                                 \
        TORCH_CHECK(_st == CURFP_STATUS_SUCCESS, "cuRFP error: ", static_cast<int>(_st));          \
    } while (0)

cublasHandle_t s_cublas = nullptr;
curfpHandle_t s_curfp = nullptr;

void ensure_cublas() {
    if (s_cublas == nullptr) {
        CUBLAS_CHECK(cublasCreate(&s_cublas));
    }
}

void ensure_curfp() {
    if (s_curfp == nullptr) {
        CURFP_CHECK(curfpCreate(&s_curfp));
    }
}

void check_cuda_float32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_cuda_int32(const torch::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__global__ void add_bias_cos_scale_kernel(float *Z, const float *b, int N, int D, float scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * D;
    if (idx >= total) return;
    int d = (int)(idx % D);
    Z[idx] = cosf(Z[idx] + b[d]) * scale;
}

__global__ void add_bias_sin_neg_scale_kernel(float *P, const float *b, int N, int D, float scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * D;
    if (idx >= total) return;
    int d = (int)(idx % D);
    P[idx] = sinf(P[idx] + b[d]) * scale;
}

__global__ void fill_rfp_diag_kernel(float *arf, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // TRANSR=N, UPLO=L. Same convention as cuda_global_kernels rfp_potrf.
    long long idx;
    if (n % 2 == 0) {
        int k = n / 2;
        if (i < k) {
            idx = 1LL + (long long)i * (n + 1);
        } else {
            int j = i - k;
            idx = (long long)j * (n + 1);
        }
    } else {
        int n1 = n - n / 2;
        if (i < n1) {
            idx = (long long)i * n;
        } else {
            int j = i - n1;
            idx = (long long)n + (long long)j * n + j;
        }
    }
    arf[idx] = value;
}

__global__ void elemental_phase_kernel(
    float *phase,
    const float *X,
    const int *Q,
    const int *N,
    const float *W,
    const float *b,
    int start,
    int nmol,
    int max_atoms,
    int rep_size,
    int nelements,
    int D,          // full D (stride in W/b)
    int D_tile,     // number of d-columns in this tile
    int d_tile_start // first global d index of this tile
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nmol * max_atoms * D_tile;
    if (idx >= total) return;

    int d_local = (int)(idx % D_tile);
    long long atom_linear = idx / D_tile;
    int a = (int)(atom_linear % max_atoms);
    int m_local = (int)(atom_linear / max_atoms);
    int m = start + m_local;
    int q = Q[m * max_atoms + a];
    if (a >= N[m] || q < 0 || q >= nelements) {
        phase[idx] = 0.0f;
        return;
    }

    int d = d_tile_start + d_local;  // global d index (for W/b indexing)
    const float *x = X + ((long long)m * max_atoms + a) * rep_size;
    const float *w = W + ((long long)q * rep_size * D) + d;
    float s = b[q * D + d];
    for (int r = 0; r < rep_size; ++r) s += x[r] * w[(long long)r * D];
    phase[idx] = s;
}

// ---------------------------------------------------------------------------
// Fused kernel: computes Z directly without materialising the phase buffer.
// Each thread handles one (m_local, d) output and loops over atoms.
// Supports col-major output: when col_major=true writes Z[d, m_local] = d*nmol+m_local.
// ---------------------------------------------------------------------------
__global__ void elemental_features_fused_kernel(
    float *LZ,
    const float *X,
    const int *Q,
    const int *N,
    const float *W,
    const float *b,
    int start,
    int nmol,
    int max_atoms,
    int rep_size,
    int nelements,
    int D,
    float scale,
    bool col_major
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nmol * D;
    if (idx >= total) return;

    int d = (int)(idx % D);
    int m_local = (int)(idx / D);
    int m = start + m_local;
    int n_atoms = N[m];

    float s = 0.0f;
    for (int a = 0; a < n_atoms; ++a) {
        int q = Q[m * max_atoms + a];
        if (q < 0 || q >= nelements) continue;
        const float *x = X + ((long long)m * max_atoms + a) * rep_size;
        const float *w = W + ((long long)q * rep_size * D) + d;
        float phase = b[q * D + d];
        for (int r = 0; r < rep_size; ++r) phase += x[r] * w[(long long)r * D];
        s += cosf(phase) * scale;
    }

    long long out_idx = col_major ? ((long long)d * nmol + m_local) : idx;
    LZ[out_idx] = s;
}

__global__ void elemental_gradient_from_phase_kernel(
    float *G,
    const float *phase,
    const float *dX,
    const int *Q,
    const int *N,
    const int *offsets,
    const float *W,
    int start,
    int nmol,
    int max_atoms,
    int rep_size,
    int nelements,
    int D,
    float scale,
    bool col_major
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_start = offsets[start];
    int chunk_end = offsets[start + nmol];
    int ngrads = chunk_end - chunk_start;
    long long total = (long long)ngrads * D;
    if (idx >= total) return;

    int d = (int)(idx % D);
    int local_g = (int)(idx / D);
    int global_g = chunk_start + local_g;
    int m = start;
    while (m + 1 <= start + nmol && offsets[m + 1] <= global_g) ++m;
    int m_local = m - start;
    int rel = global_g - offsets[m];
    int coord_atom = rel / 3;
    int xyz = rel % 3;
    int n_atoms = N[m];

    // col-major: G[d, local_g] = d * ngrads + local_g
    long long out_idx = col_major ? ((long long)d * ngrads + local_g) : idx;

    if (coord_atom >= n_atoms) {
        G[out_idx] = 0.0f;
        return;
    }

    float s = 0.0f;
    for (int center = 0; center < n_atoms; ++center) {
        int q = Q[m * max_atoms + center];
        if (q < 0 || q >= nelements) continue;
        float coeff = sinf(phase[((long long)m_local * max_atoms + center) * D + d]) * scale;
        const float *w = W + ((long long)q * rep_size * D) + d;
        const float *dx =
            dX + (((long long)m * max_atoms + center) * rep_size * max_atoms * 3) +
            coord_atom * 3 + xyz;
        for (int r = 0; r < rep_size; ++r) {
            s += coeff * w[(long long)r * D] * dx[(long long)r * max_atoms * 3];
        }
    }
    G[out_idx] = s;
}

// ---------------------------------------------------------------------------
// Additional kernels for the memory-efficient rff_predict_force_elemental path
// ---------------------------------------------------------------------------

// Gather sin(phase[m_ci, a_ci, d]) * scale into sin_phase_q[ci, d].
__global__ void gather_sin_phase_kernel(
    float *sin_phase_q,      // (n_centers_q, D)
    const float *phase,      // (nmol, max_atoms, D)
    const float *weights,    // (D,)
    const int *atom_map,     // (n_centers_q, 4): [m_local, a_center, row_offset, n_grads_mol]
    int n_centers_q,
    int max_atoms,
    int D,
    float scale
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_centers_q * D;
    if (idx >= total) return;
    int ci = (int)(idx / D);
    int d  = (int)(idx % D);
    int m_local  = atom_map[ci * 4 + 0];
    int a_center = atom_map[ci * 4 + 1];
    float ph = phase[((long long)m_local * max_atoms + a_center) * D + d];
    sin_phase_q[idx] = sinf(ph) * scale * weights[d];
}

// Scatter-dot: for each row r of gathered_dX, dot with eff[ci(r), :] and atomicAdd into F[g_row].
__global__ void scatter_dot_into_F_kernel(
    float *F,                    // (total_grads,) — output forces (scalar per grad)
    const float *gathered_dX,    // (total_rows_q, rep_size)
    const float *eff,            // (n_centers_q, rep_size)
    const int *atom_map,         // (n_centers_q, 4)
    const int *offsets_chunk,    // (nmol+1,) grad offsets
    int n_centers_q,
    int rep_size,
    int total_rows_q
) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows_q) return;

    // Binary-search for center ci.
    int lo = 0, hi = n_centers_q - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (atom_map[mid * 4 + 2] <= (int)row) lo = mid;
        else hi = mid - 1;
    }
    int ci          = lo;
    int row_offset  = atom_map[ci * 4 + 2];
    int m_local     = atom_map[ci * 4 + 0];
    int n_grads_mol = atom_map[ci * 4 + 3];
    int n_atoms_m   = n_grads_mol / 3;

    int coord_flat = (int)(row - row_offset);
    int coord_atom = coord_flat / 3;
    if (coord_atom >= n_atoms_m) return;

    const float *dxrow  = gathered_dX + row * rep_size;
    const float *effrow = eff         + ci  * rep_size;
    float s = 0.0f;
    for (int r = 0; r < rep_size; ++r) s += dxrow[r] * effrow[r];

    int g_row = offsets_chunk[m_local] + coord_flat;
    atomicAdd(&F[g_row], s);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Optimised elemental gradient path (SGEMM-based)
// ---------------------------------------------------------------------------
//
// For each element q, gather the dX rows for centre-atoms of that element
// into a contiguous buffer, run a single cublasSgemm against W[q], then
// scale by sin(phase) and scatter-add into G[ngrads, D].
//
// Handles variable n_atoms per molecule correctly.
//
// atom_map layout: (n_centers_q, 4) int32
//   col 0: m_local      (chunk-relative molecule index)
//   col 1: a_center     (atom index within molecule)
//   col 2: row_offset   (start row in gathered_dX for this center)
//   col 3: n_grads_mol  (= N[m] * 3 = rows contributed by this center)

// Kernel 1: gather dX rows for one element into a contiguous buffer.
// total_rows_q = sum of atom_map[ci, 3] over all ci.
__global__ void gather_dX_for_element_kernel(
    float *out,              // (total_rows_q, rep_size)
    const float *dX,         // (nmol_total, max_atoms, rep_size, max_atoms, 3)
    const int *atom_map,     // (n_centers_q, 4): [m_local, a_center, row_offset, n_grads_mol]
    int n_centers_q,
    int max_atoms,
    int rep_size,
    int total_rows_q,
    int start                // chunk start (m offset into full nmol_total)
) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= total_rows_q || col >= rep_size) return;

    // Binary-search for the center ci such that row_offset[ci] <= row < row_offset[ci+1].
    int lo = 0, hi = n_centers_q - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (atom_map[mid * 4 + 2] <= (int)row) lo = mid;
        else hi = mid - 1;
    }
    int ci          = lo;
    int row_offset  = atom_map[ci * 4 + 2];
    int m_local     = atom_map[ci * 4 + 0];
    int a_center    = atom_map[ci * 4 + 1];
    int n_grads_mol = atom_map[ci * 4 + 3];
    int n_atoms_m   = n_grads_mol / 3;

    int coord_flat = (int)(row - row_offset);
    int coord_atom = coord_flat / 3;
    int xyz        = coord_flat % 3;
    if (coord_atom >= n_atoms_m) { out[row * rep_size + col] = 0.0f; return; }

    int m_global        = start + m_local;
    int ncoords_per_mol = max_atoms * 3;

    // dX[m_global, a_center, col, coord_atom, xyz]
    long long dX_idx =
        ((long long)m_global * max_atoms + a_center) * (long long)rep_size * ncoords_per_mol
        + col * ncoords_per_mol
        + (long long)coord_atom * 3 + xyz;

    out[row * rep_size + col] = dX[dX_idx];
}

// Kernel 2: scale rows of P by sin(phase) and scatter-add into G.
// D_tile: number of d-columns in current tile (== D when not tiling).
// d_tile_start: first global d index of this tile (== 0 when not tiling).
// When tiling: phase and P have stride D_tile; G col-major index uses d_tile_start + d.
__global__ void scatter_add_sin_phase_kernel(
    float *G,                // (ngrads, D_full) row-major  OR  (D_full, ngrads) col-major
    const float *P,          // (total_rows_q, D_tile)
    const float *phase,      // (nc, max_atoms, D_tile)
    const int *atom_map,     // (n_centers_q, 4): [m_local, a_center, row_offset, n_grads_mol]
    const int *offsets_chunk,// (nc+1,) grad offsets relative to chunk
    int n_centers_q,
    int max_atoms,
    int total_rows_q,
    int D_tile,
    int D_full,
    int ngrads,
    float scale,
    bool col_major,
    int d_tile_start
) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long d   = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= total_rows_q || d >= D_tile) return;

    // Binary-search for center ci.
    int lo = 0, hi = n_centers_q - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (atom_map[mid * 4 + 2] <= (int)row) lo = mid;
        else hi = mid - 1;
    }
    int ci          = lo;
    int row_offset  = atom_map[ci * 4 + 2];
    int m_local     = atom_map[ci * 4 + 0];
    int a_center    = atom_map[ci * 4 + 1];
    int n_grads_mol = atom_map[ci * 4 + 3];
    int n_atoms_m   = n_grads_mol / 3;

    int coord_flat = (int)(row - row_offset);
    int coord_atom = coord_flat / 3;
    int xyz        = coord_flat % 3;
    if (coord_atom >= n_atoms_m) return;

    // phase stride is D_tile (local tile dimension)
    float coeff = sinf(phase[((long long)m_local * max_atoms + a_center) * D_tile + d]) * scale;
    int g_row   = offsets_chunk[m_local] + coord_atom * 3 + xyz;
    long long d_global = d_tile_start + d;
    long long out_idx = col_major ? (d_global * (long long)ngrads + g_row)
                                  : (g_row * (long long)D_full + d_global);
    atomicAdd(&G[out_idx], coeff * P[row * D_tile + d]);
}

// Build atom_map on CPU for element q in a chunk [start, start+nc).
// Returns flat int32 vector of 4 ints per center.
// Sets total_rows_out = total gathered rows for this element.
static std::vector<int> build_atom_map(
    const int *Q_cpu,
    const int *N_cpu,
    int start, int nc,
    int max_atoms, int q_elem,
    int &total_rows_out
) {
    std::vector<int> map;
    map.reserve(nc * max_atoms * 4);
    int row_offset = 0;
    for (int m_local = 0; m_local < nc; ++m_local) {
        int m       = start + m_local;
        int n_atoms = N_cpu[m];
        for (int a = 0; a < n_atoms; ++a) {
            if (Q_cpu[m * max_atoms + a] == q_elem) {
                map.push_back(m_local);
                map.push_back(a);
                map.push_back(row_offset);
                map.push_back(n_atoms * 3);
                row_offset += n_atoms * 3;
            }
        }
    }
    total_rows_out = row_offset;
    return map;
}

torch::Tensor rff_gradient_elemental_chunk_sgemm(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc,
    const int *Q_cpu,
    const int *N_cpu,
    const int *offsets_cpu_ptr,
    bool col_major = false
) {
    int max_atoms = (int)X.size(1);
    int rep_size  = (int)X.size(2);
    int nelements = (int)W.size(0);
    int D         = (int)b.size(1);

    int chunk_start = offsets_cpu_ptr[start];
    int chunk_end   = offsets_cpu_ptr[start + nc];
    int ngrads      = chunk_end - chunk_start;

    // row-major: (ngrads, D);  col-major: (D, ngrads) — same bytes
    auto G = col_major ? torch::zeros({D, ngrads}, X.options())
                       : torch::zeros({ngrads, D}, X.options());

    // Build chunk-local mol offsets (0-based within chunk).
    std::vector<int> off_chunk(nc + 1);
    for (int i = 0; i <= nc; ++i) off_chunk[i] = offsets_cpu_ptr[start + i] - chunk_start;
    auto off_chunk_dev = torch::empty({nc + 1}, Q.options());
    CUDA_CHECK(cudaMemcpy(off_chunk_dev.data_ptr<int>(), off_chunk.data(),
                          (size_t)(nc + 1) * sizeof(int), cudaMemcpyHostToDevice));

    const float scale = std::sqrt(2.0f / (float)D);
    const float one   = 1.0f;
    const float zero  = 0.0f;
    const int D_tile  = std::min(D, D_TILE_DEFAULT);

    // Build atom maps once (shared across D-tiles and elements).
    std::vector<std::vector<int>> atom_maps(nelements);
    std::vector<int> total_rows(nelements, 0);
    for (int q = 0; q < nelements; ++q) {
        atom_maps[q] = build_atom_map(Q_cpu, N_cpu, start, nc, max_atoms, q, total_rows[q]);
    }

    // Outer loop: D-tiles.
    for (int d0 = 0; d0 < D; d0 += D_tile) {
        int dt = std::min(D_tile, D - d0);  // actual tile width

        // Allocate phase tile: (nc, max_atoms, dt)
        auto phase_tile = torch::empty({nc, max_atoms, dt}, X.options());
        {
            int threads = 256;
            int blocks = (int)(((long long)nc * max_atoms * dt + threads - 1) / threads);
            elemental_phase_kernel<<<blocks, threads>>>(
                phase_tile.data_ptr<float>(),
                X.data_ptr<float>(),
                Q.data_ptr<int>(),
                N.data_ptr<int>(),
                W.data_ptr<float>(),
                b.data_ptr<float>(),
                start, nc, max_atoms, rep_size, nelements,
                D,   // full D (stride in W/b)
                dt,  // D_tile
                d0   // d_tile_start
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // Inner loop: elements.
        for (int q = 0; q < nelements; ++q) {
            int total_rows_q = total_rows[q];
            int n_centers_q  = (int)(atom_maps[q].size() / 4);
            if (n_centers_q == 0 || total_rows_q == 0) continue;

            auto atom_map_dev = torch::empty({n_centers_q * 4}, Q.options());
            CUDA_CHECK(cudaMemcpy(atom_map_dev.data_ptr<int>(), atom_maps[q].data(),
                                  (size_t)atom_maps[q].size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // Step 1: gather dX rows → gathered_dX (total_rows_q, rep_size)
            auto gathered_dX = torch::empty({total_rows_q, rep_size}, X.options());
            {
                dim3 block(32, 8);
                dim3 grid(
                    (unsigned int)((total_rows_q + block.x - 1) / block.x),
                    (unsigned int)((rep_size + block.y - 1) / block.y)
                );
                gather_dX_for_element_kernel<<<grid, block>>>(
                    gathered_dX.data_ptr<float>(),
                    dX.data_ptr<float>(),
                    atom_map_dev.data_ptr<int>(),
                    n_centers_q,
                    max_atoms,
                    rep_size,
                    total_rows_q,
                    start
                );
                CUDA_CHECK(cudaGetLastError());
            }

            // Step 2: SGEMM  gathered_dX (total_rows_q, rep_size) @ W[q][:,d0:d0+dt]
            //   Only the tile columns d0..d0+dt of W[q] are needed.
            //   W[q] is (rep_size, D) row-major = col-major (D, rep_size), lda=D.
            //   We use lda=D (full stride) but only emit dt output columns.
            auto P = torch::empty({total_rows_q, dt}, X.options());
            CUBLAS_CHECK(cublasSgemm(
                s_cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dt,           // m = tile width
                total_rows_q, // n
                rep_size,     // k
                &one,
                W.data_ptr<float>() + (long long)q * rep_size * D + d0,
                D,            // lda (full stride)
                gathered_dX.data_ptr<float>(),
                rep_size,     // ldb
                &zero,
                P.data_ptr<float>(),
                dt            // ldc
            ));

            // Step 3: scatter-add P * sin(phase_tile) into G (tile columns d0..d0+dt)
            {
                dim3 block(32, 8);
                dim3 grid(
                    (unsigned int)((total_rows_q + block.x - 1) / block.x),
                    (unsigned int)((dt + block.y - 1) / block.y)
                );
                scatter_add_sin_phase_kernel<<<grid, block>>>(
                    G.data_ptr<float>(),
                    P.data_ptr<float>(),
                    phase_tile.data_ptr<float>(),
                    atom_map_dev.data_ptr<int>(),
                    off_chunk_dev.data_ptr<int>(),
                    n_centers_q,
                    max_atoms,
                    total_rows_q,
                    dt,       // D_tile
                    D,        // D_full
                    ngrads,
                    scale,
                    col_major,
                    d0        // d_tile_start
                );
                CUDA_CHECK(cudaGetLastError());
            }
        }  // elements
    }  // D-tiles
    return G;
}

torch::Tensor make_offsets_cuda(const torch::Tensor &N) {
    int nmol = (int)N.size(0);
    std::vector<int> sizes(nmol);
    CUDA_CHECK(cudaMemcpy(sizes.data(), N.data_ptr<int>(), (size_t)nmol * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> offsets(nmol + 1, 0);
    for (int i = 0; i < nmol; ++i) offsets[i + 1] = offsets[i] + 3 * sizes[i];
    auto offsets_cuda = torch::empty({nmol + 1}, N.options());
    CUDA_CHECK(cudaMemcpy(offsets_cuda.data_ptr<int>(), offsets.data(), (size_t)(nmol + 1) * sizeof(int), cudaMemcpyHostToDevice));
    return offsets_cuda;
}

torch::Tensor rff_features_chunk(
    const torch::Tensor &X, const torch::Tensor &W, const torch::Tensor &b, int start, int nc
) {
    ensure_cublas();

    int M = (int)X.size(1);
    int D = (int)b.size(0);
    auto Z = torch::empty({nc, D}, X.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major (nc, M) @ (M, D) -> row-major (nc, D). Interpreted as
    // col-major Z^T(D,nc) = W_rowmajor_as_colmajor(D,M) * X^T(M,nc).
    CUBLAS_CHECK(cublasSgemm(
        s_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D,
        nc,
        M,
        &alpha,
        W.data_ptr<float>(),
        D,
        X.data_ptr<float>() + (long long)start * M,
        M,
        &beta,
        Z.data_ptr<float>(),
        D
    ));

    const float scale = std::sqrt(2.0f / (float)D);
    int threads = 256;
    int blocks = (int)(((long long)nc * D + threads - 1) / threads);
    add_bias_cos_scale_kernel<<<blocks, threads>>>(Z.data_ptr<float>(), b.data_ptr<float>(), nc, D, scale);
    CUDA_CHECK(cudaGetLastError());
    return Z;
}

torch::Tensor rff_gradient_chunk(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc
) {
    ensure_cublas();

    int M = (int)X.size(1);
    int ncoords = (int)dX.size(1);
    int D = (int)b.size(0);
    int ngrads = nc * ncoords;

    auto phase = torch::empty({nc, D}, X.options());
    auto G = torch::empty({ngrads, D}, X.options());

    const float one = 1.0f;
    const float zero = 0.0f;

    // phase = X_chunk @ W + b, row-major (nc, D).
    CUBLAS_CHECK(cublasSgemm(
        s_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D,
        nc,
        M,
        &one,
        W.data_ptr<float>(),
        D,
        X.data_ptr<float>() + (long long)start * M,
        M,
        &zero,
        phase.data_ptr<float>(),
        D
    ));

    const float scale = -std::sqrt(2.0f / (float)D);
    int threads = 256;
    int blocks = (int)(((long long)nc * D + threads - 1) / threads);
    add_bias_sin_neg_scale_kernel<<<blocks, threads>>>(
        phase.data_ptr<float>(), b.data_ptr<float>(), nc, D, scale
    );
    CUDA_CHECK(cudaGetLastError());

    // G = dX_chunk @ W, where dX chunk is row-major (ngrads, M).
    CUBLAS_CHECK(cublasSgemm(
        s_cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D,
        ngrads,
        M,
        &one,
        W.data_ptr<float>(),
        D,
        dX.data_ptr<float>() + (long long)start * ncoords * M,
        M,
        &zero,
        G.data_ptr<float>(),
        D
    ));

    auto phase_view = phase.view({nc, 1, D}).expand({nc, ncoords, D}).reshape({ngrads, D});
    G.mul_(phase_view);
    return G;
}

torch::Tensor rff_features_elemental_chunk(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc,
    bool col_major = false
) {
    int max_atoms = (int)X.size(1);
    int rep_size = (int)X.size(2);
    int nelements = (int)W.size(0);
    int D = (int)b.size(1);
    // row-major: (nc, D);  col-major: (D, nc) — same bytes, different logical shape
    auto LZ = col_major ? torch::empty({D, nc}, X.options())
                        : torch::empty({nc, D}, X.options());
    int threads = 256;
    int lz_blocks = (int)(((long long)nc * D + threads - 1) / threads);
    const float scale = std::sqrt(2.0f / (float)D);
    elemental_features_fused_kernel<<<lz_blocks, threads>>>(
        LZ.data_ptr<float>(),
        X.data_ptr<float>(),
        Q.data_ptr<int>(),
        N.data_ptr<int>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        start,
        nc,
        max_atoms,
        rep_size,
        nelements,
        D,
        scale,
        col_major
    );
    CUDA_CHECK(cudaGetLastError());
    return LZ;
}

torch::Tensor rff_gradient_elemental_chunk(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc,
    const int *Q_cpu,
    const int *N_cpu,
    const int *offsets_cpu_ptr,
    bool col_major = false
) {
    return rff_gradient_elemental_chunk_sgemm(
        X, dX, Q, N, W, b,
        start, nc, Q_cpu, N_cpu, offsets_cpu_ptr, col_major
    );
}

// Legacy overload (no CPU mirrors): used by rff_full_gramian_elemental_rfp_cuda.
// Copies Q, N to host then delegates.
torch::Tensor rff_gradient_elemental_chunk(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int start,
    int nc
) {
    int nmol_total = (int)N.size(0);
    int max_atoms  = (int)X.size(1);
    std::vector<int> Q_cpu((size_t)nmol_total * max_atoms);
    std::vector<int> N_cpu((size_t)nmol_total);
    CUDA_CHECK(cudaMemcpy(Q_cpu.data(), Q.data_ptr<int>(),
                          (size_t)nmol_total * max_atoms * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(N_cpu.data(), N.data_ptr<int>(),
                          (size_t)nmol_total * sizeof(int), cudaMemcpyDeviceToHost));
    // Build offsets on CPU
    std::vector<int> off_cpu(nmol_total + 1, 0);
    for (int i = 0; i < nmol_total; ++i) off_cpu[i + 1] = off_cpu[i] + 3 * N_cpu[i];
    return rff_gradient_elemental_chunk(
        X, dX, Q, N, W, b, start, nc,
        Q_cpu.data(), N_cpu.data(), off_cpu.data()
    );
}

}  // namespace

torch::Tensor rff_features_cuda(
    const torch::Tensor &X, const torch::Tensor &W, const torch::Tensor &b
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(X.size(0) > 0 && X.size(1) > 0 && b.size(0) > 0, "zero dimension");

    return rff_features_chunk(X, W, b, 0, (int)X.size(0));
}

std::tuple<torch::Tensor, torch::Tensor> rff_gramian_symm_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(Y.dim() == 1, "Y must be 1D (N,)");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(Y.size(0) == X.size(0), "Y.shape[0] must equal X.shape[0]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();
    ensure_curfp();

    int N = (int)X.size(0);
    int D = (int)b.size(0);
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());

    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < N; cs += chunk_size) {
        int nc = std::min(chunk_size, N - cs);
        auto Z = rff_features_chunk(X, W, b, cs, nc);

        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }

        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));

        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));

        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
    std::fprintf(
        stderr,
        "  [rff_gramian_symm_rfp]        chunks=%d  chunk_size=%d  D=%d  total=%.1f ms\n",
        (N + chunk_size - 1) / chunk_size,
        chunk_size,
        D,
        total_ms
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));

    return {ZtZ_rfp, ZtY};
}

std::tuple<torch::Tensor, torch::Tensor> rff_full_gramian_symm_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    const torch::Tensor &F,
    int energy_chunk,
    int force_chunk
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    check_cuda_float32(F, "F");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(dX.dim() == 3, "dX must be 3D (N, ncoords, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(Y.dim() == 1, "Y must be 1D (N,)");
    TORCH_CHECK(F.dim() == 1, "F must be 1D (N*ncoords,)");
    TORCH_CHECK(dX.size(0) == X.size(0), "dX.shape[0] must equal X.shape[0]");
    TORCH_CHECK(dX.size(2) == X.size(1), "dX.shape[2] must equal X.shape[1]");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(Y.size(0) == X.size(0), "Y.shape[0] must equal X.shape[0]");
    TORCH_CHECK(F.size(0) == X.size(0) * dX.size(1), "F.shape[0] must equal N*ncoords");
    TORCH_CHECK(energy_chunk > 0 && force_chunk > 0, "chunk sizes must be positive");

    ensure_cublas();
    ensure_curfp();

    int N = (int)X.size(0);
    int D = (int)b.size(0);
    int ncoords = (int)dX.size(1);
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());

    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_energy, ev_force;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_energy));
    CUDA_CHECK(cudaEventCreate(&ev_force));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < N; cs += energy_chunk) {
        int nc = std::min(energy_chunk, N - cs);
        auto Z = rff_features_chunk(X, W, b, cs, nc);

        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }

        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));

        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));

        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_energy));

    for (int cs = 0; cs < N; cs += force_chunk) {
        int nc = std::min(force_chunk, N - cs);
        int ngrads = nc * ncoords;
        auto G = rff_gradient_chunk(X, dX, W, b, cs, nc);

        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            &one,
            ZtZ_rfp.data_ptr<float>()
        ));

        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            F.data_ptr<float>() + (long long)cs * ncoords,
            1,
            &one,
            ZtY.data_ptr<float>(),
            1
        ));
    }

    CUDA_CHECK(cudaEventRecord(ev_force));
    CUDA_CHECK(cudaEventSynchronize(ev_force));
    float energy_ms = 0.0f;
    float force_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&energy_ms, ev_start, ev_energy));
    CUDA_CHECK(cudaEventElapsedTime(&force_ms, ev_energy, ev_force));
    std::fprintf(
        stderr,
        "  [rff_full_gramian_symm_rfp]   energy=%.1f ms  force=%.1f ms  total=%.1f ms\n",
        energy_ms,
        force_ms,
        energy_ms + force_ms
    );
    std::fprintf(
        stderr,
        "  [rff_full_gramian_symm_rfp]   N=%d  ncoords=%d  M=%d  D=%d  "
        "energy_chunk=%d  force_chunk=%d\n",
        N,
        ncoords,
        (int)X.size(1),
        D,
        energy_chunk,
        force_chunk
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_energy));
    CUDA_CHECK(cudaEventDestroy(ev_force));

    return {ZtZ_rfp, ZtY};
}

torch::Tensor rff_predict_energy_cuda(
    const torch::Tensor &X,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D (D,)");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(weights.size(0) == b.size(0), "weights.shape[0] must equal b.shape[0]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();

    int N = (int)X.size(0);
    int D = (int)b.size(0);
    auto E = torch::empty({N}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;

    for (int cs = 0; cs < N; cs += chunk_size) {
        int nc = std::min(chunk_size, N - cs);
        auto Z = rff_features_chunk(X, W, b, cs, nc);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            nc,
            &one,
            Z.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            E.data_ptr<float>() + cs,
            1
        ));
    }

    return E;
}

torch::Tensor rff_predict_force_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(X.dim() == 2, "X must be 2D (N, M)");
    TORCH_CHECK(dX.dim() == 3, "dX must be 3D (N, ncoords, M)");
    TORCH_CHECK(W.dim() == 2, "W must be 2D (M, D)");
    TORCH_CHECK(b.dim() == 1, "b must be 1D (D,)");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D (D,)");
    TORCH_CHECK(dX.size(0) == X.size(0), "dX.shape[0] must equal X.shape[0]");
    TORCH_CHECK(dX.size(2) == X.size(1), "dX.shape[2] must equal X.shape[1]");
    TORCH_CHECK(W.size(0) == X.size(1), "W.shape[0] must equal X.shape[1]");
    TORCH_CHECK(W.size(1) == b.size(0), "W.shape[1] must equal b.shape[0]");
    TORCH_CHECK(weights.size(0) == b.size(0), "weights.shape[0] must equal b.shape[0]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();

    int N = (int)X.size(0);
    int ncoords = (int)dX.size(1);
    int D = (int)b.size(0);
    auto F = torch::empty({N * ncoords}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;

    for (int cs = 0; cs < N; cs += chunk_size) {
        int nc = std::min(chunk_size, N - cs);
        int ngrads = nc * ncoords;
        auto G = rff_gradient_chunk(X, dX, W, b, cs, nc);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            F.data_ptr<float>() + (long long)cs * ncoords,
            1
        ));
    }

    return F;
}

torch::Tensor rff_features_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 3, "X must be 3D (nmol, max_atoms, rep_size)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D (nmol, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1D (nmol,)");
    TORCH_CHECK(W.dim() == 3, "W must be 3D (nelements, rep_size, D)");
    TORCH_CHECK(b.dim() == 2, "b must be 2D (nelements, D)");
    TORCH_CHECK(Q.size(0) == X.size(0) && Q.size(1) == X.size(1), "Q shape mismatch");
    TORCH_CHECK(N.size(0) == X.size(0), "N.shape[0] must equal X.shape[0]");
    TORCH_CHECK(W.size(1) == X.size(2), "W.shape[1] must equal X.shape[2]");
    TORCH_CHECK(W.size(0) == b.size(0) && W.size(2) == b.size(1), "W/b shape mismatch");

    return rff_features_elemental_chunk(X, Q, N, W, b, 0, (int)X.size(0));
}

// Same as rff_features_elemental_cuda but returns Z in column-major layout (D, nmol).
// Used internally by the SVD/QR solver path to avoid an extra transpose.
torch::Tensor rff_features_elemental_col_major_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 3, "X must be 3D (nmol, max_atoms, rep_size)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D (nmol, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1D (nmol,)");
    TORCH_CHECK(W.dim() == 3, "W must be 3D (nelements, rep_size, D)");
    TORCH_CHECK(b.dim() == 2, "b must be 2D (nelements, D)");
    TORCH_CHECK(Q.size(0) == X.size(0) && Q.size(1) == X.size(1), "Q shape mismatch");
    TORCH_CHECK(N.size(0) == X.size(0), "N.shape[0] must equal X.shape[0]");
    TORCH_CHECK(W.size(1) == X.size(2), "W.shape[1] must equal X.shape[2]");
    TORCH_CHECK(W.size(0) == b.size(0) && W.size(2) == b.size(1), "W/b shape mismatch");

    return rff_features_elemental_chunk(X, Q, N, W, b, 0, (int)X.size(0), /*col_major=*/true);
}

// Materialise the full gradient feature matrix G (total_naq x D) where
// total_naq = sum_i(3 * N[i]).  Rows are laid out molecule-by-molecule in the
// same order as the compact force vector used elsewhere in the codebase.
torch::Tensor rff_gradient_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 3, "X must be 3D (nmol, max_atoms, rep_size)");
    TORCH_CHECK(dX.dim() == 5, "dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D (nmol, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1D (nmol,)");
    TORCH_CHECK(W.dim() == 3, "W must be 3D (nelements, rep_size, D)");
    TORCH_CHECK(b.dim() == 2, "b must be 2D (nelements, D)");
    TORCH_CHECK(Q.size(0) == X.size(0) && Q.size(1) == X.size(1), "Q shape mismatch");
    TORCH_CHECK(N.size(0) == X.size(0), "N.shape[0] must equal X.shape[0]");
    TORCH_CHECK(W.size(1) == X.size(2), "W.shape[1] must equal X.shape[2]");
    TORCH_CHECK(W.size(0) == b.size(0) && W.size(2) == b.size(1), "W/b shape mismatch");

    ensure_cublas();

    int nmol   = (int)X.size(0);
    int D      = (int)b.size(1);

    // Build offsets on CPU: off[i] = sum_{j<i} 3*N[j]
    std::vector<int> N_cpu((size_t)nmol);
    CUDA_CHECK(cudaMemcpy(
        N_cpu.data(), N.data_ptr<int>(), (size_t)nmol * sizeof(int), cudaMemcpyDeviceToHost
    ));
    std::vector<int> off_cpu(nmol + 1, 0);
    for (int i = 0; i < nmol; ++i) off_cpu[i + 1] = off_cpu[i] + 3 * N_cpu[i];
    int total_naq = off_cpu[nmol];

    int max_atoms = (int)X.size(1);
    std::vector<int> Q_cpu((size_t)nmol * max_atoms);
    CUDA_CHECK(cudaMemcpy(
        Q_cpu.data(), Q.data_ptr<int>(), (size_t)nmol * max_atoms * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    auto G_full = torch::empty({total_naq, D}, X.options());

    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc      = std::min(chunk_size, nmol - cs);
        auto G_chunk = rff_gradient_elemental_chunk(
            X, dX, Q, N, W, b, cs, nc, Q_cpu.data(), N_cpu.data(), off_cpu.data()
        );
        int ngrads   = (int)G_chunk.size(0);
        // Copy chunk rows into the correct position in G_full
        G_full.narrow(0, off_cpu[cs], ngrads).copy_(G_chunk);
    }

    return G_full;
}

// Same as rff_gradient_elemental_cuda but returns G in column-major layout (D, total_naq).
// Used internally by the SVD/QR solver path to avoid an extra transpose.
torch::Tensor rff_gradient_elemental_col_major_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    TORCH_CHECK(X.dim() == 3, "X must be 3D (nmol, max_atoms, rep_size)");
    TORCH_CHECK(dX.dim() == 5, "dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D (nmol, max_atoms)");
    TORCH_CHECK(N.dim() == 1, "N must be 1D (nmol,)");
    TORCH_CHECK(W.dim() == 3, "W must be 3D (nelements, rep_size, D)");
    TORCH_CHECK(b.dim() == 2, "b must be 2D (nelements, D)");
    TORCH_CHECK(Q.size(0) == X.size(0) && Q.size(1) == X.size(1), "Q shape mismatch");
    TORCH_CHECK(N.size(0) == X.size(0), "N.shape[0] must equal X.shape[0]");
    TORCH_CHECK(W.size(1) == X.size(2), "W.shape[1] must equal X.shape[2]");
    TORCH_CHECK(W.size(0) == b.size(0) && W.size(2) == b.size(1), "W/b shape mismatch");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    ensure_cublas();

    int nmol   = (int)X.size(0);
    int D      = (int)b.size(1);

    // Build offsets and Q/N CPU mirrors once.
    std::vector<int> N_cpu((size_t)nmol);
    CUDA_CHECK(cudaMemcpy(
        N_cpu.data(), N.data_ptr<int>(), (size_t)nmol * sizeof(int), cudaMemcpyDeviceToHost
    ));
    std::vector<int> off_cpu(nmol + 1, 0);
    for (int i = 0; i < nmol; ++i) off_cpu[i + 1] = off_cpu[i] + 3 * N_cpu[i];
    int total_naq = off_cpu[nmol];

    int max_atoms = (int)X.size(1);
    std::vector<int> Q_cpu((size_t)nmol * max_atoms);
    CUDA_CHECK(cudaMemcpy(
        Q_cpu.data(), Q.data_ptr<int>(), (size_t)nmol * max_atoms * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    // G_full is (D, total_naq) — col-major view of the gradient matrix.
    auto G_full = torch::empty({D, total_naq}, X.options());

    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc      = std::min(chunk_size, nmol - cs);
        // G_chunk is (D, ngrads_chunk) col-major.
        auto G_chunk = rff_gradient_elemental_chunk(
            X, dX, Q, N, W, b, cs, nc,
            Q_cpu.data(), N_cpu.data(), off_cpu.data(), /*col_major=*/true
        );
        int ngrads = (int)G_chunk.size(1);  // col-major: dim 1 = ngrads
        // Copy chunk columns into the correct position in G_full.
        G_full.narrow(1, off_cpu[cs], ngrads).copy_(G_chunk);
    }

    return G_full;
}

std::tuple<torch::Tensor, torch::Tensor> rff_gramian_elemental_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    TORCH_CHECK(Y.dim() == 1 && Y.size(0) == X.size(0), "Y must be 1D (nmol,)");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    ensure_cublas();
    ensure_curfp();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());
    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc = std::min(chunk_size, nmol - cs);
        auto LZ = rff_features_elemental_chunk(X, Q, N, W, b, cs, nc);
        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }
        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));
        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_end));
    std::fprintf(
        stderr,
        "  [rff_gramian_elemental_rfp]   chunks=%d  chunk_size=%d  D=%d  total=%.1f ms\n",
        (nmol + chunk_size - 1) / chunk_size,
        chunk_size,
        D,
        total_ms
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    return {ZtZ_rfp, ZtY};
}

torch::Tensor rff_predict_energy_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(weights.dim() == 1 && weights.size(0) == b.size(1), "weights must be 1D (D,)");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    ensure_cublas();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    auto E = torch::empty({nmol}, X.options());
    const float one = 1.0f;
    const float zero = 0.0f;
    for (int cs = 0; cs < nmol; cs += chunk_size) {
        int nc = std::min(chunk_size, nmol - cs);
        auto LZ = rff_features_elemental_chunk(X, Q, N, W, b, cs, nc);
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_T,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            weights.data_ptr<float>(),
            1,
            &zero,
            E.data_ptr<float>() + cs,
            1
        ));
    }
    return E;
}

std::tuple<torch::Tensor, torch::Tensor> rff_full_gramian_elemental_rfp_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &Y,
    const torch::Tensor &F,
    int energy_chunk,
    int force_chunk
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(Y, "Y");
    check_cuda_float32(F, "F");
    TORCH_CHECK(dX.dim() == 5, "dX must be 5D (nmol, max_atoms, rep_size, max_atoms, 3)");
    TORCH_CHECK(Y.dim() == 1 && Y.size(0) == X.size(0), "Y must be 1D (nmol,)");
    TORCH_CHECK(F.dim() == 1, "F must be 1D");
    TORCH_CHECK(energy_chunk > 0 && force_chunk > 0, "chunk sizes must be positive");
    ensure_cublas();
    ensure_curfp();

    int nmol = (int)X.size(0);
    int D = (int)b.size(1);
    auto offsets = make_offsets_cuda(N);
    auto offsets_cpu = offsets.cpu();
    long long nt = (long long)D * (D + 1) / 2;
    auto ZtZ_rfp = torch::empty({nt}, X.options());
    auto ZtY = torch::zeros({D}, X.options());
    const float one = 1.0f;
    float beta = 0.0f;

    cudaEvent_t ev_start, ev_energy, ev_force;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_energy));
    CUDA_CHECK(cudaEventCreate(&ev_force));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int cs = 0; cs < nmol; cs += energy_chunk) {
        int nc = std::min(energy_chunk, nmol - cs);
        auto LZ = rff_features_elemental_chunk(X, Q, N, W, b, cs, nc);
        if (cs == 0) {
            fill_rfp_diag_kernel<<<(D + 255) / 256, 256>>>(ZtZ_rfp.data_ptr<float>(), D, 0.0f);
            CUDA_CHECK(cudaGetLastError());
        }
        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            &beta,
            ZtZ_rfp.data_ptr<float>()
        ));
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            nc,
            &one,
            LZ.data_ptr<float>(),
            D,
            Y.data_ptr<float>() + cs,
            1,
            &beta,
            ZtY.data_ptr<float>(),
            1
        ));
        beta = 1.0f;
    }

    CUDA_CHECK(cudaEventRecord(ev_energy));

    for (int cs = 0; cs < nmol; cs += force_chunk) {
        int nc = std::min(force_chunk, nmol - cs);
        auto G = rff_gradient_elemental_chunk(X, dX, Q, N, W, b, cs, nc);
        int ngrads = (int)G.size(0);
        CURFP_CHECK(curfpSsfrk(
            s_curfp,
            CURFP_OP_N,
            CURFP_FILL_MODE_LOWER,
            CURFP_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            &one,
            ZtZ_rfp.data_ptr<float>()
        ));
        CUBLAS_CHECK(cublasSgemv(
            s_cublas,
            CUBLAS_OP_N,
            D,
            ngrads,
            &one,
            G.data_ptr<float>(),
            D,
            F.data_ptr<float>() + offsets_cpu.data_ptr<int>()[cs],
            1,
            &one,
            ZtY.data_ptr<float>(),
            1
        ));
    }

    CUDA_CHECK(cudaEventRecord(ev_force));
    CUDA_CHECK(cudaEventSynchronize(ev_force));
    float energy_ms = 0.0f;
    float force_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&energy_ms, ev_start, ev_energy));
    CUDA_CHECK(cudaEventElapsedTime(&force_ms, ev_energy, ev_force));
    std::fprintf(
        stderr,
        "  [rff_full_gramian_elemental_rfp] energy=%.1f ms  force=%.1f ms  total=%.1f ms\n",
        energy_ms,
        force_ms,
        energy_ms + force_ms
    );
    std::fprintf(
        stderr,
        "  [rff_full_gramian_elemental_rfp] nmol=%d  max_atoms=%d  rep=%d  D=%d  "
        "energy_chunk=%d  force_chunk=%d\n",
        nmol,
        (int)X.size(1),
        (int)X.size(2),
        D,
        energy_chunk,
        force_chunk
    );
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_energy));
    CUDA_CHECK(cudaEventDestroy(ev_force));
    return {ZtZ_rfp, ZtY};
}

torch::Tensor rff_predict_force_elemental_cuda(
    const torch::Tensor &X,
    const torch::Tensor &dX,
    const torch::Tensor &Q,
    const torch::Tensor &N,
    const torch::Tensor &W,
    const torch::Tensor &b,
    const torch::Tensor &weights,
    int chunk_size
) {
    check_cuda_float32(X, "X");
    check_cuda_float32(dX, "dX");
    check_cuda_int32(Q, "Q");
    check_cuda_int32(N, "N");
    check_cuda_float32(W, "W");
    check_cuda_float32(b, "b");
    check_cuda_float32(weights, "weights");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    ensure_cublas();

    int nmol       = (int)X.size(0);
    int max_atoms  = (int)X.size(1);
    int rep_size   = (int)X.size(2);
    int nelements  = (int)W.size(0);
    int D          = (int)b.size(1);

    // Build CPU mirrors of Q, N, offsets once.
    std::vector<int> Q_cpu((size_t)nmol * max_atoms);
    std::vector<int> N_cpu((size_t)nmol);
    CUDA_CHECK(cudaMemcpy(Q_cpu.data(), Q.data_ptr<int>(),
                          (size_t)nmol * max_atoms * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(N_cpu.data(), N.data_ptr<int>(),
                          (size_t)nmol * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> offsets_cpu(nmol + 1, 0);
    for (int i = 0; i < nmol; ++i) offsets_cpu[i + 1] = offsets_cpu[i] + 3 * N_cpu[i];
    int total_grads = offsets_cpu[nmol];

    // Compute phase for all molecules at once.
    auto phase = torch::empty({nmol, max_atoms, D}, X.options());
    {
        int threads = 256;
        int blocks  = (int)(((long long)nmol * max_atoms * D + threads - 1) / threads);
        elemental_phase_kernel<<<blocks, threads>>>(
            phase.data_ptr<float>(),
            X.data_ptr<float>(),
            Q.data_ptr<int>(),
            N.data_ptr<int>(),
            W.data_ptr<float>(),
            b.data_ptr<float>(),
            0, nmol, max_atoms, rep_size, nelements, D, D, 0
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Build mol-offsets on device (for scatter kernel).
    auto off_dev = torch::empty({nmol + 1}, Q.options());
    CUDA_CHECK(cudaMemcpy(off_dev.data_ptr<int>(), offsets_cpu.data(),
                          (size_t)(nmol + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // F[total_grads] accumulates scalar contributions from all elements.
    auto F = torch::zeros({total_grads}, X.options());

    const float scale = std::sqrt(2.0f / (float)D);
    const float one   = 1.0f;
    const float zero  = 0.0f;

    for (int q = 0; q < nelements; ++q) {
        int total_rows_q = 0;
        std::vector<int> atom_map_cpu =
            build_atom_map(Q_cpu.data(), N_cpu.data(), 0, nmol, max_atoms, q, total_rows_q);
        int n_centers_q = (int)(atom_map_cpu.size() / 4);
        if (n_centers_q == 0 || total_rows_q == 0) continue;

        auto atom_map_dev = torch::empty({n_centers_q * 4}, Q.options());
        CUDA_CHECK(cudaMemcpy(atom_map_dev.data_ptr<int>(), atom_map_cpu.data(),
                              (size_t)atom_map_cpu.size() * sizeof(int), cudaMemcpyHostToDevice));

        // Gather sin(phase) * scale for all centres: sin_phase_q[n_centers_q, D]
        auto sin_phase_q = torch::empty({n_centers_q, D}, X.options());
        {
            int threads = 256;
            int blocks  = (int)(((long long)n_centers_q * D + threads - 1) / threads);
            gather_sin_phase_kernel<<<blocks, threads>>>(
                sin_phase_q.data_ptr<float>(),
                phase.data_ptr<float>(),
                weights.data_ptr<float>(),
                atom_map_dev.data_ptr<int>(),
                n_centers_q, max_atoms, D, scale
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // eff[n_centers_q, rep_size] = sin_phase_q[n_centers_q, D] @ W[q]^T[D, rep_size]
        // Memory: n_centers_q * rep_size * 4B  (e.g. 12000 * 312 * 4 = 15MB — negligible)
        // cuBLAS col-major:
        //   eff^T[rep, nc] = W[q] (row-major rep×D, seen as col-major D×rep, OP_T → rep×D)
        //                    × sin_phase_q^T [col-major D×nc]
        //   m=rep_size, n=n_centers_q, k=D
        //   A = W[q] OP_T:  lda=D  (W[q] stored as flat rep×D, col-stride = D)
        //   B = sin_phase_q OP_N:  ldb=D
        //   C = eff^T:  ldc=rep_size
        auto eff = torch::empty({n_centers_q, rep_size}, X.options());
        CUBLAS_CHECK(cublasSgemm(
            s_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            rep_size, n_centers_q, D,
            &one,
            W.data_ptr<float>() + (long long)q * rep_size * D, D,
            sin_phase_q.data_ptr<float>(), D,
            &zero,
            eff.data_ptr<float>(), rep_size
        ));

        // Gather dX rows: gathered_dX[total_rows_q, rep_size]
        auto gathered_dX = torch::empty({total_rows_q, rep_size}, X.options());
        {
            dim3 block(32, 8);
            dim3 grid(
                (unsigned int)((total_rows_q + block.x - 1) / block.x),
                (unsigned int)((rep_size + block.y - 1) / block.y)
            );
            gather_dX_for_element_kernel<<<grid, block>>>(
                gathered_dX.data_ptr<float>(),
                dX.data_ptr<float>(),
                atom_map_dev.data_ptr<int>(),
                n_centers_q, max_atoms, rep_size, total_rows_q, 0
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // For each row r: F[g_row(r)] += gathered_dX[r,:] · eff[ci(r),:]
        // Multiple centres contribute to the same g_row → atomicAdd.
        {
            int threads = 256;
            int blocks  = (int)(((long long)total_rows_q + threads - 1) / threads);
            scatter_dot_into_F_kernel<<<blocks, threads>>>(
                F.data_ptr<float>(),
                gathered_dX.data_ptr<float>(),
                eff.data_ptr<float>(),
                atom_map_dev.data_ptr<int>(),
                off_dev.data_ptr<int>(),
                n_centers_q, rep_size, total_rows_q
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return F;
}

}  // namespace kf::rff_cuda
