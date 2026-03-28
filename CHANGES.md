# J^T·α Trick — Implementation Summary

## What was done

Added efficient inference (J^T·α / "alpha_desc" trick) for the remaining
kernel flavours in `global_kernels` and `local_kernels`, and rewrote the
local Hessian matvec with a batched DGEMM strategy.

---

## Background: the J^T·α trick

The Hessian kernel between query molecule a and training molecule b factors as

    H[a,b] = J_a · K_desc(x_a, x_b)/σ² · J_b^T  −  rank-1 correction

Naively, predicting forces requires materialising H (shape naq_test × naq_train)
and multiplying by alpha_F.  The trick precomputes

    alpha_desc[b] = J_b^T @ alpha_F[b]        shape: (M,) per training atom

reducing each (query, training) pair from O(D²·M) to O(M) at inference time,
with the J_q back-projection paid once per query molecule.

---

## New kernel functions

### Global kernels (`src/global_kernels.{cpp,hpp,_bindings.cpp}`)

| Function | Predicts | Notes |
|---|---|---|
| `kernel_gaussian_jacobian_t_matvec` | Energies from force-trained model | O(N_q·N_t·M) vs O(N_q·N_t·D·M) |
| `kernel_gaussian_full_matvec` | Energies + forces from mixed model | shared C matrix; returns (E, F) tuple |

### Local kernels (`src/local_kernels.{cpp,hpp,_bindings.cpp}`)

| Function | Predicts | Notes |
|---|---|---|
| `kernel_gaussian_local_jacobian_t_matvec` | Energies from force-trained model | local equivalent |
| `kernel_gaussian_local_full_matvec` | Energies + forces from mixed model | single atom-pair loop; returns (E, F) tuple |

All four share the existing `compute_alpha_desc` / `local_compute_alpha_desc`
precompute step — no new precompute function required.

---

## Local Hessian matvec rewrite

`kernel_gaussian_local_hessian_matvec` was rewritten with a **global-label
DGEMM batching** strategy, replacing the original per-atom-pair scalar loop.

### Key idea

Pack **all** query atoms of element type L across **all** test molecules into
one matrix A_L before the DGEMM:

    A_L  shape: (R_q × M)   R_q = N_test × atoms_per_label    ← all test mols
    B_L  shape: (S_L × M)   S_L = N_train × atoms_per_label   ← all train mols

    DGEMM1: dist  (R_q × S_L) = −2 · A_L @ B_L^T
    DGEMM2: S_ad  (R_q × S_L) = A_L @ Bd_L^T
    exp loop + weight computation (elementwise, R_q × S_L)
    DGEMM3: G    (R_q × M)   = exp_C @ Bd_L
    DGEMM4: G    (R_q × M)  -= weight @ B_L
    self-correction + back-projection

### Why this is better than the intermediate per-molecule approach

An intermediate implementation that batched per (molecule, label) had r ≈ 3
rows in each DGEMM — BLAS-inefficient.  The global-label approach gives
R_q ≈ 300 rows (N_test=100, 3 atoms/label), which BLAS can fully saturate.

### Benchmark results (N_train=1000, natoms=9, rep_size=384, n_species=3)

| N_test | fast matvec (ms) | full H build + multiply (ms) | speedup |
|---|---|---|---|
| 1   |  68 |   579 |  8.6× |
| 10  |  73 |   937 | 12.7× |
| 50  |  61 |  2655 | 43.6× |
| 100 |  87 |  4820 | 55.8× |
| 200 | 184 |  9243 | 50.2× |

The fast path is nearly flat across N_test (BLAS-dominated); the full-H path
grows linearly because H has shape (naq_test × naq_train).

---

## Tests

| File | Tests | Covers |
|---|---|---|
| `tests/test_jacobian_t_matvec.py` | 19 | global jacobian_t_matvec, full_matvec |
| `tests/test_local_jacobian_t_matvec.py` | 19 | local jacobian_t_matvec, local full_matvec |

All 647 tests pass (595 pre-existing + 38 new + 14 from previous hessian_matvec work).

---

## Benchmarks

| File | What it measures |
|---|---|
| `benchmarks/bench_global_jacobian_t_matvec.py` | Global jacobian_t and full matvec vs naive |
| `benchmarks/bench_local_jacobian_t_matvec.py` | Local jacobian_t and full matvec vs naive |
| `benchmarks/bench_local_hessian_matvec.py` | Local hessian matvec vs full H, varying rep_size |
| `benchmarks/bench_local_hessian_matvec_user_case.py` | N_train=1000, varying N_test, rep=384 |
