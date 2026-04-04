# RFF Joint Energy+Force Training: Solver Comparison

## Problem

Train on both energies and forces simultaneously. The stacked least-squares system is:

```
| Z_cos^T      |       | E |
| -G           | w  =  | F |
| sqrt(λ) * I  |       | 0 |    (optional)
```

## Three solvers benchmarked

### 1. Cholesky on normal equations (dsyrk + dposv)

Forms `(Z Z^T + G^T G + λI) w = Z E + G^T F` — the `(D, D)` system.

**Timings** (N=2000, D=4096, ncoords=15): **10.4 s** total, dominated by `dsyrk G^T G` at 6.9 s (66%). Also requires ~1 GB for full G matrix.

**Accuracy**: Poor. Normal equations square the condition number. Energy MAE: 0.386, Force MAE: 0.735.

A batched variant (bs=200) avoids the 1 GB allocation and is slightly faster (~9.4 s), but has the same accuracy problem.

### 2. QR with regularization (dgels)

Solves the stacked system directly via QR factorization. Matrix size: `(N + N*ncoords + D, D)` = `(36096, 4096)`.

**Timings**: ~21.3 s (N=2000) / 1.87 s (N=500). The dgels solve dominates at ~86% of total time. Assembly (transpose copy + DGEMM + sin scaling) is only ~14%.

**Accuracy**: 7 orders of magnitude better than Cholesky. Energy MAE: 2.6e-7, Force MAE: 7.8e-7.

### 3. QR without regularization (dgels, preferred)

Same as above but drops the `sqrt(λ) * I` block. Matrix size: `(N + N*ncoords, D)` = `(32000, 4096)`.

**Timings**: 1.38 s at N=500 — **26% faster** than QR+reg (better than the naive 11% row reduction due to superlinear dgels scaling).

**Accuracy**: Machine precision. Energy MAE: 4.8e-16, Force MAE: 8.6e-16. The system is heavily overdetermined (32000 >> 4096), so dgels finds a unique solution without regularization. Weight magnitudes are actually *smaller* without regularization (max|w| = 0.998 vs 1.353).

## Accuracy comparison (N=500, D=2048)

| Solver | Energy MAE | Force MAE | Max |w| |
|---|---|---|---|
| Cholesky | 0.386 | 0.731 | — |
| QR + reg | 2.6e-7 | 7.8e-7 | 1.353 |
| **QR no reg** | **4.8e-16** | **8.6e-16** | **0.998** |

## FLOP comparison (N=2000, D=4096)

| | Cholesky path | QR no reg |
|---|---|---|
| dsyrk G^T G | 5.0e11 | eliminated |
| Cholesky solve | 2.3e10 | — |
| QR factorization | — | 1.1e12 |
| **Total** | **5.3e11** | **1.1e12** |

QR does ~2x more FLOPs but avoids forming the normal equations entirely. The tradeoff: **2x slower, but machine-precision accuracy and no λ tuning needed**.

## Key takeaways

1. **Cholesky on normal equations is numerically inadequate** for joint E+F training — it squares the condition number and loses 7+ orders of magnitude in accuracy.
2. **QR without regularization is the recommended solver** — fastest QR variant, best accuracy, no hyperparameter to tune.
3. **Regularization is unnecessary** when the system is overdetermined (N + N*ncoords >> D). The least-squares solution naturally constrains weight magnitude.
4. **Training is done once; inference runs millions of times**. The 2x FLOP penalty of QR over Cholesky is acceptable. Use the fused sincos inference path (0.11 s) for production.

## Fused sincos for training prep

Both `Z_cos` and `Z_sin` are computed from one DGEMM + one fused sincos loop (~0.26 s). The compiler emits a single `sincos` SVML intrinsic. This is negligible compared to the solver cost.

## Recommended pipeline

| Phase | Operation | Time |
|---|---|---|
| Train prep | `rff_train_sincos` (one DGEMM + fused sincos) | ~0.3 s |
| Train assemble | Stack `Z_cos^T` and `-G` into matrix | ~3 s |
| **Train solve** | **`dgels` (QR, no regularization)** | **~16 s** |
| Inference | `rff_predict_energy_forces_fused` (Variant A) | ~0.11 s |
