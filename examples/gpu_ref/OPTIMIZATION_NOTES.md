# RFF Force Prediction: Optimization Journey

## Problem

Predict molecular energies and forces using Random Fourier Features (RFF) for kernel ridge regression. The energy is `E = Z(X)^T w` where `Z` is the RFF feature vector, and forces are obtained via the chain rule through the molecular representation Jacobian `dX/dcoords`.

## Starting point: Full gradient feature matrix (Step 5)

The initial implementation builds the full gradient feature matrix `G` of shape `(N*ncoords, D)`, matching the C++ `rff_gradient` function:

```
Step 1: Z = omega^T X + b                      DGEMM       O(D * N * rep_size)
Step 2: G = dX_T^T @ omega                     DGEMM       O(N * ncoords * D * rep_size)
Step 3: G *= -sqrt(2/D) * sin(Z(d, mol(g)))    OMP loop    O(N * ncoords * D)
Step 4: F = -G @ w                             dgemv       O(N * ncoords * D)
```

**Benchmark** (`N=400, rep_size=512, D=4096, ncoords=15`):
- Step 2 DGEMM dominated at **65%** of total time (~0.45 s)
- Step 3 sin+scale loop took **~0.17 s**
- Total: **~0.60 s** for forces only

## Optimization 1: VML for sin/cos (mixed results)

Replaced the OMP loops containing scalar `sin()`/`cos()` with MKL VML calls (`vdSin`, `vdCos`).

**Result**: No meaningful improvement. The `ifx -xHost` compiler already emits AVX-512 SVML vectorized intrinsics for the scalar sin/cos in OMP loops. The VML calls added overhead from multiple memory passes (separate `dger` + `vdSin` + `dscal` vs. one fused loop).

**Lesson**: ifx with `-xHost` auto-vectorizes trig very well. Separate BLAS/VML calls are not always faster than a fused compiler-vectorized loop.

## Optimization 2: Loop order + molecule blocking (10x on Step 3)

The Step 3 multiply+scale loop had a cache-hostile access pattern:

```fortran
! BAD: inner loop varies id (column index) — stride of total_grads*8 bytes per element
do ig = 1, total_grads        ! outer
    do id = 1, nfeat           ! inner — cache miss every iteration
        G(ig, id) = G(ig, id) * sin(Z(id, mol_idx)) * normalization
    end do
end do
```

G is `(total_grads, nfeat)` in Fortran column-major. Accessing `G(ig, 1), G(ig, 2), G(ig, 3)...` strides across 240 KB per element — every access is a cache miss.

Fixed by swapping loop order and blocking by molecule:

```fortran
! GOOD: inner loop varies ig (row index) — contiguous access
do id = 1, nfeat               ! outer (OMP parallelized)
    do imol = 1, nmol
        s = Z(id, imol) * normalization    ! scalar, hoisted
        G(g_start:g_start+ncoords-1, id) = G(g_start:g_start+ncoords-1, id) * s
    end do
end do
```

**Result**: Step 3 went from **0.82 s to 0.085 s** — a **10x speedup**. The Step 2 DGEMM now accounted for ~95% of total gradient time.

## Optimization 3: Contract w first (Step 6 — eliminate big DGEMM)

At inference (when `w` is known), we can avoid materializing the huge `G` matrix entirely by contracting `omega` with `w` first:

```
F(g) = sum_d G(g,d) * w(d)
     = sum_r dX_T(r,g) * [sum_d omega(r,d) * (-sqrt(2/D) * sin(Z(d,mol(g))) * w(d))]
     = sum_r dX_T(r,g) * v(r, mol(g))
```

where `v = omega @ S` with `S(d,i) = -sqrt(2/D) * sin(Z(d,i) + b(d)) * w(d)`.

New algorithm:
```
Step 1: S = -sqrt(2/D) * sin(omega^T X + b) .* w    DGEMM + fused loop    O(D*N*rep_size)
Step 2: v = omega @ S                                DGEMM                 O(rep_size * D * N)
Step 3: F_i = -dX_T_i^T @ v(:,i)                    N x dgemv             O(N * ncoords * rep_size)
```

The old Step 2 was `O(N * ncoords * D * rep_size)`. The new approach is `O(N * rep_size * (D + ncoords))`. Speedup factor: `ncoords * D / (D + ncoords)` ≈ **15x** for these dimensions.

**Result**: **0.15 s** vs. 0.60 s — **4x overall speedup**. The `(N*ncoords, D)` matrix is never allocated, saving ~1 GB of memory.

## Optimization 4: Fused energy + forces with sincos (Step 7A — final)

When predicting both energies and forces, the separate paths redundantly compute `omega^T X` twice and evaluate sin and cos separately. Fusing everything:

```
Step 1: P = omega^T X                                        single DGEMM
Step 2: Z_cos = sqrt(2/D)*cos(P+b), S = -sqrt(2/D)*sin(P+b)*w   fused OMP loop
Step 3: E = Z_cos^T @ w                                      dgemv
Step 4: v = omega @ S                                        DGEMM
Step 5: F_i = -dX_T_i^T @ v(:,i)                             N x dgemv
```

The fused loop in Step 2 computes sin and cos of the **same argument**. The ifx compiler recognizes this and emits a single `sincos` SVML intrinsic — one hardware evaluation for both results. This loop also folds in bias addition, scaling, and w-multiplication in a single memory pass.

We also benchmarked a Variant B using MKL `vdSinCos` with separate scaling passes, but the fused OMP loop (Variant A) was faster due to fewer memory passes.

**Result**: **0.11 s** for both energy AND forces — faster than Step 6 forces-only (0.15 s), **5.5x faster** than the original.

## Final benchmark summary

All paths produce identical results (max difference ~8e-15, machine precision).

| Path | Computes | Time | Speedup | Memory for G |
|---|---|---|---|---|
| Step 5: Full G matrix | Forces only | 0.60 s | 1x | (N*ncoords, D) |
| Step 6: Contracted w | Forces only | 0.15 s | 4x | None |
| Step 7A: Fused sincos | Energy + Forces | **0.11 s** | **5.5x** | None |
| Step 7B: MKL vdSinCos | Energy + Forces | 0.17 s | 3.5x | None |

Parameters: `N=400, rep_size=512, D=4096, ncoords=15, 8 threads, ifx + MKL`.

## Key takeaways

1. **Loop order matters more than VML**: Fixing column-major access patterns gave 10x on the scaling loop, while MKL VML gave nothing because ifx already auto-vectorizes well.
2. **Algebraic restructuring beats micro-optimization**: Contracting `w` first eliminated the dominant DGEMM entirely — no amount of loop tuning could match that.
3. **Fused loops beat separate BLAS/VML calls**: One pass doing bias+sincos+scale+w-multiply is faster than `dger` + `vdSinCos` + `dscal` + scaling loop, because each separate call has overhead and requires a full memory pass.
4. **The compiler is smart**: ifx with `-xHost` recognizes paired sin/cos on the same argument and emits `sincos` without any manual intrinsics.
