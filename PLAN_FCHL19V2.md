# FCHL19v2 Representation — Implementation Plan

## Goal

Implement a new `fchl19v2_repr` module with selectable two-body and three-body basis
functions, to benchmark different representation designs on rMD17.

The existing `fchl19_repr` module is left **untouched** as the baseline.

---

## Study Design

### Phase 1: Find best two-body term (fix three-body at A1 baseline)

| ID | Name | Formula |
|----|------|---------|
| T1 | LogNormal (baseline) | `sigma^2 = ln(1 + eta2/r^2); mu = ln(r) - sigma^2/2; G_k = f_c(r) / (sigma*sqrt(2pi)*r^p*Rs2[k]) * exp(-(ln(Rs2[k])-mu)^2 / (2*sigma^2))` |
| T2 | GaussianR | `G_k = f_c(r) / r^p * exp(-eta2 * (r - Rs2[k])^2)` |
| T3 | GaussianLogR | `G_k = f_c(r) / r^p * exp(-eta2 * (ln(r) - ln(Rs2[k]))^2)` |
| T4 | GaussianRNoPow | `G_k = f_c(r) * exp(-eta2 * (r - Rs2[k])^2)` (no 1/r^p) |
| T5 | Bessel | `G_k = f_c(r) * sqrt(2/rcut) * sin(k*pi*r/rcut) / r` |

### Phase 2: Find best three-body term (fix two-body at Phase 1 winner)

| ID | Angular | Radial | ATM |
|----|---------|--------|-----|
| A1 | Odd Fourier (cos+sin, o=1,3,5,...) | r_bar = (r_ij+r_ik)/2 | Yes |
| A2 | Full cosine series cos(m*theta) | r_bar | Yes |
| A3 | Odd Fourier | Split r_plus/r_minus | Yes |
| A4 | Full cosine series | Split r_plus/r_minus | Yes |
| A5 | Full cosine series | Split r_plus/r_minus | No |

### Dataset: rMD17

- 2-3 molecules (e.g. ethanol, uracil, aspirin)
- Standard train/test split
- Report MAE for energy (meV) and forces (meV/Ang)

---

## Architecture

### New files

| File | Purpose |
|------|---------|
| `src/fchl19v2_repr.hpp` | Header: enums, function declarations |
| `src/fchl19v2_repr.cpp` | C++ implementation of all variants |
| `src/fchl19v2_repr_bindings.cpp` | Pybind11 bindings |
| `python/kernelforge/fchl19v2_repr.pyi` | Type stubs |
| `tests/test_fchl19v2.py` | Tests for all variants |

### Modified files

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `kf_add_cpp_module(fchl19v2_repr ...)` |

### C++ Enums

```cpp
enum class TwoBodyType { LogNormal, GaussianR, GaussianLogR, GaussianRNoPow, Bessel };
enum class ThreeBodyType { OddFourier_Rbar, CosineSeries_Rbar, OddFourier_SplitR,
                           CosineSeries_SplitR, CosineSeries_SplitR_NoATM };
```

### Rep size formula

- Two-body block: `nelements * nbasis2` (same for all two-body variants)
- Three-body block:
  - Rbar variants: `n_pairs * nbasis3 * nabasis`
  - SplitR variants: `n_pairs * nbasis3_plus * nbasis3_minus * nabasis`
- Where `n_pairs = nelements * (nelements + 1) / 2`
- And `nabasis`:
  - OddFourier: `2 * nFourier` (cos+sin pairs for odd harmonics)
  - CosineSeries: `nCosine` (cos(0), cos(theta), ..., cos((nCosine-1)*theta))

### Gradient strategy

All variants get **analytic gradients**. The gradient accumulation loop is shared;
only the per-variant `d_radial`, `d_angular`, `d_atm` computations differ.

### Cutoff / decay function (shared)

```
f_c(r, r_c) = 0.5 * (cos(pi * r / r_c) + 1)
```

---

## Implementation Order

Each step includes forward pass + analytic gradient + tests.

1. **Scaffolding**: header, bindings, CMake, stubs, empty test file
2. **T1**: LogNormal two-body (port from existing) — baseline validation
3. **T2**: GaussianR two-body
4. **T3**: GaussianLogR two-body
5. **T4**: GaussianRNoPow two-body
6. **T5**: Bessel two-body
7. **A1**: OddFourier + Rbar three-body (port from existing) — baseline validation
8. **A2**: CosineSeries + Rbar three-body
9. **A3**: OddFourier + SplitR three-body
10. **A4**: CosineSeries + SplitR three-body
11. **A5**: CosineSeries + SplitR + NoATM three-body

### Tests per variant

- Shape matches `compute_rep_size`
- Translation invariance (shift all coords)
- Finite-difference gradient check (central differences, h=1e-6, rtol=5e-6)
- Three-body zeroing when `three_body_weight=0`
- `generate()` matches `generate_and_gradients()` representation output

---

## Equations

### Two-body terms

All share the cutoff `f_c(r, rcut) = 0.5*(cos(pi*r/rcut) + 1)`.

**T1 (LogNormal):**
```
sigma^2 = ln(1 + eta2/r^2)
mu = ln(r) - sigma^2/2
G_k = f_c(r) / (sigma * sqrt(2*pi) * r^p * Rs2[k]) * exp(-(ln(Rs2[k]) - mu)^2 / (2*sigma^2))
```

**T2 (GaussianR):**
```
G_k = f_c(r) / r^p * exp(-eta2 * (r - Rs2[k])^2)
```

**T3 (GaussianLogR):**
```
G_k = f_c(r) / r^p * exp(-eta2 * (ln(r) - ln(Rs2[k]))^2)
```

**T4 (GaussianRNoPow):**
```
G_k = f_c(r) * exp(-eta2 * (r - Rs2[k])^2)
```

**T5 (Bessel):**
```
G_k = f_c(r) * sqrt(2/rcut) * sin((k+1) * pi * r / rcut) / r
```

### Three-body angular terms

**OddFourier** (o = 1, 3, 5, ...):
```
w_m = 2 * exp(-0.5 * (zeta * o)^2),  o = 2*m + 1
A[2*m]   = w_m * cos(o * theta)
A[2*m+1] = w_m * sin(o * theta)
```

**CosineSeries** (m = 0, 1, 2, ...):
```
A[m] = cos(m * theta)
```

### Three-body radial terms

**Rbar:**
```
phi_l = exp(-eta3 * (r_bar - Rs3[l])^2),  r_bar = (r_ij + r_ik) / 2
```

**SplitR:**
```
phi_{l1,l2} = exp(-eta3 * (r_plus - Rs3_plus[l1])^2) * exp(-eta3_minus * (r_minus - Rs3_minus[l2])^2)
r_plus = r_ij + r_ik,  r_minus = |r_ij - r_ik|
```

### ATM factor

```
ksi3 = (1 + 3*cos_i*cos_j*cos_k) / (r_ij*r_jk*r_ik)^q * w3
```

where cos_i, cos_j, cos_k are triangle angles at i, j, k respectively.
