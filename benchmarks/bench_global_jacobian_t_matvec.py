"""Benchmark: global kernel jacobian_t_matvec and full_matvec vs naive full-matrix approach.

Sweeps N_train to show speedup at inference time.
Representative molecule: ethanol-like (natoms=9, invdist rep_size=36, D=27).
"""

import time

import numpy as np

from kernelforge import global_kernels as _kernels

RNG = np.random.default_rng(42)
SIGMA = 2.0
N_REPEAT = 5

# Ethanol-like parameters
NATOMS = 9
M = NATOMS * (NATOMS - 1) // 2  # invdist rep_size = 36
D = 3 * NATOMS  # Cartesian forces = 27


def make_data(
    n_train: int, n_query: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_q = RNG.normal(size=(n_query, M))
    dx_q = RNG.normal(size=(n_query, D, M))
    x_t = RNG.normal(size=(n_train, M))
    dx_t = RNG.normal(size=(n_train, D, M))
    alpha_e = RNG.normal(size=(n_train,))
    alpha_f = RNG.normal(size=(n_train, D))
    return x_q, dx_q, x_t, dx_t, alpha_e, alpha_f


def timeit(fn: object, n: int = N_REPEAT) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()  # type: ignore[call-arg]
        times.append(time.perf_counter() - t0)
    return min(times)


def make_jt_naive(
    x_q: np.ndarray, x_t: np.ndarray, dx_t: np.ndarray, alpha_f: np.ndarray
) -> object:
    def fn() -> None:
        k_jt = _kernels.kernel_gaussian_jacobian_t(x_q, x_t, dx_t, SIGMA)
        _ = k_jt @ alpha_f.ravel()

    return fn


def make_jt_fast(x_q: np.ndarray, x_t: np.ndarray, alpha_desc_f: np.ndarray) -> object:
    def fn() -> None:
        _ = _kernels.kernel_gaussian_jacobian_t_matvec(x_q, x_t, alpha_desc_f, SIGMA)

    return fn


def make_full_naive(
    x_q: np.ndarray,
    dx_q: np.ndarray,
    x_t: np.ndarray,
    dx_t: np.ndarray,
    alpha_e: np.ndarray,
    alpha_f: np.ndarray,
) -> object:
    def fn() -> None:
        k_full = _kernels.kernel_gaussian_full(x_q, dx_q, x_t, dx_t, SIGMA)
        alpha = np.concatenate([alpha_e.ravel(), alpha_f.ravel()])
        _ = k_full @ alpha

    return fn


def make_full_fast(
    x_q: np.ndarray,
    dx_q: np.ndarray,
    x_t: np.ndarray,
    alpha_e: np.ndarray,
    alpha_desc_f: np.ndarray,
) -> object:
    def fn() -> None:
        _, _ = _kernels.kernel_gaussian_full_matvec(x_q, dx_q, x_t, alpha_e, alpha_desc_f, SIGMA)

    return fn


print(f"\n=== Global jacobian_t_matvec benchmark (natoms={NATOMS}, M={M}, D={D}) ===")
print(
    f"{'N_train':>8} | {'jt_naive(ms)':>14} | {'jt_fast(ms)':>13} | {'speedup_jt':>12}"
    f" | {'full_naive(ms)':>16} | {'full_fast(ms)':>15} | {'speedup_full':>13}"
)
print("-" * 110)

for n_train in [50, 100, 200, 500, 1000, 2000]:
    x_q, dx_q, x_t, dx_t, alpha_e, alpha_f = make_data(n_train, n_query=100)
    alpha_desc_f = _kernels.kernel_gaussian_compute_alpha_desc(dx_t, alpha_f)

    t_jt_naive = timeit(make_jt_naive(x_q, x_t, dx_t, alpha_f)) * 1e3
    t_jt_fast = timeit(make_jt_fast(x_q, x_t, alpha_desc_f)) * 1e3
    t_full_naive = timeit(make_full_naive(x_q, dx_q, x_t, dx_t, alpha_e, alpha_f)) * 1e3
    t_full_fast = timeit(make_full_fast(x_q, dx_q, x_t, alpha_e, alpha_desc_f)) * 1e3

    speedup_jt = t_jt_naive / t_jt_fast if t_jt_fast > 0 else float("inf")
    speedup_full = t_full_naive / t_full_fast if t_full_fast > 0 else float("inf")

    print(
        f"{n_train:>8} | {t_jt_naive:>14.2f} | {t_jt_fast:>13.2f} | {speedup_jt:>11.1f}x"
        f" | {t_full_naive:>16.2f} | {t_full_fast:>15.2f} | {speedup_full:>12.1f}x"
    )

print()
print("Notes:")
print(f"  - jacobian_t speedup asymptote: D={D} (force dimension)")
print(f"  - full_matvec speedup driven by eliminating H block ({D}x{D} per pair)")
print("  - precompute cost (compute_alpha_desc) not included in 'fast' timings")
print("  - N_query=1 throughout (single prediction)")
