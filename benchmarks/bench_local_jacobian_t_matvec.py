"""Benchmark: local kernel jacobian_t_matvec and full_matvec vs naive full-matrix approach.

Sweeps N_train to show speedup at inference time.
Representative molecule: ethanol-like (natoms=9, FCHL19 rep_size=27).
"""

import time

import numpy as np

from kernelforge import local_kernels as _kernels

RNG = np.random.default_rng(42)
SIGMA = 2.0
N_REPEAT = 5

# Ethanol-like parameters
NATOMS = 9
REP_SIZE = 384
MAX_ATOMS = NATOMS


def make_data(
    n_train: int, n_query: int = 100
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    x1 = RNG.normal(size=(n_query, MAX_ATOMS, REP_SIZE))
    dx1 = RNG.normal(size=(n_query, MAX_ATOMS, REP_SIZE, 3 * MAX_ATOMS))
    x2 = RNG.normal(size=(n_train, MAX_ATOMS, REP_SIZE))
    dx2 = RNG.normal(size=(n_train, MAX_ATOMS, REP_SIZE, 3 * MAX_ATOMS))
    q1 = np.ones((n_query, MAX_ATOMS), dtype=np.int32)
    q2 = np.ones((n_train, MAX_ATOMS), dtype=np.int32)
    n1 = np.full((n_query,), NATOMS, dtype=np.int32)
    n2 = np.full((n_train,), NATOMS, dtype=np.int32)
    naq1 = n_query * 3 * NATOMS
    naq2 = n_train * 3 * NATOMS
    return x1, dx1, x2, dx2, q1, q2, n1, n2, naq1, naq2


def timeit(fn: object, n: int = N_REPEAT) -> float:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()  # type: ignore[call-arg]
        times.append(time.perf_counter() - t0)
    return min(times)


def make_jt_naive(
    x1: np.ndarray,
    x2: np.ndarray,
    dx2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    alpha_f: np.ndarray,
) -> object:
    def fn() -> None:
        k_jac = _kernels.kernel_gaussian_jacobian(x1, x2, dx2, q1, q2, n1, n2, SIGMA)
        _ = k_jac @ alpha_f

    return fn


def make_jt_fast(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha_desc_f: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
) -> object:
    def fn() -> None:
        _ = _kernels.kernel_gaussian_local_jacobian_t_matvec(
            x1, x2, alpha_desc_f, q1, q2, n1, n2, SIGMA
        )

    return fn


def make_full_naive(
    x1: np.ndarray,
    dx1: np.ndarray,
    x2: np.ndarray,
    dx2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    alpha_e: np.ndarray,
    alpha_f: np.ndarray,
) -> object:
    def fn() -> None:
        k_full = _kernels.kernel_gaussian_full(x1, x2, dx1, dx2, q1, q2, n1, n2, SIGMA)
        alpha = np.concatenate([alpha_e, alpha_f])
        _ = k_full @ alpha

    return fn


def make_full_fast(
    x1: np.ndarray,
    dx1: np.ndarray,
    x2: np.ndarray,
    alpha_desc_f: np.ndarray,
    alpha_e: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
) -> object:
    def fn() -> None:
        _, _ = _kernels.kernel_gaussian_local_full_matvec(
            x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2, SIGMA
        )

    return fn


print(
    f"\n=== Local jacobian_t_matvec benchmark (natoms={NATOMS}, rep={REP_SIZE}, D={3 * NATOMS}) ==="
)
print(
    f"{'N_train':>8} | {'jt_naive(ms)':>14} | {'jt_fast(ms)':>13} | {'speedup_jt':>12}"
    f" | {'full_naive(ms)':>16} | {'full_fast(ms)':>15} | {'speedup_full':>13}"
)
print("-" * 110)

for n_train in [25, 50, 100, 200, 500]:
    x1, dx1, x2, dx2, q1, q2, n1, n2, _naq1, naq2 = make_data(n_train)
    alpha_e = RNG.normal(size=(n_train,))
    alpha_f = RNG.normal(size=(naq2,))
    alpha_desc_f = _kernels.kernel_gaussian_local_compute_alpha_desc(dx2, q2, n2, alpha_f)

    t_jt_naive = timeit(make_jt_naive(x1, x2, dx2, q1, q2, n1, n2, alpha_f)) * 1e3
    t_jt_fast = timeit(make_jt_fast(x1, x2, alpha_desc_f, q1, q2, n1, n2)) * 1e3
    t_full_naive = timeit(make_full_naive(x1, dx1, x2, dx2, q1, q2, n1, n2, alpha_e, alpha_f)) * 1e3
    t_full_fast = timeit(make_full_fast(x1, dx1, x2, alpha_desc_f, alpha_e, q1, q2, n1, n2)) * 1e3

    speedup_jt = t_jt_naive / t_jt_fast if t_jt_fast > 0 else float("inf")
    speedup_full = t_full_naive / t_full_fast if t_full_fast > 0 else float("inf")

    print(
        f"{n_train:>8} | {t_jt_naive:>14.2f} | {t_jt_fast:>13.2f} | {speedup_jt:>11.1f}x"
        f" | {t_full_naive:>16.2f} | {t_full_fast:>15.2f} | {speedup_full:>12.1f}x"
    )

print()
print("Notes:")
print("  - jacobian_t speedup: avoids building K_jac (nm1 x naq2) matrix")
print("  - full_matvec: combines hessian + jacobian_t energy in one loop")
print(f"  - naq2 at N_train=500: {500 * 3 * NATOMS} (vs nm2={500})")
print("  - precompute cost (compute_alpha_desc) not included in 'fast' timings")
print("  - N_query=1 throughout (single prediction)")
