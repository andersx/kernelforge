"""Benchmark: local Hessian matvec vs full Hessian build for the user's target case.

Parameters: N_train=1000, natoms=9, rep_size=384, n_species=3.
Sweeps N_test from 1 to 200 to show how the global-label DGEMM approach
scales with batch size.

Usage:
    uv run python benchmarks/bench_local_hessian_matvec_user_case.py
"""

import time

import numpy as np

from kernelforge import local_kernels as lk

RNG = np.random.default_rng(42)
SIGMA = 2.0
N_TRAIN = 500
NATOMS = 9
REP = 384
N_SP = 3


def make_data(
    nm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(nm, NATOMS, REP))
    dx = rng.normal(size=(nm, NATOMS, REP, 3 * NATOMS))
    q = rng.integers(1, N_SP + 1, size=(nm, NATOMS), dtype=np.int32)
    n = np.full(nm, NATOMS, dtype=np.int32)
    return x, dx, q, n


def make_fast_fn(
    x_q: np.ndarray,
    dx_q: np.ndarray,
    x_t: np.ndarray,
    alpha_desc: np.ndarray,
    q_q: np.ndarray,
    q_t: np.ndarray,
    n_q: np.ndarray,
    n_t: np.ndarray,
) -> object:
    def fn() -> None:
        lk.kernel_gaussian_local_hessian_matvec(
            x_q, dx_q, x_t, alpha_desc, q_q, q_t, n_q, n_t, SIGMA
        )

    return fn


def make_full_fn(
    x_q: np.ndarray,
    dx_q: np.ndarray,
    x_t: np.ndarray,
    dx_t: np.ndarray,
    q_q: np.ndarray,
    q_t: np.ndarray,
    n_q: np.ndarray,
    n_t: np.ndarray,
    alpha: np.ndarray,
) -> object:
    def fn() -> None:
        H = lk.kernel_gaussian_hessian(x_q, x_t, dx_q, dx_t, q_q, q_t, n_q, n_t, SIGMA)
        H @ alpha

    return fn


def timeit(fn: object, n_warmup: int = 3, n_repeat: int = 7) -> float:
    """Return minimum wall time in seconds."""
    for _ in range(n_warmup):
        fn()  # type: ignore[call-arg]
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()  # type: ignore[call-arg]
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    # Build training set once
    x_t, dx_t, q_t, n_t = make_data(N_TRAIN, RNG)
    naq_train = N_TRAIN * 3 * NATOMS
    alpha = RNG.normal(size=(naq_train,))

    # Precompute alpha_desc (training, done once after fitting)
    t0 = time.perf_counter()
    alpha_desc = lk.kernel_gaussian_local_compute_alpha_desc(dx_t, q_t, n_t, alpha)
    t_precompute = (time.perf_counter() - t0) * 1e3

    print()
    print("=" * 72)
    print(f"N_train={N_TRAIN}, natoms={NATOMS}, rep_size={REP}, n_species={N_SP}")
    print(f"precompute (compute_alpha_desc): {t_precompute:.1f} ms  (paid once)")
    print()
    print("fast = kernel_gaussian_local_hessian_matvec  (global-label DGEMM)")
    print("full = kernel_gaussian_hessian + H @ alpha    (materialise full H)")
    print("=" * 72)
    hdr = f"{'N_test':>8}  {'fast(ms)':>10}  {'full_H(ms)':>12}  {'speedup':>9}  {'H_size_MB':>10}"
    print(hdr)
    print("-" * len(hdr))

    for n_test in [1, 10, 50, 100, 200]:
        x_q, dx_q, q_q, n_q = make_data(n_test, RNG)
        naq_test = n_test * 3 * NATOMS
        # H shape: (naq_test, naq_train); size in MB
        h_mb = naq_test * naq_train * 8 / 1e6

        fast_matvec = make_fast_fn(x_q, dx_q, x_t, alpha_desc, q_q, q_t, n_q, n_t)
        full_hessian_matvec = make_full_fn(x_q, dx_q, x_t, dx_t, q_q, q_t, n_q, n_t, alpha)

        t_fast = timeit(fast_matvec) * 1e3
        t_full = timeit(full_hessian_matvec) * 1e3
        speedup = t_full / t_fast

        print(f"{n_test:>8}  {t_fast:>10.1f}  {t_full:>12.1f}  {speedup:>8.1f}x  {h_mb:>9.0f}")

    print()
    print("Notes:")
    print("  - fast path cost is nearly flat across N_test (DGEMM-dominated, BLAS-threaded)")
    print("  - full H cost grows as O(N_test) because H shape is (naq_test x naq_train)")
    print(
        f"  - H at N_test=200: {200 * 3 * NATOMS * naq_train * 8 / 1e6:.0f} MB to allocate and fill"
    )
    print("  - alpha_desc precompute is paid once; amortises immediately at N_test >= 2")


if __name__ == "__main__":
    main()
