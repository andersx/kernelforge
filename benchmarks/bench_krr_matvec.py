"""bench_krr_matvec.py — Compare kernel_gaussian_full_matvec vs _cached.

Measures per-step wall time for the KRR matvec (the hot path during MD).
Also prototypes element-blocked SGEMMs to estimate potential further speedup.

Usage
-----
    uv run --env-file /dev/null python benchmarks/bench_krr_matvec.py

Expected output (approximate, RTX 5070 Ti, aspirin 950 train):
    precompute_train:   X.X ms  (one-time cost)
    matvec uncached:    X.X ms / call
    matvec cached:      X.X ms / call
    speedup:            X.Xx
    element-blocked:    X.X ms / call  (prototype via torch)
"""

from __future__ import annotations

import time

import numpy as np
import torch

from kernelforge.kernelcli import load_rmd17  # type: ignore[attr-defined]  # noqa: F401


def _load_aspirin_model(n_train: int = 950) -> object:
    """Fit a CudaLocalKRRModel on aspirin rMD17 and return the fitted model."""
    from kernelforge.kernelcli import load_rmd17  # type: ignore[attr-defined]
    from kernelforge.models import CudaLocalKRRModel

    tr_coords, tr_z, tr_E, tr_F, te_coords, te_z, *_ = load_rmd17("aspirin", split=1, n_train=n_train, n_test=100)
    model = CudaLocalKRRModel(sigma=2.0, l2=1e-2, elements=[1, 6, 8])
    model.fit(
        list(tr_coords[:n_train]),
        [tr_z[:n_train][i] for i in range(n_train)],
        energies=tr_E[:n_train],
        forces=tr_F[:n_train],
    )
    return model


def _build_query(model: object) -> tuple[torch.Tensor, ...]:
    """Build a single-molecule query tensor set (one aspirin frame)."""
    from kernelforge.kernelcli import load_rmd17 as _load_rmd17  # type: ignore[attr-defined]

    from kernelforge.models.cuda_local_krr import _compute_fchl19_cuda  # type: ignore[attr-defined]

    _, _, _, _, te_coords, te_z, *_ = _load_rmd17("aspirin", split=1, n_train=100, n_test=10)
    coords_list = [te_coords[0]]
    z_list = [te_z[0]]

    X_q, dX_q, Q_q, _Q_np, N_q_np = _compute_fchl19_cuda(
        coords_list,
        z_list,
        model.elements,  # type: ignore[attr-defined]
        with_gradients=True,
        repr_params=model.repr_params,  # type: ignore[attr-defined]
        deterministic=False,
    )
    N_q = torch.tensor(N_q_np, dtype=torch.int32).cuda()
    return X_q, dX_q, Q_q, N_q


def _warmup(fn: object, n: int = 3) -> None:
    for _ in range(n):
        fn()  # type: ignore[operator]
    torch.cuda.synchronize()


def _time_fn(fn: object, n: int = 50) -> float:
    """Return average wall time in ms for n calls to fn()."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()  # type: ignore[operator]
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def bench_element_blocked(
    X_q: torch.Tensor,
    dX_q: torch.Tensor,
    Q_q: torch.Tensor,
    N_q: torch.Tensor,
    model: object,
    n_reps: int = 50,
) -> float:
    """Prototype element-blocked SGEMMs via PyTorch on GPU.

    This measures the time for the two dominant SGEMMs when split by element
    (C_qt = X_q_e @ X_t_e^T  and  inner_F_e = X_q_e @ alpha_desc_e^T) with
    the element grouping precomputed.  The reduce and back-projection steps
    are excluded — this is a lower bound on a fully element-blocked kernel.
    """
    X_t: torch.Tensor = model._X_train_cuda  # type: ignore[attr-defined]
    Q_t: torch.Tensor = model._Q_train_cuda  # type: ignore[attr-defined]
    N_t: torch.Tensor = model._N_train_cuda  # type: ignore[attr-defined]
    adF: torch.Tensor = model._alpha_desc_F_cuda  # type: ignore[attr-defined]

    nm_t, max_atoms_t, rep = X_t.shape
    N_t_flat = nm_t * max_atoms_t
    X_t_flat = X_t.view(N_t_flat, rep)
    adF_flat = adF.view(N_t_flat, rep)

    # Flatten Q_t and N_t on CPU for element grouping (one-time precompute)
    Q_t_cpu = Q_t.cpu().numpy().ravel()
    N_t_cpu = N_t.cpu().numpy()
    N_q_cpu = N_q.cpu().numpy()

    # Active atom mask for training
    active_t = np.zeros(N_t_flat, dtype=bool)
    for m in range(nm_t):
        start = m * max_atoms_t
        active_t[start : start + N_t_cpu[m]] = True

    # Per-element GPU index tensors (precomputed once — not counted in timing)
    labels = sorted(set(Q_t_cpu[active_t].tolist()))
    elem_idx_t: dict[int, torch.Tensor] = {}
    for lbl in labels:
        idx = np.where((Q_t_cpu == lbl) & active_t)[0]
        elem_idx_t[lbl] = torch.from_numpy(idx).long().cuda()

    nm_q, max_atoms_q, _ = X_q.shape
    X_q_flat = X_q.view(nm_q * max_atoms_q, rep)
    Q_q_flat = Q_q.cpu().numpy().ravel()
    active_q = np.zeros(nm_q * max_atoms_q, dtype=bool)
    for m in range(nm_q):
        start = m * max_atoms_q
        active_q[start : start + N_q_cpu[m]] = True

    elem_idx_q: dict[int, torch.Tensor] = {}
    for lbl in labels:
        idx = np.where((Q_q_flat == lbl) & active_q)[0]
        if len(idx):
            elem_idx_q[lbl] = torch.from_numpy(idx).long().cuda()

    # ---- Timed element-blocked SGEMM loop ----
    def run_blocked() -> None:
        for lbl in labels:
            if lbl not in elem_idx_q:
                continue
            iq = elem_idx_q[lbl]
            it = elem_idx_t[lbl]
            Xq_e = X_q_flat[iq]   # (Rq, rep)
            Xt_e = X_t_flat[it]   # (Rt, rep)
            aD_e = adF_flat[it]   # (Rt, rep)
            # C_e = -2 * Xq_e @ Xt_e^T   shape: (Rq, Rt)
            _C_e = torch.mm(Xq_e, Xt_e.t()).mul_(-2.0)
            # iF_e = Xq_e @ aD_e^T        shape: (Rq, Rt)
            _iF_e = torch.mm(Xq_e, aD_e.t())

    _warmup(run_blocked)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        run_blocked()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n_reps


def main() -> None:
    from kernelforge import cuda_local_kernels as _ext

    print("Loading rMD17 aspirin + fitting KRR model (950 train) …")
    model = _load_aspirin_model(950)
    print("  done.\n")

    X_q, dX_q, Q_q, N_q = _build_query(model)

    # ---- uncached matvec ----
    def run_uncached() -> None:
        _ext.kernel_gaussian_full_matvec(
            X_q, dX_q, Q_q, N_q,
            model._X_train_cuda,  # type: ignore[attr-defined]
            model._Q_train_cuda,  # type: ignore[attr-defined]
            model._N_train_cuda,  # type: ignore[attr-defined]
            model._alpha_E_cuda,  # type: ignore[attr-defined]
            model._alpha_desc_F_cuda,  # type: ignore[attr-defined]
            float(model.sigma),  # type: ignore[attr-defined]
        )

    # ---- cached matvec ----
    norms_t = model._norms_t_cuda  # type: ignore[attr-defined]
    S_adF = model._S_adF_cuda  # type: ignore[attr-defined]
    alpha_E_t = model._alpha_E_t_cuda  # type: ignore[attr-defined]
    combined_t = model._combined_t_cuda  # type: ignore[attr-defined]

    def run_cached() -> None:
        _ext.kernel_gaussian_full_matvec_cached(
            X_q, dX_q, Q_q, N_q,
            model._X_train_cuda,  # type: ignore[attr-defined]
            model._Q_train_cuda,  # type: ignore[attr-defined]
            model._N_train_cuda,  # type: ignore[attr-defined]
            model._alpha_E_cuda,  # type: ignore[attr-defined]
            model._alpha_desc_F_cuda,  # type: ignore[attr-defined]
            norms_t, S_adF, alpha_E_t, combined_t,
            float(model.sigma),  # type: ignore[attr-defined]
        )

    # ---- precompute one-time cost ----
    def run_precompute() -> None:
        _ext.precompute_train(
            model._X_train_cuda,  # type: ignore[attr-defined]
            model._Q_train_cuda,  # type: ignore[attr-defined]
            model._N_train_cuda,  # type: ignore[attr-defined]
            model._alpha_E_cuda,  # type: ignore[attr-defined]
            model._alpha_desc_F_cuda,  # type: ignore[attr-defined]
        )

    # Warm up all paths
    print("Warming up …")
    _warmup(run_uncached, n=5)
    _warmup(run_cached, n=5)
    _warmup(run_precompute, n=3)

    N_BENCH = 100
    t_precompute = _time_fn(run_precompute, n=50)
    t_uncached = _time_fn(run_uncached, n=N_BENCH)
    t_cached = _time_fn(run_cached, n=N_BENCH)

    print(f"\n{'─'*55}")
    print(f"  Aspirin, 950 training frames, single-molecule query")
    print(f"{'─'*55}")
    print(f"  precompute_train (one-time): {t_precompute:6.2f} ms")
    print(f"  matvec  uncached:            {t_uncached:6.2f} ms")
    print(f"  matvec  cached:              {t_cached:6.2f} ms")
    print(f"  speedup:                     {t_uncached / t_cached:6.2f}x")
    print(f"{'─'*55}\n")

    # ---- element-blocked prototype ----
    print("Benchmarking element-blocked SGEMM prototype …")
    t_blocked = bench_element_blocked(X_q, dX_q, Q_q, N_q, model, n_reps=N_BENCH)
    nm_t = int(model._X_train_cuda.shape[0])  # type: ignore[attr-defined]
    max_atoms_t = int(model._X_train_cuda.shape[1])  # type: ignore[attr-defined]
    Q_t_cpu = model._Q_train_cuda.cpu().numpy().ravel()  # type: ignore[attr-defined]
    N_t_cpu = model._N_train_cuda.cpu().numpy()  # type: ignore[attr-defined]
    n_total = sum(N_t_cpu)
    labels = sorted(set(int(q) for m in range(nm_t) for q in Q_t_cpu[m * max_atoms_t : m * max_atoms_t + N_t_cpu[m]]))
    elem_frac = {
        lbl: sum(int(Q_t_cpu[m * max_atoms_t + i]) == lbl
                 for m in range(nm_t) for i in range(N_t_cpu[m])) / n_total
        for lbl in labels
    }
    waste = 1.0 - sum(f ** 2 for f in elem_frac.values())

    print(f"\n{'─'*55}")
    print(f"  Element-blocked SGEMM (2 SGEMMs per element, no reduce/backproj)")
    print(f"{'─'*55}")
    print(f"  Blocked SGEMM only:          {t_blocked:6.2f} ms")
    print(f"  Useful fraction of full:     {100*(1-waste):5.1f}%  (rest is cross-element waste)")
    print(f"  Wasted dot-products:         {100*waste:5.1f}%")
    for lbl, f in sorted(elem_frac.items()):
        print(f"    element {lbl:3d}:  {100*f:5.1f}% of atoms")
    print(f"{'─'*55}")
    print()
    print("Note: blocked time excludes reduce-to-KEE and force back-projection,")
    print("which are fast and element-agnostic.  The full blocked kernel would")
    print(f"be somewhat slower than {t_blocked:.2f} ms but much faster than {t_uncached:.2f} ms.")


if __name__ == "__main__":
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Suppress the per-call stderr timing lines from the uncached matvec
    import sys

    _old_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")  # noqa: PTH123, SIM115, WPS515
    try:
        main()
    finally:
        sys.stderr.close()
        sys.stderr = _old_stderr
