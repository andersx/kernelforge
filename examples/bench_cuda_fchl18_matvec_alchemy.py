# ruff: noqa: ANN401
"""Bench: CUDA FCHL18 full_matvec with alchemy, fused vs unfused path.

full_matvec takes the fused path when N_train > N_query. The unfused path is forced by
splitting the training set into chunks of <= N_query molecules (each call then has
N_train_chunk <= N_query) and summing the partial results, which gives the same product.

Synthetic ethanol geometries (no dataset needed).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

import kernelforge.cuda_fchl18_kernel as ck
import kernelforge.fchl18_repr as repr_mod

N_TRAIN = 40
N_QUERY = 8
MAX_SIZE = 9
SIGMA = 2.5
KP: dict[str, Any] = {
    "two_body_scaling": float(np.sqrt(8)),
    "three_body_scaling": 1.6,
    "two_body_width": 0.2,
    "three_body_width": float(np.pi),
    "two_body_power": 4.0,
    "three_body_power": 2.0,
    "cut_start": 1.0,
    "cut_distance": 5.0,
    "fourier_order": 1,
    "use_atm": True,
}


def make_side(rng: np.random.Generator, n: int, base: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    coords = [base + rng.normal(scale=0.05, size=base.shape) for _ in range(n)]
    zs = [z] * n
    x, nn_, nnn = repr_mod.generate(coords, zs, max_size=MAX_SIZE, cut_distance=KP["cut_distance"])
    cp = np.stack(coords)
    zp = np.stack(zs).astype(np.int32)

    def cu(a: np.ndarray, dt: Any = np.float64) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a.astype(dt))).cuda()

    return {
        "x": cu(x),
        "n": cu(nn_, np.int32),
        "nn": cu(nnn, np.int32),
        "c": cu(cp),
        "z": cu(zp, np.int32),
        "nm": n,
        "na": len(z),
    }


def sl(s: dict[str, Any], i: int, j: int) -> dict[str, Any]:
    na = s["na"]
    return {
        "x": s["x"][i:j].contiguous(),
        "n": s["n"][i:j].contiguous(),
        "nn": s["nn"][i:j].contiguous(),
        "c": s["c"][i:j].contiguous(),
        "z": s["z"][i:j].contiguous(),
        "nm": j - i,
        "na": na,
    }


def call(
    q: dict[str, Any], t: dict[str, Any], aE: torch.Tensor, aF: torch.Tensor, **kw: Any
) -> Any:
    return ck.kernel_gaussian_full_matvec(
        q["x"], t["x"], q["n"], t["n"], q["nn"], t["nn"], q["c"], q["z"], t["c"], t["z"],
        aE, aF, sigma=SIGMA, **KP, **kw,
    )  # fmt: skip


def unfused(
    q: dict[str, Any], t: dict[str, Any], aE: torch.Tensor, aF: torch.Tensor, **kw: Any
) -> Any:
    e = torch.zeros(q["nm"], dtype=torch.float64, device="cuda")
    f = torch.zeros(q["nm"] * q["na"] * 3, dtype=torch.float64, device="cuda")
    step = q["nm"]
    for i in range(0, t["nm"], step):
        j = min(i + step, t["nm"])
        ts = sl(t, i, j)
        d = t["na"] * 3
        de, df = call(q, ts, aE[i:j].contiguous(), aF[i * d : j * d].contiguous(), **kw)
        e += de
        f += df
    return e, f


def timeit(fn: Any, n: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    rng = np.random.default_rng(0)
    z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    base = rng.normal(scale=1.2, size=(len(z), 3))
    train = make_side(rng, N_TRAIN, base, z)
    query = make_side(rng, N_QUERY, base, z)
    aE = torch.from_numpy(rng.standard_normal(N_TRAIN)).cuda()
    aF = torch.from_numpy(rng.standard_normal(N_TRAIN * len(z) * 3)).cuda()

    print(f"ethanol  N_train={N_TRAIN}  N_query={N_QUERY}  (ms per full_matvec call)")
    for name, kw in (
        ("alchemy OFF", {}),
        (
            "alchemy ON ",
            {"use_alchemy": True, "alchemy_period_width": 1.6, "alchemy_group_width": 1.6},
        ),
    ):
        e1, f1 = call(query, train, aE, aF, **kw)
        e2, f2 = unfused(query, train, aE, aF, **kw)
        err = max(float((e1 - e2).abs().max()), float((f1 - f2).abs().max()))
        t_f = timeit(lambda kw=kw: call(query, train, aE, aF, **kw))
        t_u = timeit(lambda kw=kw: unfused(query, train, aE, aF, **kw))
        print(
            f"{name}: fused {t_f:8.2f} ms | unfused(chunked x{N_TRAIN // N_QUERY}) {t_u:8.2f} ms"
            f" | speedup {t_u / t_f:5.2f}x | max|fused-unfused| {err:.2e}"
        )


if __name__ == "__main__":
    main()
