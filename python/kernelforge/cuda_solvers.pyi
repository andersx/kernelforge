import torch

def cuda_solve_svd(
    Z: torch.Tensor,
    y: torch.Tensor,
    rcond: float = 0.0,
    z_col_major: bool = False,
) -> torch.Tensor:
    """Solve min_w ||Z @ w - y||_2 via truncated SVD (GPU, FP32).

    Z is (m, n) float32 torch.Tensor, y is (m,) float32 torch.Tensor.
    If z_col_major=True, Z is (n, m) col-major (avoids an internal transpose).
    rcond: singular values < rcond * S_max are treated as zero.
           rcond <= 0 uses machine-epsilon heuristic (same as numpy default).
    Returns w: (n,) float32 CPU tensor.
    """
    ...

def cuda_solve_qr(
    Z: torch.Tensor,
    y: torch.Tensor,
    z_col_major: bool = False,
) -> torch.Tensor:
    """Solve min_w ||Z @ w - y||_2 via QR (GPU, FP32).

    Z is (m, n) float32 torch.Tensor, y is (m,) float32 torch.Tensor, m >= n.
    If z_col_major=True, Z is (n, m) col-major (avoids an internal transpose).
    Returns w: (n,) float32 CPU tensor.
    """
    ...

def cuda_solve_gels(
    Z: torch.Tensor,
    y: torch.Tensor,
    z_col_major: bool = False,
    variant: str = "SS",
) -> torch.Tensor:
    """Solve min_w ||Z @ w - y||_2 via cusolverDn<variant>gels IRS (GPU, FP32 output).

    Z is (m, n) float32 torch.Tensor, y is (m,) float32 torch.Tensor, m >= n.
    If z_col_major=True, Z is (n, m) col-major (avoids an internal transpose).
    variant selects the internal precision of the IRS solver:
      'SS' — single/single (default)
      'SH' — single/half
      'SB' — single/bfloat16
      'SX' — single/tensorfloat32
    All variants produce float32 output.
    No rcond truncation — use cuda_solve_svd for rank-deficient systems.
    Returns w: (n,) float32 CPU tensor.
    """
    ...

