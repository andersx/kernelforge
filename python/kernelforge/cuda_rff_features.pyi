"""Type stubs for cuda_rff_features — GPU global RFF primitives (FP32)."""

import torch

def rff_features(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute ``sqrt(2/D) * cos(X @ W + b)`` on GPU.

    Parameters are float32 CUDA tensors: ``X`` is ``(N, M)``, ``W`` is
    ``(M, D)``, and ``b`` is ``(D,)``.  Returns ``Z`` with shape ``(N, D)``.
    """
    ...

def rff_gramian_symm_rfp(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    Y: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RFP-packed ``Z.T @ Z`` and ``Z.T @ Y`` for energy-only training."""
    ...

def rff_full_gramian_symm_rfp(
    X: torch.Tensor,
    dX: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    Y: torch.Tensor,
    F: torch.Tensor,
    energy_chunk: int = 256,
    force_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RFP-packed full energy+force normal equations."""
    ...

def rff_predict_energy(
    X: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute energy predictions ``Z @ weights`` in chunks on GPU."""
    ...

def rff_predict_force(
    X: torch.Tensor,
    dX: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute force predictions ``G @ weights`` in chunks on GPU."""
    ...

def rff_features_elemental(
    X: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute local elemental RFF features on GPU. Returns (nmol, D) row-major."""
    ...

def rff_features_elemental_col_major(
    X: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Compute local elemental RFF features on GPU. Returns (D, nmol) col-major."""
    ...

def rff_gradient_elemental(
    X: torch.Tensor,
    dX: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute local elemental RFF gradient feature matrix G (total_naq, D) on GPU."""
    ...

def rff_gradient_elemental_col_major(
    X: torch.Tensor,
    dX: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute local elemental RFF gradient feature matrix G (D, total_naq) col-major on GPU."""
    ...

def rff_gramian_elemental_rfp(
    X: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    Y: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RFP-packed local elemental RFF energy-only normal equations."""
    ...

def rff_predict_energy_elemental(
    X: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute local elemental RFF energy predictions on GPU."""
    ...

def rff_full_gramian_elemental_rfp(
    X: torch.Tensor,
    dX: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    Y: torch.Tensor,
    F: torch.Tensor,
    energy_chunk: int = 256,
    force_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RFP-packed local elemental RFF energy+force normal equations."""
    ...

def rff_predict_force_elemental(
    X: torch.Tensor,
    dX: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute local elemental RFF force predictions on GPU."""
    ...
