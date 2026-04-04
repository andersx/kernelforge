"""Type stub for the cuda_global_kernels CUDA extension module.

cuda_global_kernels is built only when both a CUDA compiler and PyTorch
are available at build time.  Import it with a try/except guard.

All tensors must be float32 and on the same CUDA device.
"""

import torch

def kernel_gaussian_full_symm(
    X: torch.Tensor,  # (N, M)    float32 CUDA
    dX: torch.Tensor,  # (N, D, M) float32 CUDA
    sigma: float,
) -> torch.Tensor:
    """Build the symmetric energy+force training kernel matrix on GPU.

    Parameters
    ----------
    X : torch.Tensor, shape (N, M), float32 CUDA
        Training descriptors.
    dX : torch.Tensor, shape (N, D, M), float32 CUDA
        Training Jacobians.
    sigma : float
        Gaussian length-scale.

    Returns
    -------
    K_full : torch.Tensor, shape (N*(1+D), N*(1+D)), float32 CUDA
        Fully symmetric kernel matrix (both triangles filled).

    Block layout (full = N*(1+D)):
        K_full[0:N,      0:N]      -- scalar   K_EE[a,b] = exp(-||x_a-x_b||^2/(2*sigma^2))
        K_full[N:,       0:N]      -- Jacobian K_FE[a*D+d, b]
        K_full[0:N,      N:]       -- Jacobian K_EF (transpose of K_FE)
        K_full[N:,       N:]       -- Hessian  K_FF[a*D+d1, b*D+d2]
    """
    ...

def kernel_gaussian_full_matvec(
    X_q: torch.Tensor,  # (N_q, M)    float32 CUDA
    dX_q: torch.Tensor,  # (N_q, D, M) float32 CUDA
    X_t: torch.Tensor,  # (N_t, M)    float32 CUDA
    alpha_E: torch.Tensor,  # (N_t,)      float32 CUDA
    alpha_desc_F: torch.Tensor,  # (N_t, M)    float32 CUDA
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contracted energy+force inference (J^T·alpha trick).

    Computes E and F predictions without materialising the full test-train
    kernel matrix.  Uses the same J^T·alpha contraction as the CPU
    ``kernel_gaussian_full_matvec``.

    ``alpha_desc_F`` must be precomputed once after training::

        alpha_desc_F = torch.einsum('ndm,nd->nm', dX_train, alpha_F)

    Parameters
    ----------
    X_q : torch.Tensor, shape (N_q, M), float32 CUDA
        Query descriptors.
    dX_q : torch.Tensor, shape (N_q, D, M), float32 CUDA
        Query Jacobians.
    X_t : torch.Tensor, shape (N_t, M), float32 CUDA
        Training descriptors.
    alpha_E : torch.Tensor, shape (N_t,), float32 CUDA
        Energy dual coefficients.
    alpha_desc_F : torch.Tensor, shape (N_t, M), float32 CUDA
        Contracted force weights: ``Σ_d J_train[m,d,:] * alpha_F[m,d]``.
    sigma : float
        Gaussian length-scale.

    Returns
    -------
    E_pred : torch.Tensor, shape (N_q,), float32 CUDA
    F_pred : torch.Tensor, shape (N_q, D), float32 CUDA
        Physical forces F = -dE/dR.
    """
    ...
