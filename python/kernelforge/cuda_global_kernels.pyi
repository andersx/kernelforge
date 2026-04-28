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

def kernel_gaussian_full_symm_rfp(
    X: torch.Tensor,  # (N, M)    float32 CUDA
    dX: torch.Tensor,  # (N, D, M) float32 CUDA
    sigma: float,
) -> torch.Tensor:
    """Build the energy+force kernel matrix directly in RFP packed format.

    Like kernel_gaussian_full_symm but stores the result as a 1-D RFP
    packed buffer (TRANSR=N, UPLO=L) instead of a dense BIG x BIG matrix.
    Each lower-triangle element is written once via rfp_index_lower_N;
    no dense intermediate is allocated and no mirror step is performed.

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
    K_rfp : torch.Tensor, shape (BIG*(BIG+1)//2,), float32 CUDA
        Lower-triangular RFP packed kernel matrix (TRANSR=N, UPLO=L),
        where BIG = N*(1+D).
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

def kernel_gaussian_symm_rfp(
    X: torch.Tensor,  # (N, M) float32 CUDA
    sigma: float,
) -> torch.Tensor:
    """Build the energy-only NxN Gaussian kernel matrix in RFP packed format.

    Uses TRANSR=N, UPLO=L convention.  The packed buffer has N*(N+1)//2
    elements and is compatible with rfp_potrf / rfp_potrs.

    Parameters
    ----------
    X : torch.Tensor, shape (N, M), float32 CUDA
        Training descriptors.
    sigma : float
        Gaussian length-scale.

    Returns
    -------
    K_rfp : torch.Tensor, shape (N*(N+1)//2,), float32 CUDA
        Lower-triangular RFP packed kernel matrix (TRANSR=N, UPLO=L).
    """
    ...

def rfp_potrf(
    K_rfp: torch.Tensor,  # (N*(N+1)//2,) float32 CUDA, modified in-place
    N: int,
    l2: float = 0.0,
) -> int:
    """Cholesky factorisation of an RFP-packed symmetric positive definite matrix.

    Optionally adds l2 to the diagonal (Tikhonov regularisation) before
    factorising.  The input buffer is overwritten with the lower Cholesky factor L.
    Convention: TRANSR=N, UPLO=L.

    Parameters
    ----------
    K_rfp : torch.Tensor, shape (N*(N+1)//2,), float32 CUDA
        RFP-packed matrix.  Modified in-place.
    N : int
        Matrix dimension.
    l2 : float, default 0.0
        Diagonal regularisation added before factorisation.

    Returns
    -------
    info : int
        0 on success; positive value k means the k-th leading minor is not
        positive definite.
    """
    ...

def rfp_potrs(
    L_rfp: torch.Tensor,  # (N*(N+1)//2,) float32 CUDA
    B: torch.Tensor,  # (N, nrhs) float32 CUDA, modified in-place
) -> None:
    """Triangular solve using a Cholesky factor from rfp_potrf.

    Solves (L * L^T) * X = B, overwriting B with the solution X.
    Convention: TRANSR=N, UPLO=L.

    Parameters
    ----------
    L_rfp : torch.Tensor, shape (N*(N+1)//2,), float32 CUDA
        Lower Cholesky factor in RFP format (output of rfp_potrf).
    B : torch.Tensor, shape (N, nrhs), float32 CUDA
        Right-hand side.  Overwritten with the solution on exit.
    """
    ...
