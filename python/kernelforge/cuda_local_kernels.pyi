"""Type stubs for cuda_local_kernels — CUDA Gaussian kernels for FCHL19 descriptors."""

import torch

def kernel_gaussian_full_symm(
    X: torch.Tensor,
    dX: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Build the symmetric (nm+naq)² energy+force training kernel matrix.

    Parameters
    ----------
    X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
        Training atom descriptors (padded with zeros beyond N[m]).
    dX : torch.Tensor, shape (nm, max_atoms, rep, 3*max_atoms), float32, CUDA
        Training Jacobians (padded).
    Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
        Atomic labels (nuclear charges; zero-padded).
    N : torch.Tensor, shape (nm,), int32, CUDA
        Active atom counts per molecule.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    K_full : torch.Tensor, shape (nm+naq, nm+naq), float32, CUDA
        Fully symmetric kernel matrix.  naq = 3 * sum(N).
    """
    ...

def compute_alpha_desc(
    dX: torch.Tensor,
    N: torch.Tensor,
    alpha_F: torch.Tensor,
) -> torch.Tensor:
    """Precompute descriptor-space force weights for the J^T·alpha trick.

    alpha_desc[b, i2, k] = sum_{c=0}^{3*N[b]-1} dX[b, i2, k, c] * alpha_F[offs[b]+c]

    Call once after the KRR solve.  Pass the result to
    ``kernel_gaussian_full_matvec`` at inference time.

    Parameters
    ----------
    dX : torch.Tensor, shape (nm, max_atoms, rep, 3*max_atoms), float32, CUDA
        Training Jacobians.
    N : torch.Tensor, shape (nm,), int32, CUDA
        Active atom counts.
    alpha_F : torch.Tensor, shape (naq,), float32, CUDA
        Force dual coefficients from the KRR solve (alpha[nm:]).

    Returns
    -------
    alpha_desc : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
    """
    ...

def kernel_gaussian_full_matvec(
    X_q: torch.Tensor,
    dX_q: torch.Tensor,
    Q_q: torch.Tensor,
    N_q: torch.Tensor,
    X_t: torch.Tensor,
    Q_t: torch.Tensor,
    N_t: torch.Tensor,
    alpha_E: torch.Tensor,
    alpha_desc: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contracted energy+force inference (J^T·alpha trick, local version).

    Does not materialise the full test-train kernel matrix.

    Parameters
    ----------
    X_q : torch.Tensor, shape (nm_q, max_atoms_q, rep), float32, CUDA
        Query atom descriptors.
    dX_q : torch.Tensor, shape (nm_q, max_atoms_q, rep, 3*max_atoms_q), float32, CUDA
        Query Jacobians.
    Q_q : torch.Tensor, shape (nm_q, max_atoms_q), int32, CUDA
        Query atomic labels.
    N_q : torch.Tensor, shape (nm_q,), int32, CUDA
        Query active atom counts.
    X_t : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
        Training atom descriptors.
    Q_t : torch.Tensor, shape (nm_t, max_atoms_t), int32, CUDA
        Training atomic labels.
    N_t : torch.Tensor, shape (nm_t,), int32, CUDA
        Training active atom counts.
    alpha_E : torch.Tensor, shape (nm_t,), float32, CUDA
        Energy dual coefficients.
    alpha_desc : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
        Precomputed force weights (from ``compute_alpha_desc``).
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    E_pred : torch.Tensor, shape (nm_q,), float32, CUDA
        Predicted energies (baseline-subtracted).
    F_pred : torch.Tensor, shape (naq_q,), float32, CUDA
        Predicted forces as flat Cartesian vector.  naq_q = 3*sum(N_q).
    """
    ...
