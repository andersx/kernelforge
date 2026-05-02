"""Type stubs for cuda_local_kernels — CUDA Gaussian kernels for FCHL19 local descriptors.

Implements atom-pair-wise kernels with label-based screening (z1 == z2).
All functions operate on CUDA (GPU) tensors.
"""

import torch

def kernel_gaussian_rect(
    X_q: torch.Tensor,
    Q_q: torch.Tensor,
    N_q: torch.Tensor,
    X_t: torch.Tensor,
    Q_t: torch.Tensor,
    N_t: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Build kernel matrix for two molecule sets (asymmetric/rectangular case).

    Parameters
    ----------
    X_q : torch.Tensor, shape (nm_q, max_atoms_q, rep), float32, CUDA
        Query descriptor vectors.
    Q_q : torch.Tensor, shape (nm_q, max_atoms_q), int32, CUDA
        Query atomic labels.
    N_q : torch.Tensor, shape (nm_q,), int32, CUDA
        Query active atom counts.
    X_t : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
        Training descriptor vectors.
    Q_t : torch.Tensor, shape (nm_t, max_atoms_t), int32, CUDA
        Training atomic labels.
    N_t : torch.Tensor, shape (nm_t,), int32, CUDA
        Training active atom counts.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    K : torch.Tensor, shape (nm1, nm2), float64, CUDA
        Kernel matrix aggregated by label-matching atom pairs.
    """
    ...

def kernel_gaussian_symm(
    X: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Build symmetric kernel matrix for a single molecule set.

    Parameters
    ----------
    X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
        Descriptor vectors.
    Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
        Atomic labels.
    N : torch.Tensor, shape (nm,), int32, CUDA
        Active atom counts per molecule.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    K_symm : torch.Tensor, shape (nm, nm), float64, CUDA
        Symmetric kernel matrix (both triangles filled).
    """
    ...

def kernel_gaussian_symm_rfp(
    X: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Build the symmetric energy-only kernel matrix K_EE in RFP packed format.

    Uses TRANSR=N, UPLO=L convention.
    Unpack with: kernelmath.rfp_to_full(K_rfp, nm, uplo='U', transr='N')

    Parameters
    ----------
    X : torch.Tensor, shape (nm, max_atoms, rep), float32, CUDA
        Descriptor vectors.
    Q : torch.Tensor, shape (nm, max_atoms), int32, CUDA
        Atomic labels.
    N : torch.Tensor, shape (nm,), int32, CUDA
        Active atom counts per molecule.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    K_rfp : torch.Tensor, shape (nm*(nm+1)//2,), float32, CUDA
        Lower-triangle kernel packed in RFP format.
    """
    ...

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

def kernel_gaussian_full_symm_rfp(
    X: torch.Tensor,
    dX: torch.Tensor,
    Q: torch.Tensor,
    N: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Build the symmetric energy+force training kernel matrix in RFP packed format.

    Uses TRANSR=N, UPLO=L convention.  BIG = nm + naq, naq = 3*sum(N).
    Unpack with: kernelmath.rfp_to_full(K_rfp, BIG, uplo='U', transr='N')

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
    K_rfp : torch.Tensor, shape (BIG*(BIG+1)//2,), float32, CUDA
        Lower-triangle kernel packed in RFP format.
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
    alpha_desc_F: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contracted energy+force inference (J^T·alpha trick, local version with label screening).

    Does not materialise the full test-train kernel matrix.
    Uses label-based atom screening (z1 == z2) for efficiency.

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
    alpha_desc_F : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
        Precomputed force weights (from ``compute_alpha_desc``).
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    E_pred : torch.Tensor, shape (nm_q,), float32, CUDA
        Predicted energies.
    F_pred : torch.Tensor, shape (naq_q,), float32, CUDA
        Predicted forces as flat Cartesian vector.  naq_q = 3*sum(N_q).
    """
    ...

def precompute_train(
    X_t: torch.Tensor,
    Q_t: torch.Tensor,
    N_t: torch.Tensor,
    alpha_E: torch.Tensor,
    alpha_desc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute training-side constants for repeated inference (e.g. MD simulation).

    Call once after fitting.  The returned tensors are constant as long as the
    training data (X_t, alpha_E, alpha_desc) does not change.  Pass them to
    ``kernel_gaussian_full_matvec_cached`` at every inference step.

    Parameters
    ----------
    X_t : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
    Q_t : torch.Tensor, shape (nm_t, max_atoms_t), int32, CUDA
    N_t : torch.Tensor, shape (nm_t,), int32, CUDA
    alpha_E : torch.Tensor, shape (nm_t,), float32, CUDA
    alpha_desc : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA

    Returns
    -------
    norms_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
        Squared norms ||X_t[t]||².
    S_adF : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
        Dot products X_t[t] · alpha_desc[t].
    alpha_E_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
        Atom-expanded energy dual coefficients alpha_E[mol(t)].
    combined_t : torch.Tensor, shape (nm_t * max_atoms_t, rep), float32, CUDA
        Combined matrix alpha_desc + alpha_E_t[t] * X_t.
    """
    ...

def kernel_gaussian_full_matvec_cached(
    X_q: torch.Tensor,
    dX_q: torch.Tensor,
    Q_q: torch.Tensor,
    N_q: torch.Tensor,
    X_t: torch.Tensor,
    Q_t: torch.Tensor,
    N_t: torch.Tensor,
    alpha_E: torch.Tensor,
    alpha_desc: torch.Tensor,
    norms_t: torch.Tensor,
    S_adF: torch.Tensor,
    alpha_E_t: torch.Tensor,
    combined_t: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached energy+force inference for repeated calls (MD simulation).

    Variant of ``kernel_gaussian_full_matvec`` that accepts precomputed
    training-side constants from ``precompute_train``.  Eliminates per-step
    cudaMalloc/cudaFree overhead and redundant kernel launches for norms_t,
    S_adF, and combined_t.

    Parameters
    ----------
    X_q : torch.Tensor, shape (nm_q, max_atoms_q, rep), float32, CUDA
    dX_q : torch.Tensor, shape (nm_q, max_atoms_q, rep, 3*max_atoms_q), float32, CUDA
    Q_q : torch.Tensor, shape (nm_q, max_atoms_q), int32, CUDA
    N_q : torch.Tensor, shape (nm_q,), int32, CUDA
    X_t : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
    Q_t : torch.Tensor, shape (nm_t, max_atoms_t), int32, CUDA
    N_t : torch.Tensor, shape (nm_t,), int32, CUDA
    alpha_E : torch.Tensor, shape (nm_t,), float32, CUDA
    alpha_desc : torch.Tensor, shape (nm_t, max_atoms_t, rep), float32, CUDA
    norms_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
        From ``precompute_train``.
    S_adF : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
        From ``precompute_train``.
    alpha_E_t : torch.Tensor, shape (nm_t * max_atoms_t,), float32, CUDA
        From ``precompute_train``.
    combined_t : torch.Tensor, shape (nm_t * max_atoms_t, rep), float32, CUDA
        From ``precompute_train``.
    sigma : float

    Returns
    -------
    E_pred : torch.Tensor, shape (nm_q,), float32, CUDA
    F_pred : torch.Tensor, shape (naq_q,), float32, CUDA
    """
    ...

def kernel_gaussian_full_matvec_cached_graph(
    X_q: torch.Tensor,
    dX_q: torch.Tensor,
    Q_q: torch.Tensor,
    N_q: torch.Tensor,
    X_t: torch.Tensor,
    Q_t: torch.Tensor,
    N_t: torch.Tensor,
    alpha_E: torch.Tensor,
    alpha_desc: torch.Tensor,
    norms_t: torch.Tensor,
    S_adF: torch.Tensor,
    alpha_E_t: torch.Tensor,
    combined_t: torch.Tensor,
    query_indices_by_group: list[torch.Tensor],
    query_counts_by_group: list[int],
    E_pred: torch.Tensor,
    F_pred: torch.Tensor,
    sigma: float,
) -> None:
    """Graph-capture-friendly cached energy+force inference.

    Uses caller-provided output buffers and fixed per-element query index
    tensors so the cached matvec can be captured and replayed via a CUDA graph
    for repeated fixed-shape inference.
    """
    ...

def kernel_gaussian_full_matvec_cached_graph_with_offsets(
    X_q: torch.Tensor,
    dX_q: torch.Tensor,
    Q_q: torch.Tensor,
    N_q: torch.Tensor,
    offs_q: torch.Tensor,
    X_t: torch.Tensor,
    Q_t: torch.Tensor,
    N_t: torch.Tensor,
    alpha_E: torch.Tensor,
    alpha_desc: torch.Tensor,
    norms_t: torch.Tensor,
    S_adF: torch.Tensor,
    alpha_E_t: torch.Tensor,
    combined_t: torch.Tensor,
    query_indices_by_group: list[torch.Tensor],
    query_counts_by_group: list[int],
    E_pred: torch.Tensor,
    F_pred: torch.Tensor,
    sigma: float,
) -> None:
    """Graph-capture-friendly cached energy+force inference with query offsets.

    Uses caller-provided output buffers, fixed per-element query index tensors,
    and precomputed query Cartesian offsets to avoid host synchronization during
    CUDA graph capture and replay.
    """
    ...
