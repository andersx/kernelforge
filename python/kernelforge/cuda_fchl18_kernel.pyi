"""Type stubs for cuda_fchl18_kernel.

Current scope:
- fp32 and fp64
- rectangular energy-only kernel_gaussian
- symmetric energy-only kernel_gaussian_symm
- symmetric RFP kernel_gaussian_symm_rfp (TRANSR='N', UPLO='U')
- Jacobian kernel_gaussian_jacobian (dK/dR of the query molecules)
- Jacobian-T kernel_gaussian_jacobian_t (transposed store)
- Gradient kernel_gaussian_gradient (single-query, CPU G layout)
- Hessian kernel_gaussian_hessian (d2K/dR_A dR_B)
- Symmetric Hessian kernel_gaussian_hessian_symm / _symm_rfp
- Full combined kernel_gaussian_full / _symm / _symm_rfp
- inputs are precomputed FCHL18 representations on CUDA
"""

import torch

def kernel_gaussian(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    nn1: torch.Tensor,
    nn2: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
    use_atm: bool = True,
) -> torch.Tensor:
    """Compute the rectangular FCHL18 Gaussian kernel matrix on GPU.

    Parameters
    ----------
    x1, x2 : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
        Precomputed FCHL18 representations. Dtypes must match.
    n1, n2 : torch.Tensor, shape (nm,), int32, CUDA
        Number of real atoms per molecule.
    nn1, nn2 : torch.Tensor, shape (nm, max_size), int32, CUDA
        Number of neighbors per atom.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    torch.Tensor, shape (nm1, nm2), same dtype as x1, CUDA
        Rectangular energy-only kernel matrix.
    """
    ...

def kernel_gaussian_symm(
    x: torch.Tensor,
    n: torch.Tensor,
    nn: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
    use_atm: bool = True,
) -> torch.Tensor:
    """Compute the symmetric FCHL18 Gaussian kernel matrix on GPU.

    Only upper-triangle molecule pairs (a <= b) are evaluated; K[b,a] is mirrored.

    Parameters
    ----------
    x : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
        Precomputed FCHL18 representations.
    n : torch.Tensor, shape (nm,), int32, CUDA
        Number of real atoms per molecule.
    nn : torch.Tensor, shape (nm, max_size), int32, CUDA
        Number of neighbors per atom.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    torch.Tensor, shape (nm, nm), same dtype as x, CUDA
        Symmetric energy-only kernel matrix.
    """
    ...

def kernel_gaussian_symm_rfp(
    x: torch.Tensor,
    n: torch.Tensor,
    nn: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
    use_atm: bool = True,
) -> torch.Tensor:
    """Compute the symmetric FCHL18 Gaussian kernel in RFP packed format on GPU.

    Only upper-triangle molecule pairs (a <= b) are evaluated. Layout is
    TRANSR='N', UPLO='U' (kf::rfp_index_upper_N). Compatible with
    kernelmath.cho_solve_rfp. Unpack with:
    kernelmath.rfp_to_full(K_rfp, nm, uplo='L', transr='N').

    Parameters
    ----------
    x : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
        Precomputed FCHL18 representations.
    n : torch.Tensor, shape (nm,), int32, CUDA
        Number of real atoms per molecule.
    nn : torch.Tensor, shape (nm, max_size), int32, CUDA
        Number of neighbors per atom.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    torch.Tensor, shape (nm*(nm+1)//2,), same dtype as x, CUDA
        Upper-triangle RFP-packed energy-only kernel.
    """
    ...

def kernel_gaussian_jacobian(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    nn1: torch.Tensor,
    nn2: torch.Tensor,
    coords1: torch.Tensor,
    z1: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
    use_atm: bool = True,
) -> torch.Tensor:
    """Compute the FCHL18 Jacobian kernel dK/dR_A on GPU.

    J[row, b] = dK(A_a, B_b) / dR_{A_a}[alpha, mu] with
    row = row_offset[a] + alpha*3 + mu and row_offset[a] = 3 * sum_{a' < a} n1[a'].
    Matches CPU ``kernelforge.fchl18_kernel.kernel_gaussian_jacobian``.

    Unlike the CPU function, which takes raw coordinate lists and builds the query
    representation internally, this takes a precomputed ``x1`` (as
    ``kernel_gaussian`` does) plus ``coords1``/``z1``. The coordinates are needed to
    map each neighbour slot of the representation back to an atom index for the
    chain rule; they must be the same padded coordinates the representation was
    generated from.

    Parameters
    ----------
    x1, x2 : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
        Precomputed FCHL18 representations. Dtypes must match.
    n1, n2 : torch.Tensor, shape (nm,), int32, CUDA
        Number of real atoms per molecule.
    nn1, nn2 : torch.Tensor, shape (nm, max_size), int32, CUDA
        Number of neighbors per atom.
    coords1 : torch.Tensor, shape (nm1, max_size1, 3), same dtype as x1, CUDA
        Padded Cartesian coordinates of the query molecules.
    z1 : torch.Tensor, shape (nm1, max_size1), int32, CUDA
        Padded nuclear charges of the query molecules.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    torch.Tensor, shape (3 * sum(n1), nm2), same dtype as x1, CUDA
        Jacobian of the energy kernel with respect to the query coordinates.
    """
    ...

def kernel_gaussian_jacobian_t(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    nn1: torch.Tensor,
    nn2: torch.Tensor,
    coords1: torch.Tensor,
    z1: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
    use_atm: bool = True,
) -> torch.Tensor:
    """Compute the FCHL18 Jacobian-T kernel on GPU.

    Same computation as ``kernel_gaussian_jacobian``, but stores
    ``Jt[b, row] = dK(A_a, B_b) / dR_{A_a}[...]`` with shape ``(nm2, 3*sum(n1))``.
    Matches CPU ``kernelforge.fchl18_kernel.kernel_gaussian_jacobian_t``.
    """
    ...

def kernel_gaussian_gradient(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    nn1: torch.Tensor,
    nn2: torch.Tensor,
    coords1: torch.Tensor,
    z1: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 0.5,
    cut_distance: float = 1e6,
    fourier_order: int = 2,
    use_atm: bool = True,
) -> torch.Tensor:
    """Compute the FCHL18 gradient kernel for a single query molecule on GPU.

    Requires ``nm1 == 1``. Returns ``G`` with shape ``(n_atoms_A, 3, nm2)`` matching
    CPU ``kernelforge.fchl18_kernel.kernel_gaussian_gradient``.
    """
    ...

def kernel_gaussian_hessian(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    nn1: torch.Tensor,
    nn2: torch.Tensor,
    coords1: torch.Tensor,
    z1: torch.Tensor,
    coords2: torch.Tensor,
    z2: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 1.0,
    cut_distance: float = 1e6,
    fourier_order: int = 1,
    use_atm: bool = False,
) -> torch.Tensor:
    """Compute the FCHL18 Hessian kernel d2K/dR_A dR_B on GPU.

    ``H[row, col] = d2K(A_a, B_b) / dR_{A_a}[alpha, mu] dR_{B_b}[beta, nu]`` with
    ``row = row_offset[a] + alpha*3 + mu``, ``col = col_offset[b] + beta*3 + nu``,
    ``row_offset[a] = 3 * sum_{a' < a} n1[a']`` and
    ``col_offset[b] = 3 * sum_{b' < b} n2[b']``.
    Matches CPU ``kernelforge.fchl18_kernel.kernel_gaussian_hessian``.

    Like ``kernel_gaussian_jacobian``, both sides take a precomputed representation
    plus the padded coordinates and nuclear charges the representation was generated
    from; those are needed to map neighbour slots back to atom indices for the chain
    rule.

    The underlying cross-Hessian is only exact when the cutoff is inactive
    (``cut_start >= 1`` or ``cut_distance`` far beyond every interatomic distance)
    and ``use_atm`` is False, matching the CPU implementation.

    Parameters
    ----------
    x1, x2 : torch.Tensor, shape (nm, max_size, 5, max_size), float32 or float64, CUDA
        Precomputed FCHL18 representations. Dtypes must match.
    n1, n2 : torch.Tensor, shape (nm,), int32, CUDA
        Number of real atoms per molecule.
    nn1, nn2 : torch.Tensor, shape (nm, max_size), int32, CUDA
        Number of neighbors per atom.
    coords1, coords2 : torch.Tensor, shape (nm, max_size, 3), same dtype as x1, CUDA
        Padded Cartesian coordinates of each side's molecules.
    z1, z2 : torch.Tensor, shape (nm, max_size), int32, CUDA
        Padded nuclear charges of each side's molecules.
    sigma : float
        Gaussian kernel length-scale.

    Returns
    -------
    torch.Tensor, shape (3 * sum(n1), 3 * sum(n2)), same dtype as x1, CUDA
        Force-force block matrix, one dense (3*n1[a], 3*n2[b]) block per molecule pair.
    """
    ...

def kernel_gaussian_hessian_symm(
    x: torch.Tensor,
    n: torch.Tensor,
    nn: torch.Tensor,
    coords: torch.Tensor,
    z: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 1.0,
    cut_distance: float = 1e6,
    fourier_order: int = 1,
    use_atm: bool = False,
) -> torch.Tensor:
    """Compute the symmetric FCHL18 Hessian kernel on GPU.

    Shape ``(D, D)`` with ``D = 3*sum(n)``. Matches CPU
    ``kernelforge.fchl18_kernel.kernel_gaussian_hessian_symm``.
    """
    ...

def kernel_gaussian_hessian_symm_rfp(
    x: torch.Tensor,
    n: torch.Tensor,
    nn: torch.Tensor,
    coords: torch.Tensor,
    z: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 1.0,
    cut_distance: float = 1e6,
    fourier_order: int = 1,
    use_atm: bool = False,
) -> torch.Tensor:
    """Compute the symmetric FCHL18 Hessian kernel in RFP format on GPU.

    ``TRANSR='N'``, ``UPLO='U'``. Length ``D*(D+1)/2`` with ``D = 3*sum(n)``.
    Unpack with ``kernelmath.rfp_to_full(..., uplo='L', transr='N')``.
    """
    ...

def kernel_gaussian_full(
    x1: torch.Tensor,
    x2: torch.Tensor,
    n1: torch.Tensor,
    n2: torch.Tensor,
    nn1: torch.Tensor,
    nn2: torch.Tensor,
    coords1: torch.Tensor,
    z1: torch.Tensor,
    coords2: torch.Tensor,
    z2: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 1.0,
    cut_distance: float = 1e6,
    fourier_order: int = 1,
    use_atm: bool = False,
) -> torch.Tensor:
    """Compute the full FCHL18 energy+force kernel matrix on GPU.

    Block layout (``N_A=nm1``, ``N_B=nm2``, ``D_A=3*sum(n1)``, ``D_B=3*sum(n2)``):
      ``K[0:N_A, 0:N_B]`` scalar
      ``K[0:N_A, N_B:]``  jacobian_t (dK/dR_B)
      ``K[N_A:,  0:N_B]`` jacobian   (dK/dR_A)
      ``K[N_A:,  N_B:]``  hessian

    Returns
    -------
    torch.Tensor, shape (N_A+D_A, N_B+D_B), same dtype as x1, CUDA
    """
    ...

def kernel_gaussian_full_symm(
    x: torch.Tensor,
    n: torch.Tensor,
    nn: torch.Tensor,
    coords: torch.Tensor,
    z: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 1.0,
    cut_distance: float = 1e6,
    fourier_order: int = 1,
    use_atm: bool = False,
) -> torch.Tensor:
    """Compute the symmetric full FCHL18 energy+force kernel on GPU.

    Returns
    -------
    torch.Tensor, shape (N+D, N+D), same dtype as x, CUDA
        with ``N=nm`` and ``D=3*sum(n)``.
    """
    ...

def kernel_gaussian_full_symm_rfp(
    x: torch.Tensor,
    n: torch.Tensor,
    nn: torch.Tensor,
    coords: torch.Tensor,
    z: torch.Tensor,
    sigma: float,
    two_body_scaling: float = 2.0,
    two_body_width: float = 0.1,
    two_body_power: float = 6.0,
    three_body_scaling: float = 2.0,
    three_body_width: float = 3.0,
    three_body_power: float = 3.0,
    cut_start: float = 1.0,
    cut_distance: float = 1e6,
    fourier_order: int = 1,
    use_atm: bool = False,
) -> torch.Tensor:
    """Compute the symmetric full FCHL18 kernel in RFP packed format on GPU.

    TRANSR='N', UPLO='U'. Unpack with
    ``kernelmath.rfp_to_full(..., uplo='L', transr='N')``.

    Returns
    -------
    torch.Tensor, shape (BIG*(BIG+1)//2,), same dtype as x, CUDA
        with ``BIG = nm + 3*sum(n)``.
    """
    ...
