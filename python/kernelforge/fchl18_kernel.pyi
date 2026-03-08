from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def kernel_gaussian(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    n1: NDArray[np.int32],
    n2: NDArray[np.int32],
    nn1: NDArray[np.int32],
    nn2: NDArray[np.int32],
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
) -> NDArray[np.float64]:
    """FCHL18 Gaussian kernel matrix (asymmetric).

    K[a, b] = sum_{i in mol_a, j in mol_b: Z_i == Z_j}
                  exp( -(s_ii + s_jj - 2*s_ij) / sigma^2 )

    Parameters
    ----------
    x1, x2 : ndarray, shape (nm, max_size, 5, max_size), float64
        Representations from fchl18_repr.generate().
    n1, n2 : ndarray, shape (nm,), int32
        Number of real atoms per molecule.
    nn1, nn2 : ndarray, shape (nm, max_size), int32
        Number of neighbours per atom.
    sigma : float
        Gaussian kernel width.
    two_body_scaling : float, default 2.0
    two_body_width : float, default 0.1
    two_body_power : float, default 6.0
    three_body_scaling : float, default 2.0
    three_body_width : float, default 3.0
    three_body_power : float, default 3.0
    cut_start : float, default 0.5
    cut_distance : float, default 1e6
    fourier_order : int, default 2
    use_atm : bool, default True
        If False, replace the Axilrod-Teller-Muto factor (1+3*cos_i*cos_j*cos_k)
        with 1.0 in the three-body weight.

    Returns
    -------
    ndarray, shape (nm1, nm2), float64
    """
    ...

def kernel_gaussian_gradient(
    coords_A: NDArray[np.float64],
    z_A: NDArray[np.int32],
    x2: NDArray[np.float64],
    n2: NDArray[np.int32],
    nn2: NDArray[np.int32],
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
) -> NDArray[np.float64]:
    """Analytical gradient of the FCHL18 kernel w.r.t. coordinates of molecule A.

    G[alpha, mu, b] = dK[A,b] / dR_A[alpha, mu]

    Parameters
    ----------
    coords_A : ndarray, shape (n_atoms_A, 3), float64
        Cartesian coordinates of the query molecule A.
    z_A : ndarray, shape (n_atoms_A,), int32
        Nuclear charges of molecule A.
    x2 : ndarray, shape (nm2, max_size2, 5, max_size2), float64
        Pre-computed representations of training set B.
    n2 : ndarray, shape (nm2,), int32
        Number of real atoms per training molecule.
    nn2 : ndarray, shape (nm2, max_size2), int32
        Neighbour counts per atom in training set.
    sigma : float
        Gaussian kernel width.
    two_body_scaling : float, default 2.0
    two_body_width : float, default 0.1
    two_body_power : float, default 6.0
    three_body_scaling : float, default 2.0
    three_body_width : float, default 3.0
    three_body_power : float, default 3.0
    cut_start : float, default 0.5
    cut_distance : float, default 1e6
    fourier_order : int, default 2
    use_atm : bool, default True

    Returns
    -------
    ndarray, shape (n_atoms_A, 3, nm2), float64
        G[alpha, mu, b] = dK[A,b] / dR_A[alpha, mu]
    """
    ...

def kernel_gaussian_symm(
    x: NDArray[np.float64],
    n: NDArray[np.int32],
    nn: NDArray[np.int32],
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
) -> NDArray[np.float64]:
    """FCHL18 Gaussian kernel matrix (symmetric, K[a,b] == K[b,a]).

    Parameters
    ----------
    x : ndarray, shape (nm, max_size, 5, max_size), float64
    n : ndarray, shape (nm,), int32
    nn : ndarray, shape (nm, max_size), int32
    sigma : float
    (remaining hyperparameters same as kernel_gaussian)
    use_atm : bool, default True
        If False, replace the Axilrod-Teller-Muto factor (1+3*cos_i*cos_j*cos_k)
        with 1.0 in the three-body weight.

    Returns
    -------
    ndarray, shape (nm, nm), float64
    """
    ...

def kernel_gaussian_hessian(
    coords_A_list: Sequence[NDArray[np.float64]],
    z_A_list: Sequence[NDArray[np.int32]],
    coords_B_list: Sequence[NDArray[np.float64]],
    z_B_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Mixed second derivative of the FCHL18 kernel (Hessian kernel), contracted.

    H[row, col] = d²K[A,B] / dR_A[flat_row] dR_B[flat_col]

    Returns the force-force kernel block matrix used in energy+force KRR.

    Parameters
    ----------
    coords_A_list : list of ndarray, each (n_atoms_i, 3), float64
    z_A_list      : list of ndarray, each (n_atoms_i,), int32
    coords_B_list : list of ndarray, each (n_atoms_j, 3), float64
    z_B_list      : list of ndarray, each (n_atoms_j,), int32
    sigma : float
    two_body_scaling : float, default 2.0
    two_body_width : float, default 0.1
    two_body_power : float, default 6.0
    three_body_scaling : float, default 2.0
    three_body_width : float, default 3.0
    three_body_power : float, default 3.0
    cut_start : float, default 1.0
    cut_distance : float, default 1e6
    fourier_order : int, default 1
    use_atm : bool, default False

    Returns
    -------
    ndarray, shape (D_A, D_B), float64
        D_A = sum_i n_atoms_i * 3,  D_B = sum_j n_atoms_j * 3.
    """
    ...

def kernel_gaussian_hessian_symm(
    coords_list: Sequence[NDArray[np.float64]],
    z_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Symmetric FCHL18 Hessian kernel (full matrix).

    Equivalent to kernel_gaussian_hessian(mols, mols, ...) but exploits symmetry.

    Returns
    -------
    ndarray, shape (D, D), float64
        D = sum_i n_atoms_i * 3.
    """
    ...

def kernel_gaussian_hessian_symm_rfp(
    coords_list: Sequence[NDArray[np.float64]],
    z_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Symmetric FCHL18 Hessian kernel in RFP format.

    Returns
    -------
    ndarray, shape (D*(D+1)//2,), float64
        Upper-triangle RFP-packed Hessian kernel.
    """
    ...

def kernel_gaussian_symm_rfp(
    coords_list: Sequence[NDArray[np.float64]],
    z_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Symmetric FCHL18 scalar kernel in RFP format.

    Returns
    -------
    ndarray, shape (N*(N+1)//2,), float64
        Upper-triangle RFP-packed scalar kernel.
    """
    ...

def kernel_gaussian_jacobian(
    coords_A_list: Sequence[NDArray[np.float64]],
    z_A_list: Sequence[NDArray[np.int32]],
    x2: NDArray[np.float64],
    n2: NDArray[np.int32],
    nn2: NDArray[np.int32],
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
) -> NDArray[np.float64]:
    """FCHL18 Jacobian kernel dK/dR_A.

    Returns
    -------
    ndarray, shape (D_A, N_B), float64
        D_A = sum_i n_atoms_i * 3.
    """
    ...

def kernel_gaussian_jacobian_t(
    coords_train_list: Sequence[NDArray[np.float64]],
    z_train_list: Sequence[NDArray[np.int32]],
    x_test: NDArray[np.float64],
    n_test: NDArray[np.int32],
    nn_test: NDArray[np.int32],
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
) -> NDArray[np.float64]:
    """FCHL18 Jacobian-transpose kernel for energy prediction from force coefficients.

    Returns
    -------
    ndarray, shape (N_test, D_total), float64
    """
    ...

def kernel_gaussian_full(
    coords_A_list: Sequence[NDArray[np.float64]],
    z_A_list: Sequence[NDArray[np.int32]],
    coords_B_list: Sequence[NDArray[np.float64]],
    z_B_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Full combined energy+force FCHL18 kernel (asymmetric).

    Block layout::

        [0:N_A,  0:N_B ]  scalar   K[a,b]
        [0:N_A,  N_B:  ]  jac_t    dK/dR_B
        [N_A:,   0:N_B ]  jac      dK/dR_A
        [N_A:,   N_B:  ]  hessian  d²K/dR_A dR_B

    Returns
    -------
    ndarray, shape (N_A+D_A_total, N_B+D_B_total), float64
    """
    ...

def kernel_gaussian_full_symm(
    coords_list: Sequence[NDArray[np.float64]],
    z_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Full combined energy+force FCHL18 kernel (symmetric training set).

    Returns
    -------
    ndarray, shape (N+D, N+D), float64
    """
    ...

def kernel_gaussian_full_symm_rfp(
    coords_list: Sequence[NDArray[np.float64]],
    z_list: Sequence[NDArray[np.int32]],
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
) -> NDArray[np.float64]:
    """Full combined energy+force FCHL18 kernel (symmetric, RFP format).

    Returns
    -------
    ndarray, shape (BIG*(BIG+1)//2,), float64
        BIG = N + D.
    """
    ...
