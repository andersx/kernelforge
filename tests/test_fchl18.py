"""Tests for FCHL18 representation and kernel.

The reference implementations are direct Python/NumPy ports of the Fortran
routines in old_code/ffchl_module.f90 and old_code/ffchl_scalar_kernels.f90.
They are intentionally simple and not performance-optimised; the C++ code must
agree with them to numerical precision.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

import kernelforge.fchl18_repr as repr_mod
import kernelforge.fchl18_kernel as kernel_mod


# =============================================================================
# Reference (pure-Python) FCHL18 implementation
# =============================================================================


def _cut_function(r: float, cut_start: float, cut_distance: float) -> float:
    ru = cut_distance
    rl = cut_start * cut_distance
    if r >= ru:
        return 0.0
    if r <= rl:
        return 1.0
    x = (ru - r) / (ru - rl)
    return 10.0 * x**3 - 15.0 * x**4 + 6.0 * x**5


def _get_angular_norm2(t_width: float) -> float:
    pi = np.pi
    limit = 10000
    val = 0.0
    for n in range(-limit, limit + 1):
        val += np.exp(-((t_width * n) ** 2)) * (2.0 - 2.0 * np.cos(n * pi))
    return np.sqrt(val * pi) * 2.0


def _calc_cos_angle(a, b, c):
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-14 or n2 < 1e-14:
        return 0.0
    return float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))


def _calc_ksi3(atom_data, max_size, j, k, power, cut_start, cut_distance):
    """Axilrod-Teller-Muto three-body weight.
    atom_data: (5, max_size) — channel 0=distances, 1=Z, 2/3/4=xyz.
    """
    centre = atom_data[2:5, 0]
    vj = atom_data[2:5, j]
    vk = atom_data[2:5, k]

    cos_i = _calc_cos_angle(vk, centre, vj)
    cos_j = _calc_cos_angle(vj, vk, centre)
    cos_k = _calc_cos_angle(centre, vj, vk)

    dk = atom_data[0, j]
    dj = atom_data[0, k]
    di = np.linalg.norm(vj - vk)

    if di < 1e-14 or dj < 1e-14 or dk < 1e-14:
        return 0.0

    cut = (
        _cut_function(dk, cut_start, cut_distance)
        * _cut_function(dj, cut_start, cut_distance)
        * _cut_function(di, cut_start, cut_distance)
    )

    return cut * (1.0 + 3.0 * cos_i * cos_j * cos_k) / (di * dj * dk) ** power


def _compute_ksi(atom_data, max_size, n_neigh, power, cut_start, cut_distance):
    ksi = np.zeros(n_neigh)
    for k in range(1, n_neigh):
        r = atom_data[0, k]
        if r < 1e-14:
            continue
        ksi[k] = _cut_function(r, cut_start, cut_distance) / r**power
    return ksi


def _compute_threebody_fourier(
    atom_data, max_size, n_neigh, three_body_power, cut_start, cut_distance, order, pmax
):
    pi = np.pi
    cosp = np.zeros((pmax, order, max_size))
    sinp = np.zeros((pmax, order, max_size))

    for j in range(1, n_neigh):
        for k in range(j + 1, n_neigh):
            ksi3 = _calc_ksi3(atom_data, max_size, j, k, three_body_power, cut_start, cut_distance)
            if ksi3 == 0.0:
                continue

            centre = atom_data[2:5, 0]
            vj = atom_data[2:5, j]
            vk = atom_data[2:5, k]
            cos_a = _calc_cos_angle(vj, centre, vk)
            theta = np.arccos(np.clip(cos_a, -1.0, 1.0))

            pj = int(atom_data[1, k]) - 1  # Z of k-neighbour (0-indexed)
            pk = int(atom_data[1, j]) - 1  # Z of j-neighbour (0-indexed)
            if pj < 0 or pj >= pmax or pk < 0 or pk >= pmax:
                continue

            for m_idx in range(order):
                mf = float(m_idx + 1)
                cos_m = (np.cos(mf * theta) - np.cos((theta + pi) * mf)) * ksi3
                sin_m = (np.sin(mf * theta) - np.sin((theta + pi) * mf)) * ksi3

                cosp[pj, m_idx, j] += cos_m
                sinp[pj, m_idx, j] += sin_m
                cosp[pk, m_idx, k] += cos_m
                sinp[pk, m_idx, k] += sin_m

    return cosp, sinp


def _scalar_ref(
    x1,
    max_size1,
    n1_neigh,
    ksi1,
    cosp1,
    sinp1,
    x2,
    max_size2,
    n2_neigh,
    ksi2,
    cosp2,
    sinp2,
    t_width,
    d_width,
    order,
    pmax,
    ang_norm2,
    distance_scale,
    angular_scale,
):
    """Reference scalar product (no alchemy)."""
    Z1 = int(x1[1, 0])
    Z2 = int(x2[1, 0])
    if Z1 != Z2:
        return 0.0

    pi = np.pi
    g1 = np.sqrt(2.0 * pi) / ang_norm2
    s = [g1 * np.exp(-((t_width * (m + 1)) ** 2) / 2.0) for m in range(order)]
    inv_width = -1.0 / (4.0 * d_width**2)
    maxgausdist2 = (8.0 * d_width) ** 2

    aadist = 1.0
    for i in range(1, n1_neigh):
        Zi = int(x1[1, i])
        ri = x1[0, i]
        for j in range(1, n2_neigh):
            Zj = int(x2[1, j])
            if Zi != Zj:
                continue
            rj = x2[0, j]
            r2 = (rj - ri) ** 2
            if r2 >= maxgausdist2:
                continue
            d = np.exp(r2 * inv_width)

            angular = 0.0
            for m in range(order):
                ang_m = 0.0
                for p in range(pmax):
                    ang_m += cosp1[p, m, i] * cosp2[p, m, j] + sinp1[p, m, i] * sinp2[p, m, j]
                angular += ang_m * s[m]

            aadist += d * (ksi1[i] * ksi2[j] * distance_scale + angular * angular_scale)
    return aadist


def _ref_kernel_gaussian(
    x1_batch,
    n1_arr,
    nn1_arr,
    x2_batch,
    n2_arr,
    nn2_arr,
    sigma,
    two_body_scaling,
    two_body_width,
    two_body_power,
    three_body_scaling,
    three_body_width,
    three_body_power,
    cut_start,
    cut_distance,
    fourier_order,
):
    """Pure-Python reference FCHL18 Gaussian kernel."""
    nm1, max_size1 = n1_arr.shape[0], x1_batch.shape[1]
    nm2, max_size2 = n2_arr.shape[0], x2_batch.shape[1]

    true_distance_scale = two_body_scaling / 16.0
    true_angular_scale = three_body_scaling / np.sqrt(8.0)
    ang_norm2 = _get_angular_norm2(three_body_width)

    # Determine pmax
    pmax = 0
    for a in range(nm1):
        for i in range(n1_arr[a]):
            for k in range(nn1_arr[a, i]):
                z = int(x1_batch[a, i, 1, k])
                if z > pmax:
                    pmax = z
    for b in range(nm2):
        for j in range(n2_arr[b]):
            for k in range(nn2_arr[b, j]):
                z = int(x2_batch[b, j, 1, k])
                if z > pmax:
                    pmax = z

    inv_sigma2 = -1.0 / (sigma**2)
    K = np.zeros((nm1, nm2))

    # Precompute per-atom data
    def _precompute(x_batch, n_arr, nn_arr, nm, max_size):
        ksi_list = [[None] * max_size for _ in range(nm)]
        cosp_list = [[None] * max_size for _ in range(nm)]
        sinp_list = [[None] * max_size for _ in range(nm)]
        for a in range(nm):
            for i in range(n_arr[a]):
                atom = x_batch[a, i, :, :]  # (5, max_size)
                nn = nn_arr[a, i]
                ksi_list[a][i] = _compute_ksi(
                    atom, max_size, nn, two_body_power, cut_start, cut_distance
                )
                cosp_list[a][i], sinp_list[a][i] = _compute_threebody_fourier(
                    atom,
                    max_size,
                    nn,
                    three_body_power,
                    cut_start,
                    cut_distance,
                    fourier_order,
                    pmax,
                )
        return ksi_list, cosp_list, sinp_list

    ksi1_l, cosp1_l, sinp1_l = _precompute(x1_batch, n1_arr, nn1_arr, nm1, max_size1)
    ksi2_l, cosp2_l, sinp2_l = _precompute(x2_batch, n2_arr, nn2_arr, nm2, max_size2)

    # Self-scalars
    ss1 = np.zeros((nm1, max_size1))
    ss2 = np.zeros((nm2, max_size2))
    for a in range(nm1):
        for i in range(n1_arr[a]):
            ss1[a, i] = _scalar_ref(
                x1_batch[a, i],
                max_size1,
                nn1_arr[a, i],
                ksi1_l[a][i],
                cosp1_l[a][i],
                sinp1_l[a][i],
                x1_batch[a, i],
                max_size1,
                nn1_arr[a, i],
                ksi1_l[a][i],
                cosp1_l[a][i],
                sinp1_l[a][i],
                three_body_width,
                two_body_width,
                fourier_order,
                pmax,
                ang_norm2,
                true_distance_scale,
                true_angular_scale,
            )
    for b in range(nm2):
        for j in range(n2_arr[b]):
            ss2[b, j] = _scalar_ref(
                x2_batch[b, j],
                max_size2,
                nn2_arr[b, j],
                ksi2_l[b][j],
                cosp2_l[b][j],
                sinp2_l[b][j],
                x2_batch[b, j],
                max_size2,
                nn2_arr[b, j],
                ksi2_l[b][j],
                cosp2_l[b][j],
                sinp2_l[b][j],
                three_body_width,
                two_body_width,
                fourier_order,
                pmax,
                ang_norm2,
                true_distance_scale,
                true_angular_scale,
            )

    # Main kernel loop
    for a in range(nm1):
        for b in range(nm2):
            kab = 0.0
            for i in range(n1_arr[a]):
                Zi = int(x1_batch[a, i, 1, 0])
                for j in range(n2_arr[b]):
                    Zj = int(x2_batch[b, j, 1, 0])
                    if Zi != Zj:
                        continue
                    s12 = _scalar_ref(
                        x1_batch[a, i],
                        max_size1,
                        nn1_arr[a, i],
                        ksi1_l[a][i],
                        cosp1_l[a][i],
                        sinp1_l[a][i],
                        x2_batch[b, j],
                        max_size2,
                        nn2_arr[b, j],
                        ksi2_l[b][j],
                        cosp2_l[b][j],
                        sinp2_l[b][j],
                        three_body_width,
                        two_body_width,
                        fourier_order,
                        pmax,
                        ang_norm2,
                        true_distance_scale,
                        true_angular_scale,
                    )
                    kab += np.exp((ss1[a, i] + ss2[b, j] - 2.0 * s12) * inv_sigma2)
            K[a, b] = kab
    return K


# =============================================================================
# Fixture: small set of toy molecules (H2O-like, CH4-like, HCN-like)
# =============================================================================


def _water_like(seed=0):
    rng = np.random.default_rng(seed)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # O
            [0.96, 0.0, 0.0],  # H
            [-0.24, 0.93, 0.0],  # H
        ],
        dtype=np.float64,
    )
    coords += rng.standard_normal((3, 3)) * 0.05
    Z = np.array([8, 1, 1], dtype=np.int32)
    return coords, Z


def _methane_like(seed=0):
    rng = np.random.default_rng(seed)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
        dtype=np.float64,
    )
    coords += rng.standard_normal((5, 3)) * 0.05
    Z = np.array([6, 1, 1, 1, 1], dtype=np.int32)
    return coords, Z


def _hcn_like(seed=0):
    rng = np.random.default_rng(seed)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.06, 0.0, 0.0],
            [2.19, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    coords += rng.standard_normal((3, 3)) * 0.05
    Z = np.array([1, 6, 7], dtype=np.int32)
    return coords, Z


def _make_batch(molecules, max_size=10, cut_distance=8.0):
    coords_list = [c for c, _ in molecules]
    z_list = [z for _, z in molecules]
    x, n, nn = repr_mod.generate(coords_list, z_list, max_size=max_size, cut_distance=cut_distance)
    return x, n, nn


# =============================================================================
# Representation tests
# =============================================================================


def test_repr_output_shapes():
    coords, Z = _water_like()
    x, n, nn = repr_mod.generate([coords], [Z], max_size=10, cut_distance=8.0)
    assert x.shape == (1, 10, 5, 10)
    assert n.shape == (1,)
    assert nn.shape == (1, 10)
    assert n[0] == 3


def test_repr_n_atoms_correct():
    mols = [_water_like(), _methane_like(), _hcn_like()]
    coords_list = [c for c, _ in mols]
    z_list = [z for _, z in mols]
    _, n, _ = repr_mod.generate(coords_list, z_list, max_size=10, cut_distance=8.0)
    assert list(n) == [3, 5, 3]


def test_repr_padding_value():
    """Slots beyond n_atoms must still have the 1e100 padding in channel 0."""
    coords, Z = _water_like()
    x, n, nn = repr_mod.generate([coords], [Z], max_size=10, cut_distance=8.0)
    na = int(n[0])
    # Atoms beyond na should have distance 1e100 for all neighbour slots
    for i in range(na, 10):
        assert x[0, i, 0, 0] == pytest.approx(1e100)


def test_repr_self_at_index_0():
    """For every real atom, the first neighbour (index 0) is the atom itself:
    distance=0, same Z, displacement=(0,0,0)."""
    coords, Z = _water_like()
    x, n, nn = repr_mod.generate([coords], [Z], max_size=10, cut_distance=8.0)
    na = int(n[0])
    for i in range(na):
        assert x[0, i, 0, 0] == pytest.approx(0.0, abs=1e-12)  # self-distance
        assert int(x[0, i, 1, 0]) == int(Z[i])  # self-Z
        assert x[0, i, 2, 0] == pytest.approx(0.0, abs=1e-12)  # dx
        assert x[0, i, 3, 0] == pytest.approx(0.0, abs=1e-12)  # dy
        assert x[0, i, 4, 0] == pytest.approx(0.0, abs=1e-12)  # dz


def test_repr_distances_sorted():
    """Neighbours must be sorted by increasing distance for each atom."""
    coords, Z = _methane_like()
    x, n, nn = repr_mod.generate([coords], [Z], max_size=10, cut_distance=8.0)
    na = int(n[0])
    for i in range(na):
        nn_i = int(nn[0, i])
        dists = x[0, i, 0, :nn_i]
        assert np.all(dists[:-1] <= dists[1:]), f"distances not sorted for atom {i}"


def test_repr_translation_invariance():
    """Shifting all atoms by the same vector must not change the representation."""
    coords, Z = _water_like(seed=7)
    shift = np.array([3.14, -2.72, 1.41])

    x1, n1, nn1 = repr_mod.generate([coords], [Z], max_size=10, cut_distance=6.0)
    x2, n2, nn2 = repr_mod.generate([coords + shift], [Z], max_size=10, cut_distance=6.0)

    # Distances (channel 0) must be identical
    np.testing.assert_allclose(x1[0, :, 0, :], x2[0, :, 0, :], atol=1e-10)
    # Nuclear charges (channel 1) identical
    np.testing.assert_allclose(x1[0, :, 1, :], x2[0, :, 1, :], atol=1e-10)
    # Displacements may differ by the shift but the norms (== distances) are the same
    for i in range(int(n1[0])):
        for k in range(int(nn1[0, i])):
            d1 = np.linalg.norm(x1[0, i, 2:5, k])
            d2 = np.linalg.norm(x2[0, i, 2:5, k])
            assert abs(d1 - d2) < 1e-10


def test_repr_nn_consistency():
    """n_neighbors[a, i] == number of non-1e100 entries in channel 0 for that atom."""
    mols = [_water_like(), _methane_like()]
    coords_list = [c for c, _ in mols]
    z_list = [z for _, z in mols]
    x, n, nn = repr_mod.generate(coords_list, z_list, max_size=10, cut_distance=8.0)
    for a in range(len(mols)):
        for i in range(int(n[a])):
            nn_i = int(nn[a, i])
            # slots 0..nn_i-1 must have finite distances, rest 1e100
            assert np.all(x[a, i, 0, :nn_i] < 1e99), (
                f"mol {a} atom {i}: expected finite dist in first {nn_i} slots"
            )
            assert np.all(x[a, i, 0, nn_i:] > 1e99), (
                f"mol {a} atom {i}: expected 1e100 padding after slot {nn_i}"
            )


# =============================================================================
# Kernel tests
# =============================================================================

# Default hyperparameters matching the test in old_code/test_fchl_scalar.py
KERNEL_ARGS = dict(
    two_body_width=0.1,
    two_body_scaling=2.0,
    two_body_power=6.0,
    three_body_width=3.0,
    three_body_scaling=2.0,
    three_body_power=3.0,
    cut_start=0.5,
    cut_distance=8.0,
    fourier_order=2,
)


def _small_batch(n_mols=4, seed=0):
    rng = np.random.default_rng(seed)
    mol_fns = [_water_like, _methane_like, _hcn_like]
    mols = [mol_fns[rng.integers(len(mol_fns))](seed=seed + i) for i in range(n_mols)]
    x, n, nn = _make_batch(mols, max_size=10, cut_distance=KERNEL_ARGS["cut_distance"])
    return x, n, nn


def test_kernel_gaussian_shape():
    x1, n1, nn1 = _small_batch(3, seed=0)
    x2, n2, nn2 = _small_batch(4, seed=1)
    K = kernel_mod.kernel_gaussian(x1, x2, n1, n2, nn1, nn2, sigma=2.5, **KERNEL_ARGS)
    assert K.shape == (3, 4)


def test_kernel_gaussian_symm_shape():
    x, n, nn = _small_batch(5, seed=2)
    K = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    assert K.shape == (5, 5)


def test_kernel_gaussian_nonneg():
    """Gaussian kernel values must be non-negative."""
    x1, n1, nn1 = _small_batch(3, seed=3)
    x2, n2, nn2 = _small_batch(3, seed=4)
    K = kernel_mod.kernel_gaussian(x1, x2, n1, n2, nn1, nn2, sigma=2.5, **KERNEL_ARGS)
    assert np.all(K >= 0.0)


def test_kernel_gaussian_symm_is_symmetric():
    """Symmetric kernel must be exactly symmetric."""
    x, n, nn = _small_batch(6, seed=5)
    K = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    np.testing.assert_allclose(K, K.T, atol=1e-13)


def test_kernel_gaussian_symm_matches_asym_self():
    """kernel_gaussian_symm(X) must equal kernel_gaussian(X, X)."""
    x, n, nn = _small_batch(4, seed=6)
    K_sym = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    K_asym = kernel_mod.kernel_gaussian(x, x, n, n, nn, nn, sigma=2.5, **KERNEL_ARGS)
    np.testing.assert_allclose(K_sym, K_asym, rtol=1e-12, atol=1e-13)


def test_kernel_gaussian_psd():
    """Symmetric kernel matrix should be positive semidefinite."""
    x, n, nn = _small_batch(5, seed=7)
    K = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-8), f"Min eigenvalue {eigvals.min():.3e} < -1e-8"


def test_kernel_gaussian_diagonal_equals_self_kernel():
    """K_sym[a,a] must equal K_asym(mol_a, mol_a)."""
    x, n, nn = _small_batch(4, seed=8)
    K_sym = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    for a in range(int(x.shape[0])):
        xa = x[a : a + 1]
        na = n[a : a + 1]
        nna = nn[a : a + 1]
        K_self = kernel_mod.kernel_gaussian(xa, xa, na, na, nna, nna, sigma=2.5, **KERNEL_ARGS)
        assert K_sym[a, a] == pytest.approx(K_self[0, 0], rel=1e-10)


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_kernel_gaussian_matches_reference(seed):
    """C++ kernel must agree with the pure-Python reference implementation."""
    x, n, nn = _small_batch(3, seed=seed)

    # Use a smaller fourier_order and cut_distance for the reference (speed)
    kargs = dict(KERNEL_ARGS)
    kargs["fourier_order"] = 1  # reference is slow; reduce order

    K_cpp = kernel_mod.kernel_gaussian(x, x, n, n, nn, nn, sigma=2.5, **kargs)
    K_ref = _ref_kernel_gaussian(
        x,
        n,
        nn,
        x,
        n,
        nn,
        sigma=2.5,
        **kargs,
    )
    np.testing.assert_allclose(
        K_cpp, K_ref, rtol=1e-8, atol=1e-10, err_msg=f"C++ vs reference mismatch (seed={seed})"
    )


def test_kernel_gaussian_zero_if_no_shared_species():
    """K must be zero between molecules with entirely disjoint element sets."""
    # Molecule of all H
    coords_h = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    Z_h = np.array([1, 1], dtype=np.int32)
    # Molecule of all O
    coords_o = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    Z_o = np.array([8, 8], dtype=np.int32)

    x1, n1, nn1 = repr_mod.generate([coords_h], [Z_h], max_size=5, cut_distance=8.0)
    x2, n2, nn2 = repr_mod.generate([coords_o], [Z_o], max_size=5, cut_distance=8.0)

    # No shared elements between H and O molecules -> K must be 0
    K = kernel_mod.kernel_gaussian(x1, x2, n1, n2, nn1, nn2, sigma=2.5, **KERNEL_ARGS)
    assert K.shape == (1, 1)
    assert K[0, 0] == pytest.approx(0.0, abs=1e-14)


def test_kernel_gaussian_same_molecule_large_value():
    """Two identical molecules should give a larger kernel than two different ones
    where no shared elements exist."""
    # Molecule A: only O atoms
    coords_o = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    Z_o = np.array([8, 8], dtype=np.int32)
    # Molecule B: only N atoms
    coords_n = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    Z_n = np.array([7, 7], dtype=np.int32)

    xA, nA, nnA = repr_mod.generate([coords_o], [Z_o], max_size=5, cut_distance=8.0)
    xB, nB, nnB = repr_mod.generate([coords_n], [Z_n], max_size=5, cut_distance=8.0)

    # K(A,A) > 0 (same molecule, identical representations)
    KAA = kernel_mod.kernel_gaussian(xA, xA, nA, nA, nnA, nnA, sigma=2.5, **KERNEL_ARGS)[0, 0]
    # K(A,B) = 0 because O and N are disjoint element sets
    KAB = kernel_mod.kernel_gaussian(xA, xB, nA, nB, nnA, nnB, sigma=2.5, **KERNEL_ARGS)[0, 0]

    assert KAA > 0.0
    assert KAB == pytest.approx(0.0, abs=1e-14)
    assert KAA > KAB


def test_repr_then_kernel_finite():
    """End-to-end: representation + kernel must produce finite values."""
    mols = [_water_like(s) for s in range(5)] + [_methane_like(s) for s in range(3)]
    coords_list = [c for c, _ in mols]
    z_list = [z for _, z in mols]
    x, n, nn = repr_mod.generate(
        coords_list, z_list, max_size=10, cut_distance=KERNEL_ARGS["cut_distance"]
    )
    K = kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=2.5, **KERNEL_ARGS)
    assert np.all(np.isfinite(K)), "Kernel contains non-finite values"


def test_kernel_invalid_sigma():
    x, n, nn = _small_batch(2, seed=0)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian(x, x, n, n, nn, nn, sigma=0.0, **KERNEL_ARGS)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian(x, x, n, n, nn, nn, sigma=-1.0, **KERNEL_ARGS)
    with pytest.raises(Exception):
        kernel_mod.kernel_gaussian_symm(x, n, nn, sigma=0.0, **KERNEL_ARGS)
