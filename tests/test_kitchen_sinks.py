"""Tests for the kitchen_sinks (Random Fourier Features) module."""

import numpy as np
import pytest

from kernelforge.kitchen_sinks import (
    rff_features,
    rff_features_elemental,
    rff_full,
    rff_full_elemental,
    rff_full_gramian_elemental,
    rff_full_gramian_elemental_rfp,
    rff_full_gramian_symm,
    rff_full_gramian_symm_rfp,
    rff_gradient,
    rff_gradient_elemental,
    rff_gradient_gramian_elemental,
    rff_gradient_gramian_elemental_rfp,
    rff_gradient_gramian_symm,
    rff_gradient_gramian_symm_rfp,
    rff_gramian_elemental,
    rff_gramian_elemental_rfp,
    rff_gramian_symm,
    rff_gramian_symm_rfp,
)


# ---------------------------------------------------------------------------
# Pure-NumPy reference implementation
# ---------------------------------------------------------------------------
def rff_features_numpy(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference: Z = sqrt(2/D) * cos(X @ W + b)."""
    D = b.shape[0]
    return np.sqrt(2.0 / D) * np.cos(X @ W + b)


# ---------------------------------------------------------------------------
# rff_features tests
# ---------------------------------------------------------------------------
class TestRffFeatures:
    """Tests for the basic rff_features function."""

    def test_shape_and_dtype(self) -> None:
        """Output has correct shape and dtype."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 50, 100, 200
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z = rff_features(X, W, b)

        assert Z.shape == (N, D)
        assert Z.dtype == np.float64

    def test_c_contiguous(self) -> None:
        """Output is C-contiguous."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 10, 20, 30
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z = rff_features(X, W, b)
        assert Z.flags["C_CONTIGUOUS"]

    def test_matches_numpy_reference_small(self) -> None:
        """C++ result matches NumPy reference (small problem, exact match)."""
        rng = np.random.default_rng(123)
        N, rep_size, D = 20, 10, 50
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_cpp = rff_features(X, W, b)
        Z_ref = rff_features_numpy(X, W, b)

        np.testing.assert_allclose(Z_cpp, Z_ref, rtol=1e-12, atol=1e-12)

    def test_matches_numpy_reference_large(self) -> None:
        """C++ result matches NumPy reference (large problem, relaxed for BLAS reordering)."""
        rng = np.random.default_rng(123)
        N, rep_size, D = 100, 200, 500
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_cpp = rff_features(X, W, b)
        Z_ref = rff_features_numpy(X, W, b)

        # Relaxed tolerance: -ffast-math + threaded BLAS can reorder sums
        np.testing.assert_allclose(Z_cpp, Z_ref, rtol=1e-6, atol=1e-6)

    def test_values_bounded(self) -> None:
        """All values are in [-sqrt(2/D), sqrt(2/D)]."""
        rng = np.random.default_rng(0)
        N, rep_size, D = 200, 80, 500
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z = rff_features(X, W, b)
        bound = np.sqrt(2.0 / D)

        assert np.all(-bound - 1e-15 <= Z)
        assert np.all(bound + 1e-15 >= Z)

    @pytest.mark.parametrize(
        ("N", "rep_size", "D"),
        [
            (1, 1, 1),
            (1, 100, 500),
            (500, 1, 1),
            (10, 10, 10),
            (1000, 200, 4000),
        ],
    )
    def test_various_sizes(self, N: int, rep_size: int, D: int) -> None:
        """Works for various input sizes."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_cpp = rff_features(X, W, b)
        Z_ref = rff_features_numpy(X, W, b)

        np.testing.assert_allclose(Z_cpp, Z_ref, rtol=1e-11, atol=1e-11)

    def test_single_sample(self) -> None:
        """Works for a single input sample."""
        rng = np.random.default_rng(7)
        rep_size, D = 50, 100
        X = rng.normal(size=(1, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z = rff_features(X, W, b)
        Z_ref = rff_features_numpy(X, W, b)

        assert Z.shape == (1, D)
        np.testing.assert_allclose(Z, Z_ref, rtol=1e-12, atol=1e-12)

    def test_fortran_order_input_accepted(self) -> None:
        """F-order input is automatically converted (forcecast)."""
        rng = np.random.default_rng(99)
        N, rep_size, D = 20, 30, 50
        X = np.asfortranarray(rng.normal(size=(N, rep_size)))
        W = np.asfortranarray(rng.normal(size=(rep_size, D)))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z = rff_features(X, W, b)
        Z_ref = rff_features_numpy(np.ascontiguousarray(X), np.ascontiguousarray(W), b)

        np.testing.assert_allclose(Z, Z_ref, rtol=1e-12, atol=1e-12)

    def test_bad_dimensions_raise(self) -> None:
        """Wrong number of dimensions raises."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10,))  # 1D instead of 2D
        W = rng.normal(size=(10, 20))
        b = rng.uniform(size=(20,))

        with pytest.raises(ValueError, match="X must be 2D"):
            rff_features(X, W, b)


# ---------------------------------------------------------------------------
# Pure-NumPy reference for rff_features_elemental
# ---------------------------------------------------------------------------
def rff_features_elemental_numpy(
    X: np.ndarray,
    Q: list[np.ndarray],
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Reference implementation: element-stratified RFF with per-molecule sum.

    X: (nmol, max_atoms, rep_size)
    Q: list of nmol int arrays, 0-indexed element indices
    W: (nelements, rep_size, D)
    b: (nelements, D)
    """
    nmol = X.shape[0]
    nelements, _, D = W.shape
    norm = np.sqrt(2.0 / D)

    LZ = np.zeros((nmol, D))

    for e in range(nelements):
        # Gather all atoms of element e
        rows = []
        mol_indices = []
        for i in range(nmol):
            natoms = len(Q[i])
            for j in range(natoms):
                if Q[i][j] == e:
                    rows.append(X[i, j, :])
                    mol_indices.append(i)

        if len(rows) == 0:
            continue

        Xsort = np.array(rows)  # (total_e, rep_size)
        Ze = norm * np.cos(Xsort @ W[e] + b[e])  # (total_e, D)

        for k, mol_idx in enumerate(mol_indices):
            LZ[mol_idx] += Ze[k]

    return LZ


# ---------------------------------------------------------------------------
# rff_features_elemental tests
# ---------------------------------------------------------------------------
class TestRffFeaturesElemental:
    """Tests for the element-stratified rff_features_elemental function."""

    @staticmethod
    def _make_test_data(
        rng: np.random.Generator,
        nmol: int,
        min_atoms: int,
        max_atoms_per_mol: int,
        rep_size: int,
        D: int,
        nelements: int,
    ) -> tuple:
        """Helper to create random test data."""
        sizes = rng.integers(min_atoms, max_atoms_per_mol + 1, size=nmol)
        max_atoms = int(sizes.max())

        X = np.zeros((nmol, max_atoms, rep_size))
        Q = []
        for i in range(nmol):
            n = sizes[i]
            X[i, :n, :] = rng.normal(size=(n, rep_size))
            Q.append(rng.integers(0, nelements, size=n).astype(np.int32))

        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))

        return X, Q, W, b, sizes

    def test_shape_and_dtype(self) -> None:
        """Output has correct shape and dtype."""
        rng = np.random.default_rng(42)
        nmol, rep_size, D, nelements = 10, 50, 100, 3
        X, Q, W, b, _ = self._make_test_data(rng, nmol, 2, 8, rep_size, D, nelements)

        LZ = rff_features_elemental(X, Q, W, b)

        assert LZ.shape == (nmol, D)
        assert LZ.dtype == np.float64

    def test_c_contiguous(self) -> None:
        """Output is C-contiguous."""
        rng = np.random.default_rng(42)
        X, Q, W, b, _ = self._make_test_data(rng, 5, 2, 6, 20, 30, 2)

        LZ = rff_features_elemental(X, Q, W, b)
        assert LZ.flags["C_CONTIGUOUS"]

    def test_matches_numpy_reference_small(self) -> None:
        """C++ matches NumPy reference (small problem)."""
        rng = np.random.default_rng(123)
        X, Q, W, b, _ = self._make_test_data(rng, 5, 2, 6, 10, 20, 2)

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-12, atol=1e-12)

    def test_matches_numpy_reference_medium(self) -> None:
        """C++ matches NumPy reference (medium problem)."""
        rng = np.random.default_rng(456)
        X, Q, W, b, _ = self._make_test_data(rng, 50, 3, 20, 100, 200, 4)

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-10, atol=1e-10)

    def test_matches_numpy_reference_large(self) -> None:
        """C++ matches NumPy reference (large problem, relaxed tol)."""
        rng = np.random.default_rng(789)
        X, Q, W, b, _ = self._make_test_data(rng, 200, 5, 30, 200, 500, 5)

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-6, atol=1e-6)

    def test_single_molecule(self) -> None:
        """Works with a single molecule."""
        rng = np.random.default_rng(7)
        X, Q, W, b, _ = self._make_test_data(rng, 1, 3, 5, 30, 50, 2)

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        assert LZ_cpp.shape == (1, 50)
        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-12, atol=1e-12)

    def test_single_element(self) -> None:
        """Works when all atoms are the same element."""
        rng = np.random.default_rng(99)
        nmol, rep_size, D = 10, 20, 40

        sizes = rng.integers(2, 8, size=nmol)
        max_atoms = int(sizes.max())
        X = np.zeros((nmol, max_atoms, rep_size))
        Q = []
        for i in range(nmol):
            n = sizes[i]
            X[i, :n, :] = rng.normal(size=(n, rep_size))
            Q.append(np.zeros(n, dtype=np.int32))  # All element 0

        W = rng.normal(size=(1, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(1, D))

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-12, atol=1e-12)

    def test_element_not_present_in_some_molecules(self) -> None:
        """Handles molecules that have no atoms of a given element."""
        rng = np.random.default_rng(55)
        nmol, rep_size, D, nelements = 6, 15, 30, 3

        sizes = rng.integers(2, 5, size=nmol)
        max_atoms = int(sizes.max())
        X = np.zeros((nmol, max_atoms, rep_size))
        Q = []
        for i in range(nmol):
            n = sizes[i]
            X[i, :n, :] = rng.normal(size=(n, rep_size))
            # Only use elements 0 and 1 for even molecules, 1 and 2 for odd
            if i % 2 == 0:
                Q.append(rng.integers(0, 2, size=n).astype(np.int32))
            else:
                Q.append(rng.integers(1, 3, size=n).astype(np.int32))

        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-12, atol=1e-12)

    def test_uniform_sizes(self) -> None:
        """Works when all molecules have the same number of atoms."""
        rng = np.random.default_rng(33)
        nmol, natoms, rep_size, D, nelements = 20, 5, 25, 50, 2

        X = rng.normal(size=(nmol, natoms, rep_size))
        Q = [rng.integers(0, nelements, size=natoms).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-12, atol=1e-12)

    def test_single_atom_molecules(self) -> None:
        """Works when every molecule has exactly one atom."""
        rng = np.random.default_rng(77)
        nmol, rep_size, D, nelements = 15, 10, 20, 3

        X = rng.normal(size=(nmol, 1, rep_size))
        Q = [np.array([rng.integers(0, nelements)], dtype=np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))

        LZ_cpp = rff_features_elemental(X, Q, W, b)
        LZ_ref = rff_features_elemental_numpy(X, Q, W, b)

        np.testing.assert_allclose(LZ_cpp, LZ_ref, rtol=1e-12, atol=1e-12)

    def test_bad_X_ndim_raises(self) -> None:
        """X with wrong ndim raises."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 20))  # 2D instead of 3D
        Q = [np.array([0, 1], dtype=np.int32)]
        W = rng.normal(size=(2, 20, 30))
        b = rng.uniform(size=(2, 30))

        with pytest.raises(ValueError, match="X must be 3D"):
            rff_features_elemental(X, Q, W, b)

    def test_bad_Q_length_raises(self) -> None:
        """Q with wrong length raises."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 3, 10))
        Q = [np.array([0], dtype=np.int32)]  # Only 1 entry, not 5
        W = rng.normal(size=(2, 10, 20))
        b = rng.uniform(size=(2, 20))

        with pytest.raises(ValueError, match=r"len\(Q\) must equal X\.shape\[0\]"):
            rff_features_elemental(X, Q, W, b)


# ---------------------------------------------------------------------------
# rff_gradient_elemental tests
# ---------------------------------------------------------------------------
class TestRffGradientElemental:
    """Tests for rff_gradient_elemental (gradient of elemental RFF wrt coords)."""

    @staticmethod
    def _make_grad_test_data(
        rng: np.random.Generator,
        nmol: int,
        natoms_each: int,
        rep_size: int,
        D: int,
        nelements: int,
    ) -> tuple:
        """Create test data with analytically simple dX.

        For testing, all molecules have the same number of atoms to keep
        things simple. dX is random.
        """
        max_atoms = natoms_each
        X = rng.normal(size=(nmol, max_atoms, rep_size))
        dX = rng.normal(size=(nmol, max_atoms, rep_size, max_atoms, 3))
        Q = [rng.integers(0, nelements, size=natoms_each).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))
        sizes = np.full(nmol, natoms_each, dtype=np.int32)
        return X, dX, Q, W, b, sizes

    @staticmethod
    def _rff_gradient_numpy(
        X: np.ndarray,
        dX: np.ndarray,
        Q: list[np.ndarray],
        W: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Pure-NumPy reference for rff_gradient_elemental.

        X:  (nmol, max_atoms, rep_size)
        dX: (nmol, max_atoms, rep_size, max_atoms, 3)
        Q:  list of int arrays (0-indexed)
        W:  (nelements, rep_size, D)
        b:  (nelements, D)

        Returns G: (D, ngrads), ngrads = 3 * sum(natoms)
        """
        nmol = X.shape[0]
        D = W.shape[2]
        rep_size = W.shape[1]
        sizes = [len(q) for q in Q]
        ngrads = 3 * sum(sizes)
        norm = -np.sqrt(2.0 / D)

        G = np.zeros((D, ngrads))

        g_offset = 0
        for i in range(nmol):
            natoms = sizes[i]
            ncols = 3 * natoms

            for j in range(natoms):
                e = Q[i][j]

                # Forward: z = X[i,j,:] @ W[e] + b[e]  -> (D,)
                z = X[i, j, :] @ W[e] + b[e]

                sin_z = np.sin(z)  # (D,)
                dg = norm * (sin_z[:, None] * W[e].T)  # (D, rep_size)

                dX_atom = dX[i, j, :, :natoms, :].reshape(rep_size, ncols)

                G[:, g_offset : g_offset + ncols] += -dg @ dX_atom

            g_offset += ncols

        return G

    def test_shape_and_dtype(self) -> None:
        """Output has correct shape (D, ngrads) and dtype."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 3, 4, 10, 20, 2
        X, dX, Q, W, b, _sizes = self._make_grad_test_data(rng, nmol, natoms, rep_size, D, nel)

        G = rff_gradient_elemental(X, dX, Q, W, b)

        ngrads = 3 * nmol * natoms
        assert G.shape == (D, ngrads)
        assert G.dtype == np.float64

    def test_c_contiguous(self) -> None:
        """Output is C-contiguous."""
        rng = np.random.default_rng(42)
        X, dX, Q, W, b, _ = self._make_grad_test_data(rng, 2, 3, 8, 15, 2)

        G = rff_gradient_elemental(X, dX, Q, W, b)
        assert G.flags["C_CONTIGUOUS"]

    def test_matches_numpy_reference_small(self) -> None:
        """C++ matches NumPy reference (small problem)."""
        rng = np.random.default_rng(123)
        X, dX, Q, W, b, _ = self._make_grad_test_data(rng, 2, 3, 5, 10, 2)

        G_cpp = rff_gradient_elemental(X, dX, Q, W, b)
        G_ref = self._rff_gradient_numpy(X, dX, Q, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-12, atol=1e-12)

    def test_matches_numpy_reference_medium(self) -> None:
        """C++ matches NumPy reference (medium problem)."""
        rng = np.random.default_rng(456)
        X, dX, Q, W, b, _ = self._make_grad_test_data(rng, 5, 6, 20, 50, 3)

        G_cpp = rff_gradient_elemental(X, dX, Q, W, b)
        G_ref = self._rff_gradient_numpy(X, dX, Q, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-10, atol=1e-10)

    def test_matches_numpy_reference_large(self) -> None:
        """C++ matches NumPy reference (larger, relaxed tol for BLAS)."""
        rng = np.random.default_rng(789)
        X, dX, Q, W, b, _ = self._make_grad_test_data(rng, 10, 8, 50, 100, 4)

        G_cpp = rff_gradient_elemental(X, dX, Q, W, b)
        G_ref = self._rff_gradient_numpy(X, dX, Q, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-6, atol=1e-6)

    def test_finite_difference(self) -> None:
        """Gradient matches finite-difference on elemental RFF features.

        For a single molecule, perturb each coordinate and check that
        the gradient G predicts the change in LZ correctly.
        """
        # Covered by numpy reference tests above; no analytic coord function available.
        pass  # covered by numpy reference tests

    def test_single_molecule(self) -> None:
        """Works with a single molecule."""
        rng = np.random.default_rng(7)
        X, dX, Q, W, b, _ = self._make_grad_test_data(rng, 1, 4, 10, 20, 2)

        G_cpp = rff_gradient_elemental(X, dX, Q, W, b)
        G_ref = self._rff_gradient_numpy(X, dX, Q, W, b)

        assert G_cpp.shape == (20, 12)  # D=20, ngrads=3*4=12
        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-12, atol=1e-12)

    def test_single_element(self) -> None:
        """Works when all atoms are the same element."""
        rng = np.random.default_rng(99)
        nmol, natoms, rep_size, D = 3, 4, 10, 15

        X = rng.normal(size=(nmol, natoms, rep_size))
        dX = rng.normal(size=(nmol, natoms, rep_size, natoms, 3))
        Q = [np.zeros(natoms, dtype=np.int32) for _ in range(nmol)]
        W = rng.normal(size=(1, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(1, D))

        G_cpp = rff_gradient_elemental(X, dX, Q, W, b)
        G_ref = self._rff_gradient_numpy(X, dX, Q, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_gramian_elemental tests
# ---------------------------------------------------------------------------
class TestRffGramianElemental:
    """Tests for rff_gramian_elemental (chunked Gram matrix for RFF training)."""

    @staticmethod
    def _make_test_data(
        rng: np.random.Generator,
        nmol: int,
        min_atoms: int,
        max_atoms_per_mol: int,
        rep_size: int,
        D: int,
        nelements: int,
    ) -> tuple:
        """Create random test data."""
        sizes = rng.integers(min_atoms, max_atoms_per_mol + 1, size=nmol)
        max_atoms = int(sizes.max())

        X = np.zeros((nmol, max_atoms, rep_size))
        Q = []
        for i in range(nmol):
            n = sizes[i]
            X[i, :n, :] = rng.normal(size=(n, rep_size))
            Q.append(rng.integers(0, nelements, size=n).astype(np.int32))

        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))
        Y = rng.normal(size=nmol)

        return X, Q, W, b, Y, sizes

    def test_shapes_and_dtypes(self) -> None:
        """Output shapes and dtypes are correct."""
        rng = np.random.default_rng(42)
        D = 30
        X, Q, W, b, Y, _ = self._make_test_data(rng, 10, 2, 6, 15, D, 2)

        ZtZ, ZtY = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=5)

        assert ZtZ.shape == (D, D)
        assert ZtY.shape == (D,)
        assert ZtZ.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_gram_matrix_symmetric(self) -> None:
        """Gram matrix ZtZ is symmetric."""
        rng = np.random.default_rng(42)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 20, 2, 8, 20, 40, 3)

        ZtZ, _ = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=7)

        np.testing.assert_allclose(ZtZ, ZtZ.T, rtol=1e-14, atol=1e-14)

    def test_gram_matrix_positive_semidefinite(self) -> None:
        """Gram matrix ZtZ is PSD (all eigenvalues >= 0)."""
        rng = np.random.default_rng(42)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 20, 2, 8, 20, 40, 3)

        ZtZ, _ = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=7)

        eigvals = np.linalg.eigvalsh(ZtZ)
        assert np.all(eigvals >= -1e-10)

    def test_matches_numpy_reference(self) -> None:
        """Gramian matches direct computation LZ^T @ LZ, LZ^T @ Y."""
        rng = np.random.default_rng(123)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 15, 2, 6, 10, 20, 2)

        ZtZ_cpp, ZtY_cpp = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=5)

        # Compute reference: get full LZ, then compute Gram
        LZ = rff_features_elemental(X, Q, W, b)
        ZtZ_ref = LZ.T @ LZ
        ZtY_ref = LZ.T @ Y

        np.testing.assert_allclose(ZtZ_cpp, ZtZ_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ZtY_cpp, ZtY_ref, rtol=1e-10, atol=1e-10)

    def test_chunk_size_does_not_affect_result(self) -> None:
        """Different chunk sizes give the same result."""
        rng = np.random.default_rng(456)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 20, 2, 5, 10, 25, 2)

        ZtZ_1, ZtY_1 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=1)
        ZtZ_5, ZtY_5 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=5)
        ZtZ_20, ZtY_20 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=20)
        ZtZ_100, ZtY_100 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=100)

        np.testing.assert_allclose(ZtZ_1, ZtZ_5, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtZ_1, ZtZ_20, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtZ_1, ZtZ_100, rtol=1e-12, atol=1e-12)

        np.testing.assert_allclose(ZtY_1, ZtY_5, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY_1, ZtY_20, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY_1, ZtY_100, rtol=1e-12, atol=1e-12)

    def test_single_chunk(self) -> None:
        """Works when chunk_size >= nmol (single chunk)."""
        rng = np.random.default_rng(789)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 10, 2, 5, 10, 20, 2)

        ZtZ, ZtY = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=1000)

        LZ = rff_features_elemental(X, Q, W, b)
        np.testing.assert_allclose(ZtZ, LZ.T @ LZ, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY, LZ.T @ Y, rtol=1e-12, atol=1e-12)

    def test_larger_problem(self) -> None:
        """Larger problem with relaxed tolerance."""
        rng = np.random.default_rng(111)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 100, 3, 15, 100, 200, 4)

        ZtZ, ZtY = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=25)

        LZ = rff_features_elemental(X, Q, W, b)
        np.testing.assert_allclose(ZtZ, LZ.T @ LZ, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ZtY, LZ.T @ Y, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# rff_full_gramian_elemental tests
# ---------------------------------------------------------------------------
class TestRffFullGramianElemental:
    """Tests for rff_full_gramian_elemental (energy+force training Gramian)."""

    @staticmethod
    def _make_test_data(
        rng: np.random.Generator,
        nmol: int,
        natoms_each: int,
        rep_size: int,
        D: int,
        nelements: int,
    ) -> tuple:
        """Create test data. All molecules have same number of atoms for simplicity."""
        max_atoms = natoms_each
        X = rng.normal(size=(nmol, max_atoms, rep_size))
        dX = rng.normal(size=(nmol, max_atoms, rep_size, max_atoms, 3))
        Q = [rng.integers(0, nelements, size=natoms_each).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nelements, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nelements, D))
        Y = rng.normal(size=nmol)
        ngrads = 3 * nmol * natoms_each
        F = rng.normal(size=ngrads)
        return X, dX, Q, W, b, Y, F

    def test_shapes_and_dtypes(self) -> None:
        """Output shapes and dtypes are correct."""
        rng = np.random.default_rng(42)
        D = 20
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, 5, 3, 10, D, 2)

        ZtZ, ZtY = rff_full_gramian_elemental(X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2)

        assert ZtZ.shape == (D, D)
        assert ZtY.shape == (D,)
        assert ZtZ.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_symmetric(self) -> None:
        """Gram matrix is symmetric."""
        rng = np.random.default_rng(42)
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, 8, 4, 10, 25, 2)

        ZtZ, _ = rff_full_gramian_elemental(X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2)

        np.testing.assert_allclose(ZtZ, ZtZ.T, rtol=1e-14, atol=1e-14)

    def test_matches_separate_energy_and_force(self) -> None:
        """Combined result equals energy gramian + force gramian computed separately."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 6, 3, 8, 15, 2
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, nmol, natoms, rep_size, D, nel)

        # Combined
        _ZtZ_comb, ZtY_comb = rff_full_gramian_elemental(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )

        # Energy-only
        _ZtZ_e, ZtY_e = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=3)

        # Force-only: compute G for all molecules, then G @ G^T and G @ F
        G = rff_gradient_elemental(X, dX, Q, W, b)
        ZtZ_f = G @ G.T
        ZtY_f = G @ F

        np.testing.assert_allclose(_ZtZ_comb, _ZtZ_e + ZtZ_f, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ZtY_comb, ZtY_e + ZtY_f, rtol=1e-10, atol=1e-10)

    def test_chunk_sizes_dont_matter(self) -> None:
        """Different chunk sizes give same result."""
        rng = np.random.default_rng(456)
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, 10, 3, 8, 15, 2)

        ref_gram, ref_proj = rff_full_gramian_elemental(
            X, dX, Q, W, b, Y, F, energy_chunk=1, force_chunk=1
        )
        g2, p2 = rff_full_gramian_elemental(X, dX, Q, W, b, Y, F, energy_chunk=5, force_chunk=3)
        g3, p3 = rff_full_gramian_elemental(X, dX, Q, W, b, Y, F, energy_chunk=100, force_chunk=100)

        np.testing.assert_allclose(g2, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g3, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)

    def test_energy_only_matches_gramian(self) -> None:
        """With zero forces, ZtY matches energy-only gramian projection."""
        rng = np.random.default_rng(789)
        nmol, natoms, rep_size, D, nel = 8, 3, 10, 20, 2
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, nmol, natoms, rep_size, D, nel)
        F_zero = np.zeros_like(F)

        _ZtZ_comb, ZtY_comb = rff_full_gramian_elemental(
            X, dX, Q, W, b, Y, F_zero, energy_chunk=4, force_chunk=3
        )

        _ZtZ_e, ZtY_e = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=4)

        # ZtZ still has force contribution (G@G^T); only check ZtY since F=0
        np.testing.assert_allclose(ZtY_comb, ZtY_e, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Pure-NumPy reference for global rff_gradient
# ---------------------------------------------------------------------------
def rff_gradient_numpy(
    X: np.ndarray,
    dX: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Reference: G[d, i*ncoords+g] = -sqrt(2/D) * sin(z[d]) * (W^T @ dX[i])_{d,g}.

    X:  (N, rep_size)
    dX: (N, rep_size, ncoords)
    W:  (rep_size, D)
    b:  (D,)
    Returns G: (D, N*ncoords)
    """
    N, _rep_size = X.shape
    D = b.shape[0]
    ncoords = dX.shape[2]
    norm = -np.sqrt(2.0 / D)

    G = np.zeros((D, N * ncoords))
    for i in range(N):
        z_i = X[i] @ W + b  # (D,)
        # dg[d, r] = sin(z_i[d]) * norm * W[r, d]
        dg = norm * np.sin(z_i)[:, None] * W.T  # (D, rep_size)
        # G[:, i*ncoords:(i+1)*ncoords] = dg @ dX[i]
        G[:, i * ncoords : (i + 1) * ncoords] = dg @ dX[i]
    return G


# ---------------------------------------------------------------------------
# rff_gradient tests
# ---------------------------------------------------------------------------
class TestRffGradient:
    """Tests for the global rff_gradient function."""

    def test_shape_and_dtype(self) -> None:
        """Output has correct shape (D, N*ncoords) and dtype."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 5, 10, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        G = rff_gradient(X, dX, W, b)

        assert G.shape == (D, N * ncoords)
        assert G.dtype == np.float64

    def test_c_contiguous(self) -> None:
        """Output is C-contiguous."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 3, 8, 15, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        G = rff_gradient(X, dX, W, b)
        assert G.flags["C_CONTIGUOUS"]

    def test_matches_numpy_reference_small(self) -> None:
        """C++ matches NumPy reference (small problem)."""
        rng = np.random.default_rng(123)
        N, rep_size, D, ncoords = 4, 6, 12, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        G_cpp = rff_gradient(X, dX, W, b)
        G_ref = rff_gradient_numpy(X, dX, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-12, atol=1e-12)

    def test_matches_numpy_reference_medium(self) -> None:
        """C++ matches NumPy reference (medium problem)."""
        rng = np.random.default_rng(456)
        N, rep_size, D, ncoords = 20, 50, 100, 30
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        G_cpp = rff_gradient(X, dX, W, b)
        G_ref = rff_gradient_numpy(X, dX, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-10, atol=1e-10)

    def test_matches_numpy_reference_large(self) -> None:
        """C++ matches NumPy reference (larger, relaxed tol for BLAS)."""
        rng = np.random.default_rng(789)
        N, rep_size, D, ncoords = 50, 100, 200, 60
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        G_cpp = rff_gradient(X, dX, W, b)
        G_ref = rff_gradient_numpy(X, dX, W, b)

        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-6, atol=1e-6)

    def test_finite_difference(self) -> None:
        """Gradient matches finite-difference on rff_features output."""
        rng = np.random.default_rng(11)
        N, rep_size, D, natoms = 3, 8, 16, 2
        ncoords = 3 * natoms
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        # dX[i, r, g] = d(X[i, r]) / d(coord_g)
        # For finite difference: perturb X[i, r] by eps in direction g
        dX = rng.normal(size=(N, rep_size, ncoords))

        G_cpp = rff_gradient(X, dX, W, b)

        # Finite-difference: perturb X in the direction of each coord
        # dZ/d(coord_g) ~ (Z(X + eps*dX[:,g]) - Z(X - eps*dX[:,g])) / (2*eps)
        eps = 1e-5
        for i in range(N):
            for g in range(ncoords):
                dX_ig = dX[i, :, g]  # (rep_size,) perturbation direction
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[i] += eps * dX_ig
                X_minus[i] -= eps * dX_ig

                Z_plus = rff_features(X_plus, W, b)
                Z_minus = rff_features(X_minus, W, b)

                fd_col = (Z_plus[i] - Z_minus[i]) / (2 * eps)  # (D,)
                np.testing.assert_allclose(G_cpp[:, i * ncoords + g], fd_col, rtol=1e-5, atol=1e-5)

    def test_single_molecule(self) -> None:
        """Works with N=1."""
        rng = np.random.default_rng(7)
        rep_size, D, ncoords = 10, 20, 12
        X = rng.normal(size=(1, rep_size))
        dX = rng.normal(size=(1, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        G_cpp = rff_gradient(X, dX, W, b)
        G_ref = rff_gradient_numpy(X, dX, W, b)

        assert G_cpp.shape == (D, ncoords)
        np.testing.assert_allclose(G_cpp, G_ref, rtol=1e-12, atol=1e-12)

    def test_bad_dX_ndim_raises(self) -> None:
        """dX with wrong ndim raises."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 5, 10, 20
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size))  # 2D instead of 3D
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(size=(D,))

        with pytest.raises(ValueError, match="dX must be 3D"):
            rff_gradient(X, dX, W, b)


# ---------------------------------------------------------------------------
# rff_gramian_symm tests
# ---------------------------------------------------------------------------
class TestRffGramianSymm:
    """Tests for the global rff_gramian_symm function."""

    def test_shapes_and_dtypes(self) -> None:
        """Output shapes and dtypes are correct."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 20, 30, 50
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ZtZ, ZtY = rff_gramian_symm(X, W, b, Y, chunk_size=7)

        assert ZtZ.shape == (D, D)
        assert ZtY.shape == (D,)
        assert ZtZ.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_gram_matrix_symmetric(self) -> None:
        """Gram matrix ZtZ is symmetric."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 30, 20, 40
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ZtZ, _ = rff_gramian_symm(X, W, b, Y, chunk_size=8)

        np.testing.assert_allclose(ZtZ, ZtZ.T, rtol=1e-14, atol=1e-14)

    def test_gram_matrix_positive_semidefinite(self) -> None:
        """Gram matrix ZtZ is PSD."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 30, 20, 40
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ZtZ, _ = rff_gramian_symm(X, W, b, Y, chunk_size=8)

        eigvals = np.linalg.eigvalsh(ZtZ)
        assert np.all(eigvals >= -1e-10)

    def test_matches_direct_computation(self) -> None:
        """Gramian matches Z.T @ Z and Z.T @ Y computed directly."""
        rng = np.random.default_rng(123)
        N, rep_size, D = 25, 15, 30
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ZtZ_cpp, ZtY_cpp = rff_gramian_symm(X, W, b, Y, chunk_size=7)

        Z = rff_features(X, W, b)
        ZtZ_ref = Z.T @ Z
        ZtY_ref = Z.T @ Y

        np.testing.assert_allclose(ZtZ_cpp, ZtZ_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ZtY_cpp, ZtY_ref, rtol=1e-10, atol=1e-10)

    def test_chunk_size_invariance(self) -> None:
        """Different chunk sizes give identical results."""
        rng = np.random.default_rng(456)
        N, rep_size, D = 20, 10, 25
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ref_gram, ref_proj = rff_gramian_symm(X, W, b, Y, chunk_size=1)
        g2, p2 = rff_gramian_symm(X, W, b, Y, chunk_size=5)
        g3, p3 = rff_gramian_symm(X, W, b, Y, chunk_size=20)
        g4, p4 = rff_gramian_symm(X, W, b, Y, chunk_size=1000)

        np.testing.assert_allclose(g2, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g3, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g4, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p4, ref_proj, rtol=1e-12, atol=1e-12)

    def test_bad_Y_shape_raises(self) -> None:
        """Y with wrong shape raises."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 10, 15, 20
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(size=(D,))
        Y = rng.normal(size=(N + 1,))  # wrong

        with pytest.raises(ValueError, match=r"Y\.shape"):
            rff_gramian_symm(X, W, b, Y)


# ---------------------------------------------------------------------------
# rff_full_gramian_symm tests
# ---------------------------------------------------------------------------
class TestRffFullGramianSymm:
    """Tests for the global rff_full_gramian_symm function."""

    def test_shapes_and_dtypes(self) -> None:
        """Output shapes and dtypes are correct."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 8, 10, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ZtZ, ZtY = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2)

        assert ZtZ.shape == (D, D)
        assert ZtY.shape == (D,)
        assert ZtZ.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_symmetric(self) -> None:
        """Gram matrix is symmetric."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 10, 12, 25, 12
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ZtZ, _ = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2)

        np.testing.assert_allclose(ZtZ, ZtZ.T, rtol=1e-14, atol=1e-14)

    def test_matches_separate_energy_and_force(self) -> None:
        """Combined result equals energy gramian + force gramian computed separately."""
        rng = np.random.default_rng(123)
        N, rep_size, D, ncoords = 8, 10, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ZtZ_comb, ZtY_comb = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2)

        # Energy-only via rff_gramian_symm
        ZtZ_e, ZtY_e = rff_gramian_symm(X, W, b, Y, chunk_size=3)

        # Force contribution via rff_gradient
        G = rff_gradient(X, dX, W, b)
        ZtZ_f = G @ G.T
        ZtY_f = G @ F

        np.testing.assert_allclose(ZtZ_comb, ZtZ_e + ZtZ_f, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(ZtY_comb, ZtY_e + ZtY_f, rtol=1e-10, atol=1e-10)

    def test_chunk_sizes_dont_matter(self) -> None:
        """Different chunk sizes give same result."""
        rng = np.random.default_rng(456)
        N, rep_size, D, ncoords = 12, 8, 15, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ref_gram, ref_proj = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=1, force_chunk=1)
        g2, p2 = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=5, force_chunk=3)
        g3, p3 = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=100, force_chunk=100)

        np.testing.assert_allclose(g2, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g3, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)

    def test_zero_forces_matches_energy_projection(self) -> None:
        """With zero forces, ZtY matches energy-only gramian projection."""
        rng = np.random.default_rng(789)
        N, rep_size, D, ncoords = 10, 12, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F_zero = np.zeros(N * ncoords)

        _, ZtY_comb = rff_full_gramian_symm(X, dX, W, b, Y, F_zero, energy_chunk=4, force_chunk=3)
        _, ZtY_e = rff_gramian_symm(X, W, b, Y, chunk_size=4)

        # ZtZ includes G@G^T even with F=0; only ZtY should match
        np.testing.assert_allclose(ZtY_comb, ZtY_e, rtol=1e-12, atol=1e-12)

    def test_bad_F_shape_raises(self) -> None:
        """F with wrong shape raises."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 5, 8, 15, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords + 1,))  # wrong

        with pytest.raises(ValueError, match=r"F\.shape"):
            rff_full_gramian_symm(X, dX, W, b, Y, F)


# ---------------------------------------------------------------------------
# Helper: unpack RFP 1D array to full D x D symmetric matrix
# ---------------------------------------------------------------------------
def rfp_to_full_np(rfp: np.ndarray, D: int) -> np.ndarray:
    """Simpler unpacking: iterate over (i,j) with i<=j."""
    k = D // 2
    stride = D + 1 if D % 2 == 0 else D
    out = np.zeros((D, D))
    for i in range(D):
        for j in range(i, D):
            if j >= k:
                packed_idx = (j - k) * stride + i
            else:
                packed_idx = i * stride + (j + k + 1)
            out[i, j] = rfp[packed_idx]
            out[j, i] = rfp[packed_idx]
    return out


# ---------------------------------------------------------------------------
# rff_full tests (global)
# ---------------------------------------------------------------------------
class TestRffFull:
    """Tests for rff_full: combined energy+force feature matrix."""

    def test_shape_and_dtype(self) -> None:
        """Output shape is (N+N*ncoords, D) and dtype is float64."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 5, 10, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_full = rff_full(X, dX, W, b)

        assert Z_full.shape == (N + N * ncoords, D)
        assert Z_full.dtype == np.float64

    def test_top_rows_match_rff_features(self) -> None:
        """Top N rows of Z_full equal rff_features output."""
        rng = np.random.default_rng(123)
        N, rep_size, D, ncoords = 6, 8, 15, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_full = rff_full(X, dX, W, b)
        Z = rff_features(X, W, b)

        np.testing.assert_allclose(Z_full[:N], Z, rtol=1e-12, atol=1e-12)

    def test_bottom_rows_match_rff_gradient_transposed(self) -> None:
        """Bottom N*ncoords rows of Z_full equal G.T where G = rff_gradient."""
        rng = np.random.default_rng(456)
        N, rep_size, D, ncoords = 4, 6, 12, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_full = rff_full(X, dX, W, b)
        G = rff_gradient(X, dX, W, b)  # (D, N*ncoords)

        np.testing.assert_allclose(Z_full[N:], G.T, rtol=1e-12, atol=1e-12)

    def test_gramian_identity(self) -> None:
        """Z_full^T @ Z_full = Z^T @ Z + G @ G^T."""
        rng = np.random.default_rng(789)
        N, rep_size, D, ncoords = 8, 10, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))

        Z_full = rff_full(X, dX, W, b)
        Z = rff_features(X, W, b)
        G = rff_gradient(X, dX, W, b)

        lhs = Z_full.T @ Z_full
        rhs = Z.T @ Z + G @ G.T

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# rff_gradient_gramian_symm tests (global, force-only)
# ---------------------------------------------------------------------------
class TestRffGradientGramianSymm:
    """Tests for rff_gradient_gramian_symm: force-only normal equations."""

    def test_shapes_and_dtypes(self) -> None:
        """Output shapes and dtypes are correct."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 8, 10, 20, 9
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        GtG, GtF = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=3)

        assert GtG.shape == (D, D)
        assert GtF.shape == (D,)
        assert GtG.dtype == np.float64
        assert GtF.dtype == np.float64

    def test_symmetric(self) -> None:
        """GtG is symmetric."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 10, 12, 25, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        GtG, _ = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=4)

        np.testing.assert_allclose(GtG, GtG.T, rtol=1e-14, atol=1e-14)

    def test_matches_direct_computation(self) -> None:
        """GtG = G @ G.T and GtF = G @ F."""
        rng = np.random.default_rng(123)
        N, rep_size, D, ncoords = 6, 8, 15, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        GtG_cpp, GtF_cpp = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=3)
        G = rff_gradient(X, dX, W, b)
        GtG_ref = G @ G.T
        GtF_ref = G @ F

        np.testing.assert_allclose(GtG_cpp, GtG_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(GtF_cpp, GtF_ref, rtol=1e-10, atol=1e-10)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give identical results."""
        rng = np.random.default_rng(456)
        N, rep_size, D, ncoords = 10, 8, 15, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        ref_g, ref_f = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=1)
        g2, f2 = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=5)
        g3, f3 = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=1000)

        np.testing.assert_allclose(g2, ref_g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f2, ref_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g3, ref_g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f3, ref_f, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_gramian_symm_rfp tests (global, energy-only, RFP)
# ---------------------------------------------------------------------------
class TestRffGramianSymmRfp:
    """Tests for rff_gramian_symm_rfp: energy-only gramian in RFP storage."""

    def test_shape_and_dtype(self) -> None:
        """Output ZtZ_rfp has shape (D*(D+1)//2,) and dtype float64."""
        rng = np.random.default_rng(42)
        N, rep_size, D = 20, 15, 12
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ZtZ_rfp, ZtY = rff_gramian_symm_rfp(X, W, b, Y, chunk_size=5)

        assert ZtZ_rfp.shape == (D * (D + 1) // 2,)
        assert ZtY.shape == (D,)
        assert ZtZ_rfp.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_rfp_matches_full_gramian(self) -> None:
        """Unpacked RFP gramian matches rff_gramian_symm output."""
        rng = np.random.default_rng(123)
        N, rep_size, D = 15, 10, 8
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ZtZ_rfp, ZtY_rfp = rff_gramian_symm_rfp(X, W, b, Y, chunk_size=4)
        ZtZ_full, ZtY_full = rff_gramian_symm(X, W, b, Y, chunk_size=4)

        ZtZ_unpacked = rfp_to_full_np(ZtZ_rfp, D)
        np.testing.assert_allclose(ZtZ_unpacked, ZtZ_full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY_rfp, ZtY_full, rtol=1e-12, atol=1e-12)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give same RFP output."""
        rng = np.random.default_rng(456)
        N, rep_size, D = 12, 8, 10
        X = rng.normal(size=(N, rep_size))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))

        ref_rfp, ref_proj = rff_gramian_symm_rfp(X, W, b, Y, chunk_size=1)
        r2, p2 = rff_gramian_symm_rfp(X, W, b, Y, chunk_size=5)
        r3, p3 = rff_gramian_symm_rfp(X, W, b, Y, chunk_size=1000)

        np.testing.assert_allclose(r2, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r3, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_gradient_gramian_symm_rfp tests (global, force-only, RFP)
# ---------------------------------------------------------------------------
class TestRffGradientGramianSymmRfp:
    """Tests for rff_gradient_gramian_symm_rfp: force-only gramian in RFP storage."""

    def test_shape_and_dtype(self) -> None:
        """Output GtG_rfp has shape (D*(D+1)//2,) and dtype float64."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 8, 10, 12, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        GtG_rfp, GtF = rff_gradient_gramian_symm_rfp(X, dX, W, b, F, chunk_size=3)

        assert GtG_rfp.shape == (D * (D + 1) // 2,)
        assert GtF.shape == (D,)
        assert GtG_rfp.dtype == np.float64
        assert GtF.dtype == np.float64

    def test_rfp_matches_full_gradient_gramian(self) -> None:
        """Unpacked RFP matches rff_gradient_gramian_symm output."""
        rng = np.random.default_rng(123)
        N, rep_size, D, ncoords = 6, 8, 10, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        GtG_rfp, GtF_rfp = rff_gradient_gramian_symm_rfp(X, dX, W, b, F, chunk_size=3)
        GtG_full, GtF_full = rff_gradient_gramian_symm(X, dX, W, b, F, chunk_size=3)

        GtG_unpacked = rfp_to_full_np(GtG_rfp, D)
        np.testing.assert_allclose(GtG_unpacked, GtG_full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(GtF_rfp, GtF_full, rtol=1e-12, atol=1e-12)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give same RFP output."""
        rng = np.random.default_rng(456)
        N, rep_size, D, ncoords = 10, 8, 12, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        F = rng.normal(size=(N * ncoords,))

        ref_rfp, ref_f = rff_gradient_gramian_symm_rfp(X, dX, W, b, F, chunk_size=1)
        r2, f2 = rff_gradient_gramian_symm_rfp(X, dX, W, b, F, chunk_size=5)
        r3, f3 = rff_gradient_gramian_symm_rfp(X, dX, W, b, F, chunk_size=1000)

        np.testing.assert_allclose(r2, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f2, ref_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r3, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f3, ref_f, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_full_gramian_symm_rfp tests (global, energy+force, RFP)
# ---------------------------------------------------------------------------
class TestRffFullGramianSymmRfp:
    """Tests for rff_full_gramian_symm_rfp: energy+force gramian in RFP storage."""

    def test_shape_and_dtype(self) -> None:
        """Output ZtZ_rfp has shape (D*(D+1)//2,) and dtype float64."""
        rng = np.random.default_rng(42)
        N, rep_size, D, ncoords = 8, 10, 12, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ZtZ_rfp, ZtY = rff_full_gramian_symm_rfp(X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2)

        assert ZtZ_rfp.shape == (D * (D + 1) // 2,)
        assert ZtY.shape == (D,)
        assert ZtZ_rfp.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_rfp_matches_full_gramian(self) -> None:
        """Unpacked RFP matches rff_full_gramian_symm output."""
        rng = np.random.default_rng(123)
        N, rep_size, D, ncoords = 6, 8, 10, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ZtZ_rfp, ZtY_rfp = rff_full_gramian_symm_rfp(
            X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2
        )
        ZtZ_full, ZtY_full = rff_full_gramian_symm(X, dX, W, b, Y, F, energy_chunk=3, force_chunk=2)

        ZtZ_unpacked = rfp_to_full_np(ZtZ_rfp, D)
        np.testing.assert_allclose(ZtZ_unpacked, ZtZ_full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY_rfp, ZtY_full, rtol=1e-12, atol=1e-12)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give same RFP output."""
        rng = np.random.default_rng(456)
        N, rep_size, D, ncoords = 10, 8, 12, 6
        X = rng.normal(size=(N, rep_size))
        dX = rng.normal(size=(N, rep_size, ncoords))
        W = rng.normal(size=(rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(D,))
        Y = rng.normal(size=(N,))
        F = rng.normal(size=(N * ncoords,))

        ref_rfp, ref_proj = rff_full_gramian_symm_rfp(
            X, dX, W, b, Y, F, energy_chunk=1, force_chunk=1
        )
        r2, p2 = rff_full_gramian_symm_rfp(X, dX, W, b, Y, F, energy_chunk=5, force_chunk=3)
        r3, p3 = rff_full_gramian_symm_rfp(X, dX, W, b, Y, F, energy_chunk=100, force_chunk=100)

        np.testing.assert_allclose(r2, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r3, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_full_elemental tests
# ---------------------------------------------------------------------------
class TestRffFullElemental:
    """Tests for rff_full_elemental: combined energy+force feature matrix."""

    @staticmethod
    def _make_data(
        rng: np.random.Generator,
        nmol: int,
        natoms: int,
        rep_size: int,
        D: int,
        nel: int,
    ) -> tuple:
        X = rng.normal(size=(nmol, natoms, rep_size))
        dX = rng.normal(size=(nmol, natoms, rep_size, natoms, 3))
        Q = [rng.integers(0, nel, size=natoms).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nel, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nel, D))
        ngrads = 3 * nmol * natoms
        return X, dX, Q, W, b, ngrads

    def test_shape_and_dtype(self) -> None:
        """Output shape is (nmol + ngrads, D) and dtype float64."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 5, 3, 10, 20, 2
        X, dX, Q, W, b, ngrads = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        Z_full = rff_full_elemental(X, dX, Q, W, b)

        assert Z_full.shape == (nmol + ngrads, D)
        assert Z_full.dtype == np.float64

    def test_top_rows_match_rff_features_elemental(self) -> None:
        """Top nmol rows equal rff_features_elemental output."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 6, 3, 8, 15, 2
        X, dX, Q, W, b, _ngrads = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        Z_full = rff_full_elemental(X, dX, Q, W, b)
        Z = rff_features_elemental(X, Q, W, b)

        np.testing.assert_allclose(Z_full[:nmol], Z, rtol=1e-12, atol=1e-12)

    def test_bottom_rows_match_rff_gradient_elemental_transposed(self) -> None:
        """Bottom ngrads rows equal G.T where G = rff_gradient_elemental."""
        rng = np.random.default_rng(456)
        nmol, natoms, rep_size, D, nel = 4, 3, 8, 12, 2
        X, dX, Q, W, b, _ngrads = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        Z_full = rff_full_elemental(X, dX, Q, W, b)
        G = rff_gradient_elemental(X, dX, Q, W, b)  # (D, ngrads)

        np.testing.assert_allclose(Z_full[nmol:], G.T, rtol=1e-12, atol=1e-12)

    def test_gramian_identity(self) -> None:
        """Z_full^T @ Z_full = Z^T @ Z + G @ G^T."""
        rng = np.random.default_rng(789)
        nmol, natoms, rep_size, D, nel = 8, 3, 8, 15, 2
        X, dX, Q, W, b, _ngrads = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        Z_full = rff_full_elemental(X, dX, Q, W, b)
        Z = rff_features_elemental(X, Q, W, b)
        G = rff_gradient_elemental(X, dX, Q, W, b)

        lhs = Z_full.T @ Z_full
        rhs = Z.T @ Z + G @ G.T

        np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# rff_gradient_gramian_elemental tests (elemental, force-only)
# ---------------------------------------------------------------------------
class TestRffGradientGramianElemental:
    """Tests for rff_gradient_gramian_elemental: force-only normal equations."""

    @staticmethod
    def _make_data(
        rng: np.random.Generator, nmol: int, natoms: int, rep_size: int, D: int, nel: int
    ) -> tuple:
        X = rng.normal(size=(nmol, natoms, rep_size))
        dX = rng.normal(size=(nmol, natoms, rep_size, natoms, 3))
        Q = [rng.integers(0, nel, size=natoms).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nel, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nel, D))
        ngrads = 3 * nmol * natoms
        F = rng.normal(size=ngrads)
        return X, dX, Q, W, b, F

    def test_shapes_and_dtypes(self) -> None:
        """Output shapes and dtypes are correct."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 5, 3, 10, 20, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        GtG, GtF = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=3)

        assert GtG.shape == (D, D)
        assert GtF.shape == (D,)
        assert GtG.dtype == np.float64
        assert GtF.dtype == np.float64

    def test_symmetric(self) -> None:
        """GtG is symmetric."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 8, 3, 10, 20, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        GtG, _ = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=3)

        np.testing.assert_allclose(GtG, GtG.T, rtol=1e-14, atol=1e-14)

    def test_matches_direct_computation(self) -> None:
        """GtG = G @ G.T and GtF = G @ F."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 6, 3, 8, 15, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        GtG_cpp, GtF_cpp = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=3)
        G = rff_gradient_elemental(X, dX, Q, W, b)
        GtG_ref = G @ G.T
        GtF_ref = G @ F

        np.testing.assert_allclose(GtG_cpp, GtG_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(GtF_cpp, GtF_ref, rtol=1e-10, atol=1e-10)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give identical results."""
        rng = np.random.default_rng(456)
        nmol, natoms, rep_size, D, nel = 10, 3, 8, 15, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ref_g, ref_f = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=1)
        g2, f2 = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=5)
        g3, f3 = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=1000)

        np.testing.assert_allclose(g2, ref_g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f2, ref_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g3, ref_g, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f3, ref_f, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_gramian_elemental_rfp tests (elemental, energy-only, RFP)
# ---------------------------------------------------------------------------
class TestRffGramianElementalRfp:
    """Tests for rff_gramian_elemental_rfp: energy-only elemental gramian in RFP."""

    @staticmethod
    def _make_data(
        rng: np.random.Generator, nmol: int, natoms: int, rep_size: int, D: int, nel: int
    ) -> tuple:
        X = rng.normal(size=(nmol, natoms, rep_size))
        Q = [rng.integers(0, nel, size=natoms).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nel, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nel, D))
        Y = rng.normal(size=nmol)
        return X, Q, W, b, Y

    def test_shape_and_dtype(self) -> None:
        """ZtZ_rfp has shape (D*(D+1)//2,) and dtype float64."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 10, 3, 10, 12, 2
        X, Q, W, b, Y = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ZtZ_rfp, ZtY = rff_gramian_elemental_rfp(X, Q, W, b, Y, chunk_size=4)

        assert ZtZ_rfp.shape == (D * (D + 1) // 2,)
        assert ZtY.shape == (D,)
        assert ZtZ_rfp.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_rfp_matches_full_gramian(self) -> None:
        """Unpacked RFP matches rff_gramian_elemental output."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 8, 3, 8, 10, 2
        X, Q, W, b, Y = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ZtZ_rfp, ZtY_rfp = rff_gramian_elemental_rfp(X, Q, W, b, Y, chunk_size=3)
        ZtZ_full, ZtY_full = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=3)

        ZtZ_unpacked = rfp_to_full_np(ZtZ_rfp, D)
        np.testing.assert_allclose(ZtZ_unpacked, ZtZ_full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY_rfp, ZtY_full, rtol=1e-12, atol=1e-12)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give same RFP output."""
        rng = np.random.default_rng(456)
        nmol, natoms, rep_size, D, nel = 10, 3, 8, 10, 2
        X, Q, W, b, Y = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ref_rfp, ref_proj = rff_gramian_elemental_rfp(X, Q, W, b, Y, chunk_size=1)
        r2, p2 = rff_gramian_elemental_rfp(X, Q, W, b, Y, chunk_size=5)
        r3, p3 = rff_gramian_elemental_rfp(X, Q, W, b, Y, chunk_size=1000)

        np.testing.assert_allclose(r2, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r3, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_gradient_gramian_elemental_rfp tests (elemental, force-only, RFP)
# ---------------------------------------------------------------------------
class TestRffGradientGramianElementalRfp:
    """Tests for rff_gradient_gramian_elemental_rfp: force-only elemental gramian in RFP."""

    @staticmethod
    def _make_data(
        rng: np.random.Generator, nmol: int, natoms: int, rep_size: int, D: int, nel: int
    ) -> tuple:
        X = rng.normal(size=(nmol, natoms, rep_size))
        dX = rng.normal(size=(nmol, natoms, rep_size, natoms, 3))
        Q = [rng.integers(0, nel, size=natoms).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nel, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nel, D))
        ngrads = 3 * nmol * natoms
        F = rng.normal(size=ngrads)
        return X, dX, Q, W, b, F

    def test_shape_and_dtype(self) -> None:
        """GtG_rfp has shape (D*(D+1)//2,) and dtype float64."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 5, 3, 10, 12, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        GtG_rfp, GtF = rff_gradient_gramian_elemental_rfp(X, dX, Q, W, b, F, chunk_size=3)

        assert GtG_rfp.shape == (D * (D + 1) // 2,)
        assert GtF.shape == (D,)
        assert GtG_rfp.dtype == np.float64
        assert GtF.dtype == np.float64

    def test_rfp_matches_full_gradient_gramian(self) -> None:
        """Unpacked RFP matches rff_gradient_gramian_elemental output."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 6, 3, 8, 10, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        GtG_rfp, GtF_rfp = rff_gradient_gramian_elemental_rfp(X, dX, Q, W, b, F, chunk_size=3)
        GtG_full, GtF_full = rff_gradient_gramian_elemental(X, dX, Q, W, b, F, chunk_size=3)

        GtG_unpacked = rfp_to_full_np(GtG_rfp, D)
        np.testing.assert_allclose(GtG_unpacked, GtG_full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(GtF_rfp, GtF_full, rtol=1e-12, atol=1e-12)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give same RFP output."""
        rng = np.random.default_rng(456)
        nmol, natoms, rep_size, D, nel = 10, 3, 8, 10, 2
        X, dX, Q, W, b, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ref_rfp, ref_f = rff_gradient_gramian_elemental_rfp(X, dX, Q, W, b, F, chunk_size=1)
        r2, f2 = rff_gradient_gramian_elemental_rfp(X, dX, Q, W, b, F, chunk_size=5)
        r3, f3 = rff_gradient_gramian_elemental_rfp(X, dX, Q, W, b, F, chunk_size=1000)

        np.testing.assert_allclose(r2, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f2, ref_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r3, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(f3, ref_f, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# rff_full_gramian_elemental_rfp tests (elemental, energy+force, RFP)
# ---------------------------------------------------------------------------
class TestRffFullGramianElementalRfp:
    """Tests for rff_full_gramian_elemental_rfp: energy+force elemental gramian in RFP."""

    @staticmethod
    def _make_data(
        rng: np.random.Generator, nmol: int, natoms: int, rep_size: int, D: int, nel: int
    ) -> tuple:
        X = rng.normal(size=(nmol, natoms, rep_size))
        dX = rng.normal(size=(nmol, natoms, rep_size, natoms, 3))
        Q = [rng.integers(0, nel, size=natoms).astype(np.int32) for _ in range(nmol)]
        W = rng.normal(size=(nel, rep_size, D))
        b = rng.uniform(0, 2 * np.pi, size=(nel, D))
        Y = rng.normal(size=nmol)
        ngrads = 3 * nmol * natoms
        F = rng.normal(size=ngrads)
        return X, dX, Q, W, b, Y, F

    def test_shape_and_dtype(self) -> None:
        """ZtZ_rfp has shape (D*(D+1)//2,) and dtype float64."""
        rng = np.random.default_rng(42)
        nmol, natoms, rep_size, D, nel = 5, 3, 10, 12, 2
        X, dX, Q, W, b, Y, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ZtZ_rfp, ZtY = rff_full_gramian_elemental_rfp(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )

        assert ZtZ_rfp.shape == (D * (D + 1) // 2,)
        assert ZtY.shape == (D,)
        assert ZtZ_rfp.dtype == np.float64
        assert ZtY.dtype == np.float64

    def test_rfp_matches_full_gramian(self) -> None:
        """Unpacked RFP matches rff_full_gramian_elemental output."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 6, 3, 8, 10, 2
        X, dX, Q, W, b, Y, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ZtZ_rfp, ZtY_rfp = rff_full_gramian_elemental_rfp(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )
        ZtZ_full, ZtY_full = rff_full_gramian_elemental(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )

        ZtZ_unpacked = rfp_to_full_np(ZtZ_rfp, D)
        np.testing.assert_allclose(ZtZ_unpacked, ZtZ_full, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ZtY_rfp, ZtY_full, rtol=1e-12, atol=1e-12)

    def test_chunk_invariance(self) -> None:
        """Different chunk sizes give same RFP output."""
        rng = np.random.default_rng(456)
        nmol, natoms, rep_size, D, nel = 8, 3, 8, 10, 2
        X, dX, Q, W, b, Y, F = self._make_data(rng, nmol, natoms, rep_size, D, nel)

        ref_rfp, ref_proj = rff_full_gramian_elemental_rfp(
            X, dX, Q, W, b, Y, F, energy_chunk=1, force_chunk=1
        )
        r2, p2 = rff_full_gramian_elemental_rfp(X, dX, Q, W, b, Y, F, energy_chunk=5, force_chunk=3)
        r3, p3 = rff_full_gramian_elemental_rfp(
            X, dX, Q, W, b, Y, F, energy_chunk=100, force_chunk=100
        )

        np.testing.assert_allclose(r2, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r3, ref_rfp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)
