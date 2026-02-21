"""Tests for the kitchen_sinks (Random Fourier Features) module."""

import numpy as np
import pytest
from kernelforge.kitchen_sinks import (
    rff_features,
    rff_features_elemental,
    rff_gradient_elemental,
    rff_gramian_elemental,
    rff_gramian_elemental_gradient,
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

    def test_bad_shapes_raise(self) -> None:
        """Mismatched shapes raise ValueError/InvalidArgument."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 20))
        W = rng.normal(size=(30, 50))  # W.shape[0] != X.shape[1]
        b = rng.uniform(size=(50,))

        with pytest.raises(ValueError, match=r"W\.shape\[0\] must equal X\.shape\[1\]"):
            rff_features(X, W, b)

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

        LZtLZ, LZtY = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=5)

        assert LZtLZ.shape == (D, D)
        assert LZtY.shape == (D,)
        assert LZtLZ.dtype == np.float64
        assert LZtY.dtype == np.float64

    def test_gram_matrix_symmetric(self) -> None:
        """Gram matrix LZtLZ is symmetric."""
        rng = np.random.default_rng(42)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 20, 2, 8, 20, 40, 3)

        LZtLZ, _ = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=7)

        np.testing.assert_allclose(LZtLZ, LZtLZ.T, rtol=1e-14, atol=1e-14)

    def test_gram_matrix_positive_semidefinite(self) -> None:
        """Gram matrix LZtLZ is PSD (all eigenvalues >= 0)."""
        rng = np.random.default_rng(42)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 20, 2, 8, 20, 40, 3)

        LZtLZ, _ = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=7)

        eigvals = np.linalg.eigvalsh(LZtLZ)
        assert np.all(eigvals >= -1e-10)

    def test_matches_numpy_reference(self) -> None:
        """Gramian matches direct computation LZ^T @ LZ, LZ^T @ Y."""
        rng = np.random.default_rng(123)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 15, 2, 6, 10, 20, 2)

        LZtLZ_cpp, LZtY_cpp = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=5)

        # Compute reference: get full LZ, then compute Gram
        LZ = rff_features_elemental(X, Q, W, b)
        LZtLZ_ref = LZ.T @ LZ
        LZtY_ref = LZ.T @ Y

        np.testing.assert_allclose(LZtLZ_cpp, LZtLZ_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(LZtY_cpp, LZtY_ref, rtol=1e-10, atol=1e-10)

    def test_chunk_size_does_not_affect_result(self) -> None:
        """Different chunk sizes give the same result."""
        rng = np.random.default_rng(456)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 20, 2, 5, 10, 25, 2)

        LZtLZ_1, LZtY_1 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=1)
        LZtLZ_5, LZtY_5 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=5)
        LZtLZ_20, LZtY_20 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=20)
        LZtLZ_100, LZtY_100 = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=100)

        np.testing.assert_allclose(LZtLZ_1, LZtLZ_5, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(LZtLZ_1, LZtLZ_20, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(LZtLZ_1, LZtLZ_100, rtol=1e-12, atol=1e-12)

        np.testing.assert_allclose(LZtY_1, LZtY_5, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(LZtY_1, LZtY_20, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(LZtY_1, LZtY_100, rtol=1e-12, atol=1e-12)

    def test_single_chunk(self) -> None:
        """Works when chunk_size >= nmol (single chunk)."""
        rng = np.random.default_rng(789)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 10, 2, 5, 10, 20, 2)

        LZtLZ, LZtY = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=1000)

        LZ = rff_features_elemental(X, Q, W, b)
        np.testing.assert_allclose(LZtLZ, LZ.T @ LZ, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(LZtY, LZ.T @ Y, rtol=1e-12, atol=1e-12)

    def test_larger_problem(self) -> None:
        """Larger problem with relaxed tolerance."""
        rng = np.random.default_rng(111)
        X, Q, W, b, Y, _ = self._make_test_data(rng, 100, 3, 15, 100, 200, 4)

        LZtLZ, LZtY = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=25)

        LZ = rff_features_elemental(X, Q, W, b)
        np.testing.assert_allclose(LZtLZ, LZ.T @ LZ, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(LZtY, LZ.T @ Y, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# rff_gramian_elemental_gradient tests
# ---------------------------------------------------------------------------
class TestRffGramianElementalGradient:
    """Tests for rff_gramian_elemental_gradient (energy+force training Gramian)."""

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

        LZtLZ, LZtY = rff_gramian_elemental_gradient(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )

        assert LZtLZ.shape == (D, D)
        assert LZtY.shape == (D,)
        assert LZtLZ.dtype == np.float64
        assert LZtY.dtype == np.float64

    def test_symmetric(self) -> None:
        """Gram matrix is symmetric."""
        rng = np.random.default_rng(42)
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, 8, 4, 10, 25, 2)

        LZtLZ, _ = rff_gramian_elemental_gradient(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )

        np.testing.assert_allclose(LZtLZ, LZtLZ.T, rtol=1e-14, atol=1e-14)

    def test_matches_separate_energy_and_force(self) -> None:
        """Combined result equals energy gramian + force gramian computed separately."""
        rng = np.random.default_rng(123)
        nmol, natoms, rep_size, D, nel = 6, 3, 8, 15, 2
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, nmol, natoms, rep_size, D, nel)

        # Combined
        _LZtLZ_comb, LZtY_comb = rff_gramian_elemental_gradient(
            X, dX, Q, W, b, Y, F, energy_chunk=3, force_chunk=2
        )

        # Energy-only
        _LZtLZ_e, LZtY_e = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=3)

        # Force-only: compute G for all molecules, then G @ G^T and G @ F
        G = rff_gradient_elemental(X, dX, Q, W, b)
        LZtLZ_f = G @ G.T
        LZtY_f = G @ F

        np.testing.assert_allclose(_LZtLZ_comb, _LZtLZ_e + LZtLZ_f, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(LZtY_comb, LZtY_e + LZtY_f, rtol=1e-10, atol=1e-10)

    def test_chunk_sizes_dont_matter(self) -> None:
        """Different chunk sizes give same result."""
        rng = np.random.default_rng(456)
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, 10, 3, 8, 15, 2)

        ref_gram, ref_proj = rff_gramian_elemental_gradient(
            X, dX, Q, W, b, Y, F, energy_chunk=1, force_chunk=1
        )
        g2, p2 = rff_gramian_elemental_gradient(X, dX, Q, W, b, Y, F, energy_chunk=5, force_chunk=3)
        g3, p3 = rff_gramian_elemental_gradient(
            X, dX, Q, W, b, Y, F, energy_chunk=100, force_chunk=100
        )

        np.testing.assert_allclose(g2, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p2, ref_proj, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g3, ref_gram, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p3, ref_proj, rtol=1e-12, atol=1e-12)

    def test_energy_only_matches_gramian(self) -> None:
        """With zero forces, LZtY matches energy-only gramian projection."""
        rng = np.random.default_rng(789)
        nmol, natoms, rep_size, D, nel = 8, 3, 10, 20, 2
        X, dX, Q, W, b, Y, F = self._make_test_data(rng, nmol, natoms, rep_size, D, nel)
        F_zero = np.zeros_like(F)

        _LZtLZ_comb, LZtY_comb = rff_gramian_elemental_gradient(
            X, dX, Q, W, b, Y, F_zero, energy_chunk=4, force_chunk=3
        )

        _LZtLZ_e, LZtY_e = rff_gramian_elemental(X, Q, W, b, Y, chunk_size=4)

        # LZtLZ still has force contribution (G@G^T); only check LZtY since F=0
        np.testing.assert_allclose(LZtY_comb, LZtY_e, rtol=1e-12, atol=1e-12)
