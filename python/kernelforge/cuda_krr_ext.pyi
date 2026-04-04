"""Type stub for the cuda_krr_ext C extension module.

cuda_krr_ext is built only when CUDA is available at build time.
Import it with a try/except guard in production code.
"""

import numpy as np
from numpy.typing import NDArray

class CudaKRRState:
    """Persistent GPU state for Gaussian KRR with joint E+F training.

    After ``train_ef()`` the training Jacobians are contracted into
    ``W_F_bar`` and freed from GPU memory.  Subsequent ``predict()`` calls
    reuse the compact persistent state (3 * N_train * M + 3 * N_train floats
    on GPU).
    """

    def __init__(self) -> None: ...
    def train_ef(
        self,
        X: NDArray[np.float32],  # (N, M)
        dXT: NDArray[np.float32],  # (N*D, M)
        E: NDArray[np.float32],  # (N,)
        F: NDArray[np.float32],  # (N*D,)  physical forces F = -dE/dR
        sigma: float,
        lam: float,
    ) -> NDArray[np.float32]:
        """Train the model and initialise the GPU inference state.

        Parameters
        ----------
        X : ndarray (N, M) float32 C-contiguous
            Training descriptors.
        dXT : ndarray (N*D, M) float32 C-contiguous
            Flattened Jacobians — dX (N, D, M) reshaped to (N*D, M).
        E : ndarray (N,) float32
            Training energies.
        F : ndarray (N*D,) float32
            Training physical forces F = -dE/dR (CUDA negates them internally).
        sigma : float
            Gaussian kernel length-scale.
        lam : float
            L2 regularisation (added to kernel diagonal).

        Returns
        -------
        alpha : ndarray (N*(1+D),) float32
            Dual coefficients.  alpha[:N] are energy weights; alpha[N:] force weights.
        """
        ...

    def predict(
        self,
        X_te: NDArray[np.float32],  # (N_test, M)
        dXT_te: NDArray[np.float32],  # (N_test*D, M)
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Predict energies and forces.

        Parameters
        ----------
        X_te : ndarray (N_test, M) float32 C-contiguous
        dXT_te : ndarray (N_test*D, M) float32 C-contiguous

        Returns
        -------
        E_pred : ndarray (N_test,) float32
        F_pred : ndarray (N_test*D,) float32  — physical forces F = -dE/dR
        """
        ...

    def get_state(
        self,
    ) -> tuple[
        NDArray[np.float32],  # X_train    (N, M)
        NDArray[np.float32],  # W_F_bar    (N, M)
        NDArray[np.float32],  # W_combined (N, M)
        NDArray[np.float32],  # W_F_self   (N,)
        NDArray[np.float32],  # alpha_E    (N,)
        NDArray[np.float32],  # norms_tr   (N,)
    ]:
        """Extract the six inference-state arrays from GPU to host (for save)."""
        ...

    def load_state(
        self,
        X_train: NDArray[np.float32],  # (N, M)
        W_F_bar: NDArray[np.float32],  # (N, M)
        W_combined: NDArray[np.float32],  # (N, M)
        W_F_self: NDArray[np.float32],  # (N,)
        alpha_E: NDArray[np.float32],  # (N,)
        norms_tr: NDArray[np.float32],  # (N,)
        sigma: float,
        N_train: int,
        M: int,
        D: int,
    ) -> None:
        """Reconstruct the GPU inference state from saved arrays (for load)."""
        ...

    @property
    def n_train(self) -> int:
        """Number of training molecules."""
        ...

    @property
    def m(self) -> int:
        """Descriptor dimension M = N_atoms*(N_atoms-1)/2."""
        ...

    @property
    def d(self) -> int:
        """Degrees of freedom per molecule D = 3*N_atoms."""
        ...

    @property
    def sigma(self) -> float:
        """Gaussian kernel length-scale."""
        ...

    @property
    def is_ready(self) -> bool:
        """True after train_ef() or load_state()."""
        ...
