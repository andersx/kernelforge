import numpy as np
from numpy.typing import NDArray

def rff_features(
    X: NDArray[np.float64],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def rff_features_elemental(
    X: NDArray[np.float64],
    Q: list[NDArray[np.int32]],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def rff_gradient_elemental(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    Q: list[NDArray[np.int32]],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def rff_gramian_elemental(
    X: NDArray[np.float64],
    Q: list[NDArray[np.int32]],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
    Y: NDArray[np.float64],
    chunk_size: int = ...,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def rff_gramian_elemental_gradient(
    X: NDArray[np.float64],
    dX: NDArray[np.float64],
    Q: list[NDArray[np.int32]],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
    Y: NDArray[np.float64],
    F: NDArray[np.float64],
    energy_chunk: int = ...,
    force_chunk: int = ...,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
