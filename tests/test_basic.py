import numpy as np
import kernelforge as kf

def test_inverse_distance_shapes():
    X = np.random.rand(5, 3)
    D = kf.inverse_distance(X)
    assert D.shape == (5*4//2,)

