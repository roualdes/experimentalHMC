import pytest

import experimentalhmc as ehmc
import numpy as np

def test_onlinemeanvar():
    N = 100
    D = 3
    rng = np.random.default_rng()
    x = rng.normal(size = (N, D))

    omv = ehmc.OnlineMeanVar(D)
    for n in range(N):
        omv.update(x[n])

    assert np.allclose(omv.location(), np.mean(x, axis = 0))
    assert np.allclose(omv.scale(), np.std(x, axis = 0))
    assert np.isclose(omv.N(), N)

    omv.reset()
    assert np.allclose(omv.location(), np.zeros(D))
    assert np.allclose(omv.scale(), np.zeros(D))
    assert np.isclose(omv.N(), 0)
