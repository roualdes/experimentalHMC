import pytest

import experimentalhmc as ehmc
import numpy as np

def test_onlinemean():
    N = 100
    C = 2
    D = 3
    rng = np.random.default_rng()
    x = rng.normal(size = (N, C, D))

    omv = ehmc.OnlineMean(C, D)
    for n in range(N):
        omv.update(x[n])

    assert np.allclose(omv.mean(), np.mean(x, axis = 0))

def test_onlinemeanvar():
    N = 100
    C = 2
    D = 3
    rng = np.random.default_rng()
    x = rng.normal(size = (N, C, D))

    omv = ehmc.OnlineMeanVar(C, D)
    for n in range(N):
        omv.update(x[n])

    assert np.allclose(omv.mean(), np.mean(x, axis = 0))
    assert np.allclose(omv.var(), np.var(x, axis = 0))
    assert np.allclose(omv.std(), np.std(x, axis = 0))
