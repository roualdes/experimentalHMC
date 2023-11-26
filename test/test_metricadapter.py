import pytest

import experimentalhmc as ehmc
import numpy as np

def mad2(x):
    return np.median(np.abs(x - np.median(x, axis = 0)), axis = 0) ** 2

def test_onlinemean():
    N = 1_000
    D = 3
    rng = np.random.default_rng()
    x = rng.normal(size = (N, D))

    mmv = ehmc.MetricOnlineMeanVar(D)
    mmad = ehmc.MetricOnlineMAD(D)

    for n in range(N):
        mmv.update(x[n])
        mmad.update(x[n])

    w = N / (N + 5)
    assert np.allclose(mmv.metric(), w * np.var(x, axis = 0) + (1 - w) * 1e-3)
    assert np.allclose(mmv.location(), np.mean(x, axis = 0))

    assert np.allclose(mmad.metric(), w * mad2(x) + (1 - w) * 1e-3, atol = 1e-1)
    assert np.allclose(mmad.location(), np.median(x, axis = 0), atol = 1e-1)
