import pytest

import experimentalhmc as ehmc
import numpy as np

def test_onlinemean():
    N = 100
    D = 3
    rng = np.random.default_rng()
    x = rng.normal(size = (N, D))

    ma = ehmc.MetricAdapter(D)
    for n in range(N):
        ma.update(x[n])

    w = N / (N + 5)
    assert np.allclose(ma.metric(), w * np.var(x, axis = 0) + (1 - w) * 1e-3)
    assert np.allclose(ma.mean(), np.mean(x, axis = 0))
