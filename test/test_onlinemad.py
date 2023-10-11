import pytest

import experimentalhmc as ehmc
import numpy as np

def mad(x):
    return np.median(np.abs(x - np.median(x)))

def test_onlinequantile():
    N = 5_000
    D = 3
    rng = np.random.default_rng()
    x = rng.normal(size = (N, D))

    omad = ehmc.OnlineMAD(D)

    for n in range(N):
        omad.update(x[n, :])

    om = omad.mad()
    bm = [mad(x[:, d]) for d in range(D)]
    assert np.isclose(om[0], bm[0], atol = 1e-1)
    assert np.isclose(om[1], bm[1], atol = 1e-1)
    assert np.isclose(om[2], bm[2], atol = 1e-1)
