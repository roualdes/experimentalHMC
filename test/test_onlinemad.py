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
        omad.update(x[n])

    bmed = np.zeros(D)
    bmad = np.zeros(D)
    for d in range(D):
        y = x[:, d]
        bmed[d] = np.median(y)
        bmad[d] = mad(y)

    assert np.allclose(np.round(omad.location(), 2), np.round(bmed, 2), atol = 1e-1)
    assert np.allclose(np.round(omad.scale(), 2), np.round(bmad, 2), atol = 1e-1)
