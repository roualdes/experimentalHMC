import pytest

import experimentalhmc as ehmc
import numpy as np

def test_onlinequantile():
    N = 5_000
    rng = np.random.default_rng()
    x = rng.normal(size = N)

    oq05 = ehmc.OnlineQuantile(0.05)
    oq25 = ehmc.OnlineQuantile(0.25)
    oq50 = ehmc.OnlineQuantile(0.5)
    oq75 = ehmc.OnlineQuantile(0.75)
    oq95 = ehmc.OnlineQuantile(0.95)

    for n in range(N):
        oq05.update(x[n])
        oq25.update(x[n])
        oq50.update(x[n])
        oq75.update(x[n])
        oq95.update(x[n])

    qs = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.isclose(oq05.quantile(), qs[0], atol = 1e-1)
    assert np.isclose(oq25.quantile(), qs[1], atol = 1e-1)
    assert np.isclose(oq50.quantile(), qs[2], atol = 1e-1)
    assert np.isclose(oq75.quantile(), qs[3], atol = 1e-1)
    assert np.isclose(oq95.quantile(), qs[4], atol = 1e-1)
