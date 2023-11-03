import pytest

import experimentalhmc as ehmc

import numpy as np

def test_uniform_rng():

    N = 1_000
    R = 500

    RNG = np.random.default_rng()
    rngs = RNG.integers(0, np.iinfo(np.int64).max, size = R)
    Us = [ehmc.UniformRNG(rng) for rng in rngs]
    U_means = np.empty(R)
    U_stds = np.empty(R)
    for r in range(R):
        u = Us[r].rand(N)
        U_means[r] = u.mean()
        U_stds[r] = u.std(ddof = 1)

    assert np.abs(U_means.mean() - 0.5) < 4 * U_means.std(ddof = 1)
    B = U_means > 0.5
    assert np.abs(B.mean() - 0.5) < 4 * B.std(ddof = 1)
    assert np.isclose(U_stds.mean(), np.sqrt(1 / 12), atol = 0.1)
    # TODO test reproducibility


def test_normal_rng():

    N = 1_000
    R = 500

    RNG = np.random.default_rng()
    rngs = RNG.integers(0, np.iinfo(np.int64).max, size = R)
    Ns = [ehmc.NormalRNG(rng) for rng in rngs]
    N_means = np.empty(R)
    N_stds = np.empty(R)
    for r in range(R):
        z = Ns[r].rand(N)
        N_means[r] = z.mean()
        N_stds[r] = z.std(ddof = 1)

    assert np.abs(N_means.mean()) < 4 * N_means.std(ddof = 1)
    B = N_means > 0.0
    assert np.abs(B.mean() - 0.5) < 4 * B.std(ddof = 1)
    assert np.isclose(N_stds.mean(), 1.0, atol = 0.1)
