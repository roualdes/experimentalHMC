import pytest

import experimentalhmc as ehmc

import numpy as np

def test_uniform_rng():
    N = 1_000
    R = 500

    RNG = np.random.default_rng()
    seeds = RNG.integers(0, np.iinfo(np.int64).max, size = R)
    rngs = [ehmc.RNG(seeds[r]) for r in range(R)]
    U_means = np.empty(R)
    U_stds = np.empty(R)
    for r in range(R):
        u = ehmc.uniform_rand(rngs[r].key(), N)
        U_means[r] = np.mean(u)
        U_stds[r] = np.std(u, ddof = 1)

    assert np.abs(np.mean(U_means) - 0.5) < 4 * np.std(U_means, ddof = 1)
    B = U_means > 0.5
    assert np.abs(np.mean(B) - 0.5) < 4 * np.std(B, ddof = 1)
    assert np.isclose(np.mean(U_stds), np.sqrt(1 / 12), atol = 0.1)
    # TODO test reproducibility


def test_normal_rng():
    N = 1_000
    R = 500

    RNG = np.random.default_rng()
    seeds = RNG.integers(0, np.iinfo(np.int64).max, size = R)
    rngs = [ehmc.RNG(seeds[r]) for r in range(R)]
    N_means = np.empty(R)
    N_stds = np.empty(R)
    for r in range(R):
        z = ehmc.normal_rand(rngs[r].key(), N)
        N_means[r] = np.mean(z)
        N_stds[r] = np.std(z, ddof = 1)

    assert np.abs(np.mean(N_means)) < 4 * np.std(N_means, ddof = 1)
    B = N_means > 0.0
    assert np.abs(np.mean(B) - 0.5) < 4 * np.std(B, ddof = 1)
    assert np.isclose(np.mean(N_stds), 1.0, atol = 0.1)


def test_choice_rand():
    # TODO test choice_rand()
    pass
