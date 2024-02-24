from .ehmc import rand_uniform_broadcast, rand_uniform, rand_normal_broadcast, rand_normal

import bisect

import numpy as np

def uniform_rand(xoshiro_key, N: int = None):
    "Generate N uniform(0, 1) random numbers.  If N is None, N = 1."
    if N is not None:
        assert type(N) == int
        u = np.empty(N)
        rand_uniform_broadcast(xoshiro_key, N, u)
        return u
    else:
        return rand_uniform(xoshiro_key)


def choice_rand(xoshiro_key, x, p = None, N: int = None):
    """Random choose N elements from x with probabilities p.  If p is
    None then p = 1 / np.size(x).
    """
    if p is None:
        p = np.ones_like(x) / np.size(x)

    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    us = uniform_rand(xoshiro_key, N)

    if N is not None:
        idx = np.empty_like(us)
        for (n, u) in enumerate(us):
            idx[n] = bisect.bisect_right(cdf, u)
    else:
        idx = bisect.bisect_right(cdf, us)

    return x[idx]


def normal_rand(xoshiro_key, N: int = None):
    "Generate N Normal(0, 1) random numbers.  If N is None, N = 1."
    if N is not None:
        assert type(N) == int
        z = np.empty(N)
        rand_normal_broadcast(xoshiro_key, N, z)
        return z
    else:
        return rand_normal(xoshiro_key)

def _rand_one_binomial(xoshiro_key, K, p):
    if K == 0 or p == 0.0:
        return 0

    if p == 1.0:
        return K

    x = -1
    S = 0
    while S <= K:
        g = np.ceil(np.log(uniform_rand(xoshiro_key)) / np.log1p(-p))
        S += g
        x += 1
    return x

def binomial_rand(xoshiro_key, K, p, N: int = None):
    "Generate N Binomial(K, p) random numbers.  If N is None, N = 1."
    if N is not None:
        assert type(N) == int
        x = -np.ones(N)
        for s in range(size):
            x[s] = _rand_one_binomial(xoshiro_key, K, p)
        return x
    else:
        return _rand_one_binomial(xoshiro_key, K, p)
