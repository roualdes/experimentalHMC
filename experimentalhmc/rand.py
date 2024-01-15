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
