from .ehmc import normal_rand, uniform_rand, _set_seed

import ctypes

import numpy as np

class RNG():
    def __init__(self, seed: int):
        self._seed = ctypes.c_uint64(seed)
        self._xoshiro_seed = np.empty(4, dtype = np.uint64)
        _set_seed(self._seed, self._xoshiro_seed)

class NormalRNG(RNG):
    def rand(self, N: int = None):
        return normal_rand(self._xoshiro_seed, N)

class UniformRNG(RNG):
    def rand(self, N: int = None):
        return uniform_rand(self._xoshiro_seed, N)
