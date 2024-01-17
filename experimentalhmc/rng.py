from .ehmc import _set_seed

import ctypes

import numpy as np

class RNG():
    def __init__(self, seed: int = None):
        if seed is None:
            self._seed = np.random.default_rng().integers(0, np.iinfo(np.int64).max)
        else:
            self._seed = seed

        self._seed = ctypes.c_uint64(self._seed)
        self._xoshiro_seed = np.empty(4, dtype = np.uint64)
        _set_seed(self._seed, self._xoshiro_seed)

    def key(self):
        return self._xoshiro_seed
