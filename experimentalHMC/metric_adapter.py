from .onlinemeanvar import OnlineMeanVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class MetricAdapter():
    def __init__(self, chains: int = 1, dims: int = 1):
        self._onlinemoments = OnlineMeanVar(chains, dims)

    def update(self, x: FloatArray):
        self._onlinemoments.update(x)

    def metric(self):
        N = self._onlinemoments._N
        w = N / (N + 5)
        return w * self._onlinemoments.var() + (1 - w) # * np.ones(shape = (self._chains, self._dims))

    def mean(self):
        return self._onlinemoments.mean()
