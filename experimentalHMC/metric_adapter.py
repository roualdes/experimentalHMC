from .onlinemeanvar import OnlineMeanVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class MetricAdapter():
    def __init__(self, chains: int = 1, dims: int = 1):
        self._chains = chains
        self._dims = dims
        self._onlinemoments = OnlineMeanVar(self._chains, self._dims)
        self._metric = np.ones(shape = (self._chains, self._dims))

    def update(self, x: FloatArray):
        self._onlinemoments.update(x)
        N = self._onlinemoments._N
        w = N / (N + 5)
        self._metric = w * self._onlinemoments.var() + (1 - w) * np.ones(self._dims)

    def metric(self):
        return self._metric

    def mean(self):
        return self._onlinemoments.mean()
