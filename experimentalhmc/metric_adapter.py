from .onlinemeanvar import OnlineMeanVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class MetricAdapter():
    def __init__(self, dims: int = 1):
        self._onlinemoments = OnlineMeanVar(dims)

    def update(self, x: FloatArray):
        self._onlinemoments.update(x)

    def reset(self):
        self._onlinemoments.reset()

    def metric(self):
        N = self._onlinemoments.N()
        if N > 2:
            w = N / (N + 5)
            return w * self._onlinemoments.var() + (1 - w) * 1e-3
        else:
            return np.ones_like(np._onlinemoments.var())

    def mean(self):
        return self._onlinemoments.mean()
