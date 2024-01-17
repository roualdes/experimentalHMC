from .onlinequantile import OnlineQuantile

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMAD():
    def __init__(self, dims: int = 1):
        self._N = 0
        self._dims = dims
        self._innermed = [OnlineQuantile(0.5) for d in range(self._dims)]
        self._outermed = [OnlineQuantile(0.5) for d in range(self._dims)]

    def update(self, x: FloatArray):
        self._N += 1
        for d in range(self._dims):
            self._innermed[d].update(x[d])
            median = self._innermed[d].quantile()
            self._outermed[d].update(np.abs(x[d] - median))

    def reset(self):
        self._N = 0
        self._innermed = [OnlineQuantile(0.5) for d in range(self._D)]
        self._outermed = [OnlineQuantile(0.5) for d in range(self._D)]

    def N(self):
        return self._N

    def location(self):
        return np.array([self._innermed[d].quantile() for d in range(self._dims)])

    def scale(self):
        return np.array([self._outermed[d].quantile() for d in range(self._dims)])
