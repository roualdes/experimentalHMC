from .onlinequantile import OnlineQuantile

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMAD():
    def __init__(self, D: int):
        self._N = 0
        self._D = D
        self._innermed = [OnlineQuantile(0.5) for d in range(self._D)]
        self._outermed = [OnlineQuantile(0.5) for d in range(self._D)]

    def update(self, x: FloatArray):
        for d in range(self._D):
            self._innermed[d].update(x[d])
            median = self._innermed[d].quantile()
            self._outermed[d].update(np.abs(x[d] - median))

    def mad(self):
        return np.asarray([self._outermed[d].quantile() for d in range(self._D)])
