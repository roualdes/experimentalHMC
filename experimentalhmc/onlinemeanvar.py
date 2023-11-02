import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMeanVar():
    def __init__(self, chains: int = 1, dims: int = 1):
        self._N = 0
        self._m = np.zeros(shape = (chains, dims))
        self._v = np.zeros(shape = (chains, dims))

    def update(self, x: FloatArray):
        self._N += 1
        w = 1 / self._N
        d = x - self._m
        self._m += d * w
        self._v += -self._v * w + w * (1 - w) * d ** 2

    def mean(self):
        return self._m

    def var(self):
        return self._v

    def std(self):
        return np.sqrt(self._v)
