import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMeanVar():
    def __init__(self, dims: int = 1):
        self._N = 0
        self._m = np.zeros(dims)
        self._v = np.zeros(dims)

    def update(self, x: FloatArray):
        self._N += 1
        w = 1 / self._N
        d = x - self._m
        self._m += d * w
        self._v += -self._v * w + w * (1 - w) * d ** 2

    def reset(self):
        self._N = 0
        self._m[:] = 0.0
        self._v[:] = 0.0

    def N(self):
        return self._N

    def location(self):
        return self._m

    def scale(self):
        return np.sqrt(self._v)
