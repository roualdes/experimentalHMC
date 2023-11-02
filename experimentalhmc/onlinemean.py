import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMean():
    def __init__(self, chains: int = 1, dims: int = 1):
        self._N = 0
        self._m = np.zeros(shape = (chains, dims))

    def update(self, x: FloatArray):
        self._N += 1
        w = 1 / self._N
        d = x - self._m
        self._m += d * w

    def mean(self):
        return self._m
