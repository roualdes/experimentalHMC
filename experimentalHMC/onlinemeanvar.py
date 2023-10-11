import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMeanVar():
    def __init__(self, d: int = 1):
        self.N = 0
        self.m = np.zeros(d)
        self.v = np.zeros(d)

    def update(self, x: FloatArray):
        self.N += 1
        w = 1 / self.N
        d = x - self.m
        self.m += d * w
        self.v += -self.v * w + w * (1 - w) * d ** 2

    def mean(self):
        return self.m

    def var(self):
        return self.v

    def std(self):
        return np.sqrt(self.v)
