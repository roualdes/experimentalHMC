from .onlinequantile import OnlineQuantile

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMAD():
    def __init__(self, D: int):
        self.N = 0
        self.D = D
        self.innermed = [OnlineQuantile(0.5) for d in range(self.D)]
        self.outermed = [OnlineQuantile(0.5) for d in range(self.D)]

    def update(self, x: FloatArray):
        for d in range(self.D):
            self.innermed[d].update(x[d])
            median = self.innermed[d].quantile()
            self.outermed[d].update(np.abs(x[d] - median))

    def mad(self):
        return [self.outermed[d].quantile() for d in range(self.D)]
