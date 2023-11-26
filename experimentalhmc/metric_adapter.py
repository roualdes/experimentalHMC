from .onlinemeanvar import OnlineMeanVar
from .onlinemad import OnlineMAD

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class MetricAdapter():
    def update(self, x: FloatArray):
        self.estimator.update(x)

    def reset(self):
        self.estimator.reset()

    def metric(self):
        N = self.estimator.N()
        V = self.estimator.scale() ** 2
        if N > 2:
            w = N / (N + 5)
            return w * V + (1 - w) * 1e-3
        else:
            return np.ones_like(V)

    def location(self):
        return self.estimator.location()

class MetricOnlineMeanVar(MetricAdapter):
    def __init__(self, dims: int = 1):
        self.estimator = OnlineMeanVar(dims)

class MetricOnlineMAD(MetricAdapter):
    def __init__(self, dims: int = 1):
        self.estimator = OnlineMAD(dims)
