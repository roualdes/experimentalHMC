from .dual_average import DualAverage

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class StepSizeAdapter():
    def __init__(self, dims: int = 1, delta: float = 0.8, **kwargs):
        self._dims = dims
        self._delta = delta
        self._dual_average = DualAverage(dims, **kwargs)

    def update(self, adapt_stat: FloatArray):
        adapt_stat = 1.0 if adapt_stat > 1.0 else adapt_stat
        self._dual_average.update(self._delta - adapt_stat)

    def reset(self, mu = np.log(10.0)):
        self._dual_average.reset(mu = mu)

    def optimum(self, smooth = True) -> float:
        return self._dual_average.optimum(smooth = smooth)
