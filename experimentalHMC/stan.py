from .dualaverage import DualAverage
from .windowedaptation import WindowedAdaptation
from .metric_adapter import MetricAdapter

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class Stan():
    def __init__(self,
                 chains,
                 dims,
                 iterations = 1000,
                 warmup = None,
                 metric = None,
                 step_size = None,
                 max_tree_depth = 10,
                 max_delta_H = 1000,
                 **kwargs
                 ):

        self._chains = chains
        self._dims = dims
        self._iterations = iterations
        self._warmup = warmup
        self._draws = np.zeros(shape = (self.iterations, self._chains, self._dims))
        self._metric = metric or np.ones(shape = (self._chains, self._dims))
        self._metric_adapter = MetricAdapter(self._chains, self._dims)
        self._step_size = step_size or np.ones(self._chains)
        self._dual_average = DualAverage(self._chains, **kwargs)
        self._schedule = WindowedAdaptation(self._warmup, **kwargs)

        self._draws[0] = 1.0 # TODO initialize draws
        self._step_size = 1.0 # TODO initialize step_size

    def sample() -> FloatArray:
        return self._draws

    def set_initial_draw(x: FloatArray) -> None:
        self._draws[0] = x

    def set_step_size(x: FloatArray) -> None:
        self._step_size = x

    def draws() -> FloatArray:
        return self._draws
