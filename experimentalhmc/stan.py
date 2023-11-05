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
                 seed = np.random.default_rng().integers(0, np.iinfo(np.int64).max),
                 iterations = 1000,
                 warmup = None,
                 metric = None,
                 step_size = None,
                 initial_draw = None,
                 max_tree_depth = 10,
                 max_delta_H = 1000,
                 **kwargs
                 ):

        self._chains = chains
        self._dims = dims
        self._seed = seed
        self._iterations = iterations
        self._warmup = warmup
        self._metric = metric or np.ones(shape = (self._chains, self._dims))
        self._step_size = step_size or np.ones(self._chains)

        self._schedule = WindowedAdaptation(self._warmup, **kwargs)
        self._metric_adapter = MetricAdapter(self._chains, self._dims)
        self._dual_average = DualAverage(self._chains, **kwargs)

        self._draws = np.zeros(shape = (self._iterations, self._chains, self._dims))
        # TODO initialize xoshiro then gen U(-2, 2)
        self._draws[0] = initial_draw or 0


    def sample() -> FloatArray:
        return self._draws

    def draws() -> FloatArray:
        return self._draws
