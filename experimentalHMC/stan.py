from .dualaverage import DualAverage
from .

import numpy as np



class Stan():
    def __init__(self, dims,
                 metric = None,
                 stepsize = None,
                 maxtreedepth = 10,
                 maxdeltaH = 1000,
                 iterations = 1000,
                 warmup = None,
                 mu = np.log(10),
                 gamma = 0.05,
                 t0 = 10,
                 kappa = 0.75,
                 ):

        self._dims = dims
        self._metric = metric or np.ones(self._dims)
