from .ehmc import _stan_transition
from .initialize_draws import initialize_draws
from .windowedadaptation import WindowedAdaptation
from .metric_adapter import MetricOnlineMeanVar
from .rng import RNG
from .step_size_adapter import StepSizeAdapter
from .step_size_initializer import step_size_initializer

from typing import Iterator

import ctypes

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class Stan():
    def __init__(self,
                 dims,
                 log_density_gradient,
                 chains = 4,
                 warmup = 1000,
                 seed = None,
                 metric = None,
                 step_size = 1.0,
                 delta = 0.8,
                 initialize_step_size = True,
                 initial_draw = None,
                 initial_draw_radius = 2,
                 max_tree_depth = 10,
                 max_delta_H = 1000.0,
                 **kwargs
                 ):

        self._log_density_gradient = log_density_gradient
        C_LDG = ctypes.CFUNCTYPE(ctypes.c_double,
                                 ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double)
                                 )
        self._ldg = C_LDG(self._log_density_gradient)

        self._chains = chains
        self._dims = dims
        self._rngs = [RNG(seed) for _ in range(self._chains)]
        self._iteration = 0
        self._warmup = warmup
        self._delta = delta
        self._metric = metric or np.ones(shape = (self._chains, self._dims))
        self._max_delta_H = ctypes.c_double(float(max_delta_H))
        self._max_tree_depth = ctypes.c_int(int(max_tree_depth))

        self._draws = np.zeros(shape = (self._chains, self._dims))
        if initial_draw is not None:
            self._draws[:] = initial_draw
        else:
            for chain in range(self._chains):
                self._draws[chain] = initialize_draws(self._rngs[chain].key(),
                                                      self._dims,
                                                      self._log_density_gradient,
                                                      **kwargs)

        self._step_size = [ctypes.c_double(step_size) for _ in range(self._chains)]
        if initialize_step_size:
            for chain in range(self._chains):
                self._step_size[chain].value = step_size_initializer(self._draws[chain],
                                                                     self._dims,
                                                                     self._step_size[chain].value,
                                                                     self._metric[chain],
                                                                     self._rngs[chain].key(),
                                                                     self._log_density_gradient)

        self._schedule = WindowedAdaptation(self._warmup, **kwargs)
        self._metric_adapter = [MetricOnlineMeanVar(self._dims) for _ in range(self._chains)]
        self._step_size_adapter = [StepSizeAdapter(self._delta, **kwargs) for _ in range(self._chains)]

        self._adapt_stat = [ctypes.c_double(0.0) for _ in range(self._chains)]
        self._energy = [ctypes.c_double(0.0) for _ in range(self._chains)]
        self._divergent = [ctypes.c_bool(False) for _ in range(self._chains)]
        self._n_leapfrog = [ctypes.c_int(0) for _ in range(self._chains)]
        self._tree_depth = [ctypes.c_int(0) for _ in range(self._chains)]

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> FloatArray:
        return self.sample()

    def warmup(self) -> int:
        return self._warmup

    def dims(self) -> int:
        return self._dims

    def chains(self) -> int:
        return self._chains

    def sample(self) -> FloatArray:
        for chain in range(self._chains):
            _stan_transition(self._draws[chain],
                             self._ldg,
                             self._rngs[chain].key(),
                             ctypes.byref(self._adapt_stat[chain]),
                             ctypes.byref(self._divergent[chain]),
                             ctypes.byref(self._n_leapfrog[chain]),
                             ctypes.byref(self._tree_depth[chain]),
                             ctypes.byref(self._energy[chain]),
                             self._dims,
                             self._metric[chain],
                             self._step_size[chain],
                             self._max_delta_H,
                             self._max_tree_depth)

            if self._iteration <= self._warmup:
                self._step_size_adapter[chain].update(self._adapt_stat[chain].value)
                self._step_size[chain].value = self._step_size_adapter[chain].optimum(smooth = False)

                update_metric = self._schedule.firstwindow() <= self._iteration
                update_metric &= self._iteration <= self._schedule.lastwindow()
                if update_metric:
                    self._metric_adapter[chain].update(self._draws[chain])

                if self._iteration == self._schedule.closewindow():
                    self._step_size[chain].value = step_size_initializer(self._draws[chain],
                                                                         self._dims,
                                                                         self._step_size[chain].value,
                                                                         self._metric[chain],
                                                                         self._rngs[chain].key(),
                                                                         self._log_density_gradient)
                    self._step_size_adapter[chain].reset(mu = np.log(10 * self._step_size[chain].value))

                    self._metric[chain] = self._metric_adapter[chain].metric()
                    self._metric_adapter[chain].reset()

                    self._schedule.calculate_next_window()
            else:
                self._step_size[chain].value = self._step_size_adapter[chain].optimum(smooth = True)

        self._iteration += 1
        return self._draws

    def _array_from_list_ctypes(self, x, dtype):
        C = self._chains
        return np.fromiter((x[c].value for c in range(C)), dtype, count = C)

    def diagnostics(self):
        """Various properties that might change each iteration"""
        C = self._chains
        return {
            "iteration": self._iteration,
            "metric": self._metric,
            "adapt_stat": self._array_from_list_ctypes(self._adapt_stat, np.float64),
            "divergent": self._array_from_list_ctypes(self._divergent, np.bool_),
            "n_leapfrog": self._array_from_list_ctypes(self._n_leapfrog, np.intc),
            "tree_depth": self._array_from_list_ctypes(self._tree_depth, np.intc),
            "energy": self._array_from_list_ctypes(self._energy, np.float64),
            "step_size": self._array_from_list_ctypes(self._step_size, np.float64),
        }
