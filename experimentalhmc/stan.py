from .dual_average import DualAverage
from .ehmc_cpp import _rand_normal, _rand_uniform, _stan_transition
from .initialize_draws import initialize_draws
from .windowedadaptation import WindowedAdaptation
from .metric_adapter import MetricAdapter
from .rng import RNG
from .step_size_adapter import StepSizeAdapter
from .step_size_initializer import step_size_initializer


from numpy.ctypeslib import ndpointer
from typing import Iterator

import ctypes

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
double_array = ndpointer(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))

class Stan(RNG):
    def __init__(self,
                 dims,
                 log_density_gradient,
                 seed = np.random.default_rng().integers(0, np.iinfo(np.int64).max),
                 warmup = 1000,
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

        C_LDG = ctypes.CFUNCTYPE(ctypes.c_double,
                                 ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double))
        self._ldg = C_LDG(log_density_gradient)

        self._seed = seed
        super().__init__(self._seed)

        self._iteration = 0
        self._warmup = warmup
        self._dims = dims
        self._delta = delta
        self._metric = metric or np.ones(self._dims)
        self._max_delta_H = ctypes.c_double(float(max_delta_H))
        self._max_tree_depth = ctypes.c_int(int(max_tree_depth))

        if initial_draw is not None:
            self._draw = initial_draw
        else:
            self._draw = initialize_draws(self._xoshiro_seed, self._dims, self._ldg,  **kwargs)

        self._step_size = ctypes.pointer(ctypes.c_double(float(step_size)))
        if initialize_step_size:
            self._step_size.contents.value = step_size_initializer(self._draw,
                                                                   self._dims,
                                                                   self._step_size.contents.value,
                                                                   self._metric,
                                                                   self._xoshiro_seed,
                                                                   self._ldg)

        self._schedule = WindowedAdaptation(self._warmup, **kwargs)
        self._metric_adapter = MetricAdapter(self._dims)
        self._step_size_adapter = StepSizeAdapter(self._delta, **kwargs)

        self._adapt_stat = ctypes.pointer(ctypes.c_double(0.0))
        self._divergent = ctypes.pointer(ctypes.c_bool(False))
        self._n_leapfrog = ctypes.pointer(ctypes.c_int(0))
        self._tree_depth = ctypes.pointer(ctypes.c_int(0))
        self._energy = ctypes.pointer(ctypes.c_double(0.0))

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> FloatArray:
        return self.sample()

    def warmup(self) -> int:
        return self._warmup

    def sample(self) -> FloatArray:
        _stan_transition(self._draw,
                         self._ldg,
                         self._xoshiro_seed,
                         self._adapt_stat,
                         self._divergent,
                         self._n_leapfrog,
                         self._tree_depth,
                         self._energy,
                         self._dims,
                         self._metric,
                         self._step_size.contents.value,
                         self._max_delta_H,
                         self._max_tree_depth)
        if self._iteration <= self._warmup:
            self._step_size_adapter.update(self._adapt_stat.contents.value)
            self._step_size.contents.value = self._step_size_adapter.optimum(smooth = False)

            update_metric = self._schedule.firstwindow() <= self._iteration
            update_metric &= self._iteration <= self._schedule.lastwindow()
            if update_metric:
                self._metric_adapter.update(self._draw)

            if self._iteration == self._schedule.closewindow():
                self._step_size.contents.value = step_size_initializer(self._draw,
                                                                       self._dims,
                                                                       self._step_size.contents.value,
                                                                       self._metric,
                                                                       self._xoshiro_seed,
                                                                       self._ldg)
                self._step_size_adapter.reset(mu = np.log(10 * self._step_size.contents.value))

                self._metric = self._metric_adapter.metric()
                self._metric_adapter.reset()

                self._schedule.calculate_next_window()

        else:
            pass
            self._step_size.contents.value = self._step_size_adapter.optimum(smooth = True)

        self._iteration += 1

        return self._draw

    def diagnostics(self):
        """Various properties that might change each iteration"""
        return {
            "iteration": self._iteration,
            "metric": self._metric,
            "adapt_stat": self._adapt_stat.contents.value,
            "divergent": self._divergent.contents.value,
            "n_leapfrog": self._n_leapfrog.contents.value,
            "tree_depth": self._tree_depth.contents.value,
            "energy": self._energy.contents.value,
            "step_size": self._step_size.contents.value,
        }
