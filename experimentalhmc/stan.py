from .dual_average import DualAverage
from .ehmc_cpp import _rand_normal, _rand_uniform
from .initialize_draws import initialize_draws
from .windowedaptation import WindowedAdaptation
from .metric_adapter import MetricAdapter
from .rng import RNG

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
                 step_size = None,
                 delta = 0.85,
                 initialize_step_size = True,
                 initial_draw = None,
                 initial_draw_radius = 2,
                 max_tree_depth = 10,
                 max_delta_H = 1000,
                 **kwargs
                 ):

        C_LDG = ctypes.CFUNCTYPE(ctypes.c_double, double_array, double_array)
        self._ldg = C_LDG(log_density_gradient)

        self._seed = seed
        super().__init__(self._seed)

        self._dims = dims
        self._iteration = 0
        self._warmup = warmup
        self._metric = metric or np.ones(self._dims)

        print("before initial draw")
        if initial_draw:
            self._draw = initial_draw
        else:
            self._draw = initialize_draws(self._xoshiro_seed, self._dims, self._ldg,  **kwargs)

        print("after initial draw")
        self._step_size = step_size or 1.0
        # TODO follow this through: initialization, updating, setting, reseting
        self._step_size = ctypes.pointer(ctypes.c_double(self._step_size))
        if initialize_step_size:
            # TODO write StepSizeInitializer
            # self._step_size.contents.value = StepSizeInitializer(self._step_size.contents.value, self._xoshiro_seed, self._ldg)
            pass

        self._schedule = WindowedAdaptation(self._warmup, **kwargs)
        self._metric_adapter = MetricAdapter(self._dims)
        self._step_size_adapter = StepSizeAdapter(delta, **kwargs)

        self._adapt_stat = ctypes.pointer(ctypes.c_double(0.0))
        self._divergent = ctypes.pointer(ctypes.c_bool(False))
        self._n_leapfrog = ctypes.pointer(ctypes.c_int(0))
        self._tree_depth = ctypes.pointer(ctypes.c_int(0))
        self._energy = ctypes.pointer(ctypes.c_double(0.0))

        self._max_tree_depth = ctypes.c_double(max_tree_depth)
        self._max_delta_H = ctypes.c_double(max_delta_H)
        print("end initialization")

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> FloatArray:
        return self.sample()

    def sample(self) -> FloatArray:
        self._iteration += 1
        # TODO transition
        stan_transition(self._draw,
                        self._ldg,
                        self._xoshiro_seed,
                        self._accept_stat,
                        self._divergent,
                        self._n_leapfrog,
                        self._tree_depth,
                        self._energy,
                        self._dims,
                        self._metric,
                        self._step_size,
                        self._max_delta_H,
                        self._max_tree_depth)
        if self._iteration <= self._warmup:
            self._step_size_adapter.update(self._adapt_stat)
            self._step_size = self._step_size_adapter.optimum(smooth = False)

            update_metric = self._schedule.firstwindow() <= self._iteration
            update_metric &= self._iteration <= self._schedule.lastwindow()
            if update_metric:
                self._metric_adapter.update(self._draw)

            if self._iteration == self._schedule.closewindow():
                # initialize_stepsize
                self._step_size.contents.value = self._step_size_adapter.optimum(smooth = False)
                self._step_size_adapter.reset()

                self._metric = self._metric_adapter.metric()
                self._metric_adapter.reset()

                self._schedule.calculate_next_window()

        else:
            self._step_size = self._step_size_adapter.optimum(smooth = True)

    def diagnostics(self):
        """Various properties that might change each iteration"""
        return {
            "iteration": self._iteration,
            "adapt_stat": self._adapt_stat,
            "divergent": self._divergent,
            "n_leapfrog": self._n_leapfrog,
            "tree_depth": self._tree_depth,
            "energy": self._energy,
            "step_size": self._step_size,
            "metric": self._metric,
        }
