from .ehmc import _stan_transition
from .leapfrog import leapfrog
from .initialize_draws import initialize_draws
from .windowedadaptation import WindowedAdaptation
from .metric_adapter import MetricOnlineMeanVar
from .rand import binomial_rand, choice_rand, normal_rand, uniform_rand
from .rng import RNG
from .step_size_adapter import StepSizeAdapter
from .step_size_initializer import step_size_initializer
from .power_two import power_two

from typing import Iterator

import ctypes
import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class SBHMC():
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
                 max_leapfrogs = 1_000,
                 max_delta_H = 1_000.0,
                 cut_off = 0.0,
                 p = 0.75,
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
        self._max_leapfrogs = max_leapfrogs
        self._max_delta_H = max_delta_H
        self._cut_off = cut_off
        self._p = p

        self._draws = np.zeros(shape = (self._chains, self._dims))
        if initial_draw is not None:
            self._draws[:] = initial_draw
        else:
            for chain in range(self._chains):
                self._draws[chain] = initialize_draws(self._rngs[chain].key(),
                                                      self._dims,
                                                      self._log_density_gradient,
                                                      **kwargs)

        self._step_size = [step_size for _ in range(self._chains)]
        if initialize_step_size:
            for chain in range(self._chains):
                self._step_size[chain] = step_size_initializer(self._draws[chain],
                                                               self._dims,
                                                               self._step_size[chain],
                                                               self._metric[chain],
                                                               self._rngs[chain].key(),
                                                               self._log_density_gradient)

        self._schedule = WindowedAdaptation(self._warmup, **kwargs)
        self._metric_adapter = [MetricOnlineMeanVar(self._dims) for _ in range(self._chains)]
        self._step_size_adapter = [StepSizeAdapter(self._delta, **kwargs) for _ in range(self._chains)]

        self._adapt_stat = [0.0 for _ in range(self._chains)]
        self._energy = [0.0 for _ in range(self._chains)]
        self._divergent = [False for _ in range(self._chains)]
        self._n_leapfrog = [0 for _ in range(self._chains)]
        self._tree_depth = [0 for _ in range(self._chains)]

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

    def _u_turn(self, p, pt):
        return np.dot(p, pt) <= self._cut_off

    def _u_turn_steps(self, position, momentum, rng, step_size) -> tuple[bool, int, float]:
        alpha = 0.0
        n_leapfrog = 0
        H = 0.0
        divergent = False
        dist = 0.0

        position_new = position.copy()
        momentum_new = momentum.copy()
        gradient = np.empty_like(position_new)

        ld = self._log_density_gradient(position_new, gradient)
        H0 = -ld + 0.5 * np.dot(momentum_new, momentum_new)

        while n_leapfrog < self._max_leapfrogs:
            n_leapfrog += 1
            ld = leapfrog(position_new,
                          momentum_new,
                          step_size,
                          1,
                          gradient,
                          self._log_density_gradient)

            H = -ld + 0.5 * np.dot(momentum_new, momentum_new)
            if np.isnan(H):
                H = np.inf

            if H - H0 > self._max_delta_H:
                divergent = True

            delta_H = H0 - H
            exp_delta_H = np.exp(delta_H)
            if delta_H > 0.0:
                alpha += 1.0
            else:
                alpha += exp_delta_H

            if divergent:
                break

            new_dist = np.sum((position - position_new) ** 2)
            if new_dist <= dist:
                u_turn = True
            else:
                dist = new_dist

            if u_turn:
                break

        adapt_stat = alpha / n_leapfrog
        return divergent, n_leapfrog, adapt_stat, H0

    def _transition(self, position, momentum, rng, step_size, metric) -> tuple[float, bool, float]:
        direction = choice_rand(rng, [-1, 1])
        epsilon = direction * step_size * np.sqrt(metric)
        d, N, a, H0 = self._u_turn_steps(position.copy(), momentum.copy(), rng, epsilon)

        divergent = d
        if divergent:
            return a, d, np.inf

        n_leapfrog = N
        adapt_stat = a

        L = binomial_rand(rng, n_leapfrog, self._p)
        position_new = position.copy()
        momentum_new = momentum.copy()
        gradient = np.empty_like(momentum_new)

        ld = leapfrog(position_new,
                      momentum_new,
                      epsilon,
                      L,
                      gradient,
                      self._log_density_gradient)
        H1 = -ld + 0.5 * np.dot(momentum_new, momentum_new)

        _, Nb, _, _ = self._u_turn_steps(position_new, momentum_new, rng, -epsilon)

        r = H1 + np.log(N) - H0 - np.log(Nb + 1)
        if uniform_rand(rng) < np.minimum(1.0, np.exp(r)):
            position[:] = position_new
            energy = H1
        else:
            energy = H0
        return adapt_stat, divergent, energy

    def sample(self) -> FloatArray:
        for chain in range(self._chains):
            momentum = normal_rand(self._rngs[chain].key(), self._dims)
            a, d, e = self._transition(self._draws[chain],
                                       momentum,
                                       self._rngs[chain].key(),
                                       self._step_size[chain],
                                       self._metric[chain])

            self._adapt_stat[chain] = a
            self._divergent[chain] = d
            self._energy[chain] = e

            if self._iteration <= self._warmup:
                self._step_size_adapter[chain].update(self._adapt_stat[chain])
                self._step_size[chain] = self._step_size_adapter[chain].optimum(smooth = False)

                update_metric = self._schedule.firstwindow() <= self._iteration
                update_metric &= self._iteration <= self._schedule.lastwindow()
                if update_metric:
                    self._metric_adapter[chain].update(self._draws[chain])

                if self._iteration == self._schedule.closewindow():
                    self._step_size[chain] = step_size_initializer(self._draws[chain],
                                                                   self._dims,
                                                                   self._step_size[chain],
                                                                   self._metric[chain],
                                                                   self._rngs[chain].key(),
                                                                   self._log_density_gradient)
                    self._step_size_adapter[chain].reset(mu = np.log(10 * self._step_size[chain]))

                    self._metric[chain] = self._metric_adapter[chain].metric()
                    self._metric_adapter[chain].reset()

                    self._schedule.calculate_next_window()
            else:
                self._step_size[chain] = self._step_size_adapter[chain].optimum(smooth = True)

        self._iteration += 1
        return self._draws

    def diagnostics(self):
        """Various properties that might change each iteration"""
        C = self._chains
        return {
            "iteration": self._iteration,
            "metric": self._metric,
            "adapt_stat": np.array(self._adapt_stat),
            "divergent": np.array(self._divergent),
            "n_leapfrog": np.array(self._n_leapfrog),
            "tree_depth": np.array(self._tree_depth),
            "energy": np.array(self._energy),
            "step_size": np.array(self._step_size),
        }
