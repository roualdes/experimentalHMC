from .ehmc import _stan_transition
from .leapfrog import leapfrog
from .initialize_draws import initialize_draws
from .windowedadaptation import WindowedAdaptation
from .metric_adapter import MetricOnlineMeanVar
from .rand import choice_rand, normal_rand, uniform_rand
from .rng import RNG
from .step_size_adapter import StepSizeAdapter
from .step_size_initializer import step_size_initializer
from .power_two import power_two

from typing import Iterator

import ctypes

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

    def _u_turn(self, pb, pf, pt):
        normb = np.linalg.norm(pb)
        normf = np.linalg.norm(pf)
        normt = np.linalg.norm(pt)
        f = np.dot(pf, pt) / (normf * normt)
        b = np.dot(pb, pt) / (normb * normt)
        return f <= self._cut_off or b <= self._cut_off

    def _transition(self, position, rng, step_size, metric) -> tuple[float, bool, float]:
        alpha = 0.0
        n_leapfrog = 0
        H = 0.0
        divergent = False

        u_turned = False

        momentum = normal_rand(rng.key(), self._dims)
        position_new = position.copy()
        momentum_new = momentum.copy()
        gradient = np.empty_like(position_new)

        ld = self._log_density_gradient(position_new, gradient)
        H0 = -ld + 0.5 * np.dot(momentum_new, momentum_new)
        direction = choice_rand(rng.key(), [-1, 1])

        # one leapfrog backwards
        position_backwards = position.copy()
        momentum_backwards = momentum.copy()
        gradient_backwards = gradient.copy()
        leapfrog(position_backwards,
                 momentum_backwards,
                 -1 * direction * step_size * np.sqrt(metric),
                 1,
                 gradient_backwards,
                 self._log_density_gradient)

        # momentum for u-turn checks
        momentum_subtree = momentum.copy()
        momentum_total = momentum_new.copy()

        # memory efficient multinomial sampling along trajectory
        position_prime = position.copy()
        max_weight = -np.inf

        while n_leapfrog < self._max_leapfrogs:
            n_leapfrog += 1
            ld = leapfrog(position_new,
                          momentum_new,
                          direction * step_size * np.sqrt(metric),
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

            if exp_delta_H > max_weight:
                position_prime = position_new.copy()
                max_weight = exp_delta_H

            momentum_total += momentum_new
            # TODO count the various u_turns
            u_turn_m_mn = self._u_turn(momentum, momentum_new, momentum_total)
            u_turn_m_ms = self._u_turn(momentum, momentum_subtree, momentum_total)
            u_turn_ms_mn = self._u_turn(momentum_subtree, momentum_new, momentum_total)
            u_turn_mb_mn = self._u_turn(momentum_backwards, momentum_new, momentum_total)
            u_turn_mb_ms = self._u_turn(momentum_backwards, momentum_subtree, momentum_total)

            u_turned = u_turn_m_mn | u_turn_m_ms | u_turn_ms_mn | u_turn_mb_mn | u_turn_mb_ms

            if power_two(n_leapfrog):
                momentum_subtree = momentum_new.copy()

            if u_turned:
                break

        if not divergent:
            position[:] = position_prime.copy()

        adapt_stat = alpha / n_leapfrog
        return adapt_stat, divergent, H


    def sample(self) -> FloatArray:
        for chain in range(self._chains):
            self._adapt_stat[chain], self._divergent[chain], self._energy[chain] = self._transition(self._draws[chain], self._rngs[chain], self._step_size[chain], self._metric[chain])

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
