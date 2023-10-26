import numpy as np

class DualAverage():
    def __init__(self, C, mu = np.log(10), gamma = 0.05, t0 = 10, kappa = 0.75):
        self._sbar = np.zeros(C)
        self._x = np.zeros(C)
        self._xbar = np.zeros(C)
        self._mu = np.full(mu, C)
        self._gamma = np.full(gamma, C)
        self._t0 = np.full(t0, C)
        self._kappa = np.full(kappa, C)
        self._count = 0

    def update(self, g):
        self._counter += 1
        eta = 1 / (self._counter + self._t0)
        self._sbar = (1 - eta) * self._sbar + eta * g
        self._x = self._mu - self._sbar * np.sqrt(self._counter) / self._gamma
        xeta = self._counter ** -self._kappa
        self._xbar = (1 - xeta) * self._xbar + xeta * self._x

    def optimum(self, smooth = True):
        if smooth:
            return np.exp(self._xbar)
        return np.exp(self._x)

    def reset(self, mu = np.log(10)):
        self._mu[:] = mu
        self._sbar[:] = 0
        self._xbar[:] = 0
        self._counter = 0
