import numpy as np

class DualAverage():
    def __init__(self, chains: int,
                 delta: float = 0.85,
                 mu: float = np.log(10.0),
                 gamma: float = 0.05,
                 kappa: float = 0.75,
                 t0: int = 10):
        self._sbar = np.zeros(chains)
        self._x = np.zeros(chains)
        self._xbar = np.zeros(chains)
        self._mu = np.full(chains, mu)
        self._delta = np.full(chains, delta)
        self._gamma = np.full(chains, gamma)
        self._t0 = np.full(chains, t0)
        self._kappa = np.full(chains, kappa)
        self._counter = 0.0

    def update(self, adapt_stat: float):
        self._counter += 1.0
        eta = 1 / (self._counter + self._t0)
        self._sbar = (1.0 - eta) * self._sbar + eta * (self._delta - adapt_stat)
        self._x = self._mu - self._sbar * np.sqrt(self._counter) / self._gamma
        xeta = self._counter ** -self._kappa
        self._xbar = (1.0 - xeta) * self._xbar + xeta * self._x

    def optimum(self, smooth = True):
        if smooth:
            return np.exp(self._xbar)
        return np.exp(self._x)

    def reset(self, mu = np.log(10.0)):
        self._mu[:] = mu
        self._sbar[:] = 0.0
        self._xbar[:] = 0.0
        self._counter = 0.0
