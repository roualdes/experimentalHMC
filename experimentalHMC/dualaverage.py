import numpy as np

class DualAverage():
    def __init__(self, C, mu = np.log(10), gamma = 0.05, t0 = 10, kappa = 0.75):
        self.sbar = np.zeros(C)
        self.x = np.zeros(C)
        self.xbar = np.zeros(C)
        self.mu = np.full(mu, C)
        self.gamma = np.full(gamma, C)
        self.t0 = np.full(t0, C)
        self.kappa = np.full(kappa, C)
        self.count = 0

    def update(self, g):
        self.counter += 1
        eta = 1 / (self.counter + self.t0)
        self.sbar = (1 - eta) * self.sbar + eta * g
        self.x = self.mu - self.sbar * np.sqrt(self.counter) / self.gamma
        xeta = self.counter ** -self.kappa
        self.xbar = (1 - xeta) * self.xbar + xeta * self.x

    def optimum(self, smooth = True):
        if smooth:
            return np.exp(self.xbar)
        return np.exp(self.x)

    def reset(self, mu = np.log(10)):
        self.mu[:] = mu
        self.sbar[:] = 0
        self.xbar[:] = 0
        self.counter = 0
