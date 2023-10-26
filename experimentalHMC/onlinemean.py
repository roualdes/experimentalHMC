import numpy as np

class OnlineMean():
    def __init__(self, d):
        self._N = 0
        self._m = np.zeros(d)

    def update(self, x):
        self._N += 1
        w = 1 / self._N
        d = x - self._m
        self._m += d * w

    def mean(self):
        return self._m
