import numpy as np

class OnlineMean():
    def __init__(self, d):
        self.N = 0
        self.m = np.zeros(d)

    def update(self, x):
        self.N += 1
        w = 1 / self.N
        d = x - self.m
        self.m += d * w

    def mean(self):
        return self.m
