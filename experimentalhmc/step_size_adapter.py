from .dual_average import DualAverage

class StepSizeAdapter():
    def __init__(self, delta: float = 0.85, **kwargs):
        self._delta = delta
        self._dual_average = DualAverage(**kwargs)

    def update(self, adapt_stat: float):
        self._dual_average.update(self._delta - adapt_stat)

    def reset(self, mu = np.log(10.0)):
        self._dual_average.reset(mu = mu)

    def optimum(self, smooth = True) -> float:
        return self._dual_average.optimum(smooth = smooth)
