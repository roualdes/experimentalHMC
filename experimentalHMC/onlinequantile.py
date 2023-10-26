class OnlineQuantile():
    def __init__(self, p: float):
        self._p = p
        self._N = 0
        self._N0 = 1
        self._N1 = 2
        self._N2 = 3
        self._N3 = 4
        self._Q0 = 0
        self._Q1 = 0
        self._Q3 = 0
        self._Q4 = 0
        self._q = 0

    def update(self, s: float):
        self._N += 1
        if self._N > 5:
            if s <= self._Q3:
                self._N3 += 1
                if s <= self._q:
                    self._N2 += 1
                    if s <= self._Q1:
                        self._N1 += 1
                        if s <= self._Q0:
                            if s == self._Q0:
                                self._N0 += 1
                            else:
                                self._Q0 = s
                                self._N0 = 1
            elif s > self._Q4:
                self._Q4 = s

            s = (self._N - 1) * self._p * 0.5 + 1 - self._N1
            if s >= 1 and self._N2 - self._N1 > 1:
                h = self._N2 - self._N1
                delta = (self._q - self._Q1 ) / h
                d1 = self._PchipDerivative(self._N1 - self._N0,
                                          (self._Q1 - self._Q0) / (self._N1 - self._N0),
                                          h,
                                          delta)
                d2 = self._PchipDerivative(h,
                                          delta,
                                          self._N3 - self._N2,
                                          (self._Q3 - self._q) / (self._N3 - self._N2))
                self._Q1 += self._HermiteInterpolationOne(h, delta, d1, d2)
                self._N1 += 1

            elif s <= -1 and self._N1 - self._N0 > 1:
                h = self._N1 - self._N0
                delta = (self._Q1 - self._Q0) / h
                d1 = self._PchipDerivativeEnd(h,
                                        delta,
                                        self._N2 - self._N1,
                                        (self._q - self._Q1) / (self._N2 - self._N1))
                d2 = self._PchipDerivative(h,
                                          delta,
                                          self._N2 - self._N1,
                                          (self._q - self._Q1) / (self._N2 - self._N1))
                self._Q1 += self._HermiteInterpolationOne(h, -delta, -d2, -d1)
                self._N1 -= 1

            s = (self._N - 1) * self._p + 1 - self._N2
            if s >= 1 and self._N3 - self._N2 > 1:
                h = self._N3 - self._N2
                delta = (self._Q3 - self._q) / h
                d1 = self._PchipDerivative(self._N2 - self._N1,
                                     (self._q - self._Q1) / (self._N2 - self._N1),
                                     h,
                                     delta)
                d2 = self._PchipDerivative(h,
                                     delta,
                                     self._N - self._N3,
                                     (self._Q4 - self._Q3) / (self._N - self._N3))
                self._q += self._HermiteInterpolationOne(h, delta, d1, d2)
                self._N2 += 1
            elif s < -1 and self._N2 - self._N1 > 1:
                h = self._N2 - self._N1
                delta = (self._q - self._Q1) / h
                d1 = self._PchipDerivative(self._N1 - self._N0,
                                          (self._Q1 - self._Q0) / (self._N1 - self._N0),
                                          h,
                                          delta)
                d2 = self._PchipDerivative(h,
                                          delta,
                                          self._N3 - self._N2,
                                          (self._Q3 - self._q) / (self._N3 - self._N2))
                self._q += self._HermiteInterpolationOne(h, -delta, -d2, -d1)
                self._N2 -= 1

            s = (self._N - 1) * (1 + self._p) * 0.5 + 1 - self._N3
            if s >= 1 and self._N - self._N3 > 1:
                h = self._N - self._N3
                delta = (self._Q4 - self._Q3) / h
                d1 = self._PchipDerivative(self._N3 - self._N2,
                                          (self._Q3 - self._q) / (self._N3 - self._N2),
                                          h,
                                          delta)
                d2 = self._PchipDerivativeEnd(h,
                                             delta,
                                             self._N3 - self._N2,
                                             (self._Q3 - self._q) / (self._N3 - self._N2))
                self._Q3 += self._HermiteInterpolationOne(h, delta, d1, d2)
                self._N3 += 1
            elif s <= -1 and self._N3 - self._N2 > 1:
                h = self._N3 - self._N2
                delta = (self._Q3 - self._q) / h
                d1 = self._PchipDerivative(self._N2 - self._N1,
                                          (self._q - self._Q1) / (self._N2 - self._N1),
                                          h,
                                          delta)
                d2 = self._PchipDerivative(h,
                                          delta,
                                          self._N - self._N3,
                                          (self._Q4 - self._Q3) / (self._N - self._N3))
                self._Q3 += self._HermiteInterpolationOne(h, -delta, -d2, -d1)
                self._N3 -= 1

        elif self._N == 5:

            if s > self._Q4:
                self._Q0 = self._Q1
                self._Q1 = self._q
                self._q = self._Q3
                self._Q3 = self._Q4
                self._Q4 = s
            elif s > self._Q3:
                self._Q0 = self._Q1
                self._Q1 = self._q
                self._q = self._Q3
                self._Q3 = s
            elif s > self._q:
                self._Q0 = self._Q1
                self._Q1 = self._q
                self._q = s
            elif s > self._Q1:
                self._Q0 = self._Q1
                self._Q1 = s
            else:
                self._Q0 = s

        elif self._N == 4:
            if s < self._Q1:
                self._Q4 = self._Q3
                self._Q3 = self._q
                self._q = self._Q1
                self._Q1 = s
            elif s < self._q:
                self._Q4 = self._Q3
                self._Q3 = self._q
                self._q = s
            elif s < self._Q3:
                self._Q4 = self._Q3
                self._Q3 = s
            else:
                self._Q4 = s

        elif self._N == 3:
            if s < self._Q1:
                self._Q3 = self._q
                self._q = self._Q1
                self._Q1 = s
            elif s < self._q:
                self._Q3 = self._q
                self._q = s
            else:
                self._Q3 = s

        elif self._N == 2:
            if s > self._q:
                self._Q1 = self._q
                self._q = s
            else:
                self._Q1 = s
        else:
            self._q =  s


    def _PchipDerivative(self, h1, delta1, h2, delta2):
        return (h1 + h2) * 3 * delta1 * delta2 / ((h1 * 2 + h2) * delta1 + (h2 * 2 + h1) * delta2)


    def _PchipDerivativeEnd(self, h1, delta1, h2, delta2):
        d = (delta1 - delta2) * h1 / (h1 + h2) + delta1;
        return 0 if d < 0 else d


    def _HermiteInterpolationOne(self, h1, delta1, d1, d2):
        return ((d1 + d2 - delta1 * 2) / h1 + delta1 * 3 - d1 * 2 - d2) / h1 + d1

    def quantile(self):
        return self._q
