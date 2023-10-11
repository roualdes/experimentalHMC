class OnlineQuantile():
    def __init__(self, p: float):
        self.p = p
        self.N = 0
        self.N0 = 1
        self.N1 = 2
        self.N2 = 3
        self.N3 = 4
        self.Q0 = 0
        self.Q1 = 0
        self.Q3 = 0
        self.Q4 = 0
        self.q = 0

    def update(self, s: float):
        self.N += 1
        if self.N > 5:
            if s <= self.Q3:
                self.N3 += 1
                if s <= self.q:
                    self.N2 += 1
                    if s <= self.Q1:
                        self.N1 += 1
                        if s <= self.Q0:
                            if s == self.Q0:
                                self.N0 += 1
                            else:
                                self.Q0 = s
                                self.N0 = 1
            elif s > self.Q4:
                self.Q4 = s

            s = (self.N - 1) * self.p * 0.5 + 1 - self.N1
            if s >= 1 and self.N2 - self.N1 > 1:
                h = self.N2 - self.N1
                delta = (self.q - self.Q1 ) / h
                d1 = self.PchipDerivative(self.N1 - self.N0,
                                          (self.Q1 - self.Q0) / (self.N1 - self.N0),
                                          h,
                                          delta)
                d2 = self.PchipDerivative(h,
                                          delta,
                                          self.N3 - self.N2,
                                          (self.Q3 - self.q) / (self.N3 - self.N2))
                self.Q1 += self.HermiteInterpolationOne(h, delta, d1, d2)
                self.N1 += 1

            elif s <= -1 and self.N1 - self.N0 > 1:
                h = self.N1 - self.N0
                delta = (self.Q1 - self.Q0) / h
                d1 = self.PchipDerivativeEnd(h,
                                        delta,
                                        self.N2 - self.N1,
                                        (self.q - self.Q1) / (self.N2 - self.N1))
                d2 = self.PchipDerivative(h,
                                          delta,
                                          self.N2 - self.N1,
                                          (self.q - self.Q1) / (self.N2 - self.N1))
                self.Q1 += self.HermiteInterpolationOne(h, -delta, -d2, -d1)
                self.N1 -= 1

            s = (self.N - 1) * self.p + 1 - self.N2
            if s >= 1 and self.N3 - self.N2 > 1:
                h = self.N3 - self.N2
                delta = (self.Q3 - self.q) / h
                d1 = self.PchipDerivative(self.N2 - self.N1,
                                     (self.q - self.Q1) / (self.N2 - self.N1),
                                     h,
                                     delta)
                d2 = self.PchipDerivative(h,
                                     delta,
                                     self.N - self.N3,
                                     (self.Q4 - self.Q3) / (self.N - self.N3))
                self.q += self.HermiteInterpolationOne(h, delta, d1, d2)
                self.N2 += 1
            elif s < -1 and self.N2 - self.N1 > 1:
                h = self.N2 - self.N1
                delta = (self.q - self.Q1) / h
                d1 = self.PchipDerivative(self.N1 - self.N0,
                                          (self.Q1 - self.Q0) / (self.N1 - self.N0),
                                          h,
                                          delta)
                d2 = self.PchipDerivative(h,
                                          delta,
                                          self.N3 - self.N2,
                                          (self.Q3 - self.q) / (self.N3 - self.N2))
                self.q += self.HermiteInterpolationOne(h, -delta, -d2, -d1)
                self.N2 -= 1

            s = (self.N - 1) * (1 + self.p) * 0.5 + 1 - self.N3
            if s >= 1 and self.N - self.N3 > 1:
                h = self.N - self.N3
                delta = (self.Q4 - self.Q3) / h
                d1 = self.PchipDerivative(self.N3 - self.N2,
                                          (self.Q3 - self.q) / (self.N3 - self.N2),
                                          h,
                                          delta)
                d2 = self.PchipDerivativeEnd(h,
                                             delta,
                                             self.N3 - self.N2,
                                             (self.Q3 - self.q) / (self.N3 - self.N2))
                self.Q3 += self.HermiteInterpolationOne(h, delta, d1, d2)
                self.N3 += 1
            elif s <= -1 and self.N3 - self.N2 > 1:
                h = self.N3 - self.N2
                delta = (self.Q3 - self.q) / h
                d1 = self.PchipDerivative(self.N2 - self.N1,
                                          (self.q - self.Q1) / (self.N2 - self.N1),
                                          h,
                                          delta)
                d2 = self.PchipDerivative(h,
                                          delta,
                                          self.N - self.N3,
                                          (self.Q4 - self.Q3) / (self.N - self.N3))
                self.Q3 += self.HermiteInterpolationOne(h, -delta, -d2, -d1)
                self.N3 -= 1

        elif self.N == 5:

            if s > self.Q4:
                self.Q0 = self.Q1
                self.Q1 = self.q
                self.q = self.Q3
                self.Q3 = self.Q4
                self.Q4 = s
            elif s > self.Q3:
                self.Q0 = self.Q1
                self.Q1 = self.q
                self.q = self.Q3
                self.Q3 = s
            elif s > self.q:
                self.Q0 = self.Q1
                self.Q1 = self.q
                self.q = s
            elif s > self.Q1:
                self.Q0 = self.Q1
                self.Q1 = s
            else:
                self.Q0 = s

        elif self.N == 4:
            if s < self.Q1:
                self.Q4 = self.Q3
                self.Q3 = self.q
                self.q = self.Q1
                self.Q1 = s
            elif s < self.q:
                self.Q4 = self.Q3
                self.Q3 = self.q
                self.q = s
            elif s < self.Q3:
                self.Q4 = self.Q3
                self.Q3 = s
            else:
                self.Q4 = s

        elif self.N == 3:
            if s < self.Q1:
                self.Q3 = self.q
                self.q = self.Q1
                self.Q1 = s
            elif s < self.q:
                self.Q3 = self.q
                self.q = s
            else:
                self.Q3 = s

        elif self.N == 2:
            if s > self.q:
                self.Q1 = self.q
                self.q = s
            else:
                self.Q1 = s
        else:
            self.q =  s


    def PchipDerivative(self, h1, delta1, h2, delta2):
        return (h1 + h2) * 3 * delta1 * delta2 / ((h1 * 2 + h2) * delta1 + (h2 * 2 + h1) * delta2)


    def PchipDerivativeEnd(self, h1, delta1, h2, delta2):
        d = (delta1 - delta2) * h1 / (h1 + h2) + delta1;
        return 0 if d < 0 else d


    def HermiteInterpolationOne(self, h1, delta1, d1, d2):
        return ((d1 + d2 - delta1 * 2) / h1 + delta1 * 3 - d1 * 2 - d2) / h1 + d1

    def quantile(self):
        return self.q
