import math

import numpy as np


def fft_nextgoodsize(N):
    if N <= 2:
        return 2

    while True:
        m = N

        while np.remainder(m, 2) == 0:
            m /= 2

        while np.remainder(m, 3) == 0:
            m /= 3

        while np.remainder(m, 5) == 0:
            m /= 5

        if m <= 1:
            return N

        N += 1


def autocovariance(x):
    N = np.size(x)
    Mt2 = 2 * fft_nextgoodsize(N)
    yc = np.zeros(Mt2)
    yc[:N] = x - np.mean(x)
    t = np.fft.fft(yc)
    ac = np.fft.fft(np.conj(t) * t)
    return np.real(ac)[:N] / (N * N * 2)


def isconstant(x):
    mn = np.min(x)
    mx = np.max(x)
    return np.isclose(mn, mx)


def ess(x):
    niterations, nchains = np.shape(x)

    if niterations < 3 or np.any(np.isnan(x)):
        return np.nan

    if np.any(np.isinf(x)):
        return np.nan

    if isconstant(x):
        return np.nan

    acov = np.apply_along_axis(autocovariance, 0, x)
    chain_mean = np.mean(x, axis = 0)
    mean_var = np.mean(acov[0, :]) * niterations / (niterations - 1)
    var_plus = mean_var * (niterations - 1) / niterations

    if nchains > 1:
        var_plus += np.var(chain_mean, ddof = 1)

    rhohat = np.zeros(niterations)
    rhohat_even = 1
    rhohat[0] = rhohat_even
    rhohat_odd = 1 - (mean_var - np.mean(acov[1, :])) / var_plus
    rhohat[1] = rhohat_odd

    t = 1
    while t < niterations - 4 and rhohat_even + rhohat_odd > 0:
        rhohat_even = 1 - (mean_var - np.mean(acov[t + 1, :])) / var_plus
        rhohat_odd = 1 - (mean_var - np.mean(acov[t + 2, :])) / var_plus

        if rhohat_even + rhohat_odd >= 0:
            rhohat[t + 1] = rhohat_even
            rhohat[t + 2] = rhohat_odd

        t += 2

    max_t = t
    if rhohat_even > 0:
        rhohat[max_t + 1] = rhohat_even

    t = 1
    while t <= max_t - 3:
        if rhohat[t + 1] + rhohat[t + 2] > rhohat[t - 1] + rhohat[t]:
            rhohat[t + 1] = (rhohat[t - 1] + rhohat[t]) / 2
            rhohat[t + 2] = rhohat[t + 1]
        t += 2

    ess = nchains * niterations
    tau = -1 + 2 * np.sum(rhohat[0:np.maximum(1, max_t)]) + rhohat[max_t + 1]
    tau = np.maximum(tau, 1 / np.log10(ess))
    return ess / tau


def isodd(x: float):
    return x & 0x1


def Qinv(x):
    # x where Q(x) = p for 0 < p <= 0.5
    # Abramowitz and Stegun formula 26.2.23
    # The absolute value of the error should be less than 4.5 e -4
    c = np.asarray([2.515517, 0.802853, 0.010328])
    d = np.asarray([1.432788, 0.189269, 0.001308])
    numerator = (c[2] * x + c[1]) * x + c[0]
    denominator = ((d[2] * x + d[1]) * x + d[0]) * x + 1.0
    return x - numerator / denominator


def normal_quantile(p):
    assert np.all(p > 0.0) and np.all(p < 1)
    q = np.empty_like(p)
    idx = p >= 0.5
    q[~idx] = -Qinv( np.sqrt(-2.0 * np.log(p[~idx])) )
    q[idx] = Qinv( np.sqrt(-2.0 * np.log(1 - p[idx])) )
    return q


def splitchains(x):
    niterations = np.shape(x)[0]
    if niterations < 2:
        return x

    if isodd(niterations):
        niterations -= 1

    half = np.floor_divide(niterations, 2)
    return np.concatenate((x[:half, :], x[half:, :]), axis = 1)


def ess_mean(x):
    N, D, C = np.shape(x)
    esses = np.zeros(D)
    for d in range(D):
        esses[d] = ess(splitchains(x[:, d, :]))
    return esses


def ess_basic(x):
    N, D, C = np.shape(x)
    esses = np.zeros(D)
    for d in range(D):
        esses[d] = ess(x[:, d, :])
    return esses


def ess_tail(x):
    q05, q95 = np.quantile(x, [0.05, 0.95])
    I05 = x <= q05
    I95 = x <= q95

    N, D, C = np.shape(x)
    esses = np.zeros(D)
    for d in range(D):
        essq05 = ess(splitchains(I05[:, d, :]))
        essq95 = ess(splitchains(I95[:, d, :]))
        esses[d] = np.minimum(essq05, essq95)
    return esses


def ess_quantile(x, prob: float = 0.5):
    q50 = np.quantile(x, [prob])
    I50 = x <= q50

    N, D, C = np.shape(x)
    esses = np.zeros(D)
    for d in range(D):
        esses[d] = ess(splitchains(I50[:, d, :]))
    return esses


def ess_std(x):
    m = np.mean(x)
    N, D, C = np.shape(x)
    esses = np.zeros(D)
    for d in range(D):
        esses[d] = ess(splitchains(np.abs(x[:, d, :] - m)))
    return esses


def rhat(x):
    if np.any(np.isnan(x)):
        return np.nan

    if np.any(np.isinf(x)):
        return np.nan

    if isconstant(x):
        return np.nan

    niterations, nchains = np.shape(x)
    chain_mean = np.mean(x, axis = 0)
    chain_var = np.var(x, axis = 0, ddof = 1)
    var_between = niterations * np.var(chain_mean, ddof = 1)
    var_within = np.mean(chain_var)

    return np.sqrt((var_between / var_within + niterations - 1) / niterations)


def rhat_basic(x):
    N, D, C = np.shape(x)
    rhats = np.zeros(D)
    for d in range(D):
        rhats[d] = rhat(splitchains(x[:, d, :]))
    return rhats


def fold(x):
    return np.abs(x - np.median(x))


# def tiedrank(x):
#     x = x.ravel()

#     n = np.size(x)
#     p = np.s
# array = numpy.array([4,2,7,1])
# temp = array.argsort()
# ranks = numpy.empty_like(temp)
# ranks[temp] = numpy.arange(len(array))
