
"""
Utility functions for time series analysis
==========================================
"""

import numpy as np


def estimateACov(x, lags):
    acov = np.zeros(len(lags))
    xMu = np.mean(x)
    for h in lags:
        if h > 0:
            xs = x[h:] - xMu
            xt = x[:-h] - xMu
        elif h == 0:
            xs = x - xMu
            xt = x - xMu
        acov[h] = np.mean(xs * xt)
    return acov


def buildGamma(acov, m):
    Gammap = np.zeros(shape=(m, m))
    for i in range(m):
        for j in range(i, m):
            Gammap[i, j] = acov[j-i]
            Gammap[j, i] = Gammap[i, j]
    return Gammap


def estimateCoefsAndNoiseVarARpYW(acov, p, N):
    Gammap = buildGamma(acov=acov, m=p)
    gammaph = acov[1:]
    phiHat = ... # complete
    sigma2Hat = ... # complete
    phiCovHat = ... # complete
    return phiHat, phiCovHat, sigma2Hat


def simulateARp(phi, w):
    p = len(phi)
    x = np.zeros(len(w), dtype=np.double)
    x[:p] = w[:p]
    for i in range(p, len(w)):
        for j in range(p):
            x[i] += phi[j] * x[i-(j+1)]
        x[i] += w[i]
    return x


def forecast(x, acov, mu, m, max_h):
    Gamma_m = buildGamma(acov=acov, m=m)
    forecasts_means = np.empty(max_h, dtype=np.double)
    forecasts_vars = np.empty(max_h, dtype=np.double)
    xMinusMu_mR = (x-mu)[::-1][:m]
    for h in range(1, max_h+1):
        gamma_mh = ... # complete
        a_m = ... # complete
        forecasts_means[h-1] = ... # complete
        forecasts_vars[h-1] = ... # complete
    return forecasts_means, forecasts_vars
