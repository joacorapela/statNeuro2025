
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


def buildGammap(acov, p):
    Gammap = np.zeros(shape=(p, p))
    for i in range(p):
        for j in range(i, p):
            Gammap[i, j] = acov[j-i]
            Gammap[j, i] = Gammap[i, j]
    return Gammap


def estimateCoefsAndNoisVarARpYW(acov, p, N):
    Gammap = buildGammap(acov=acov, p=p)
    gammaph = acov[1:]
    phiHat = ...
    sigma2Hat = ...
    phiCovHat = ...
    return phiHat, phiCovHat, sigma2Hat
