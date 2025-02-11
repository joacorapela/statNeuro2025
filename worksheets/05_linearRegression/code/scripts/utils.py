import math
import numpy as np


def computeBootstrapCIs(resamples, alpha):
    """Computes bootstrap confidence intervals at a significance level alpha
    from bootstrap resamples.

    :param resamples: bootstrap resamples

    :type resamples: list like

    :parm alpha: significance level

    :type float

    :return: a 2-tuple containing the lower and upper bound of the confidence interval.
    :rtype: tuple
    """
    sorted_resamples = np.sort(resamples)
    lower = sorted_resamples[int(alpha/2*len(resamples))]
    upper = sorted_resamples[int((1-alpha/2)*len(resamples))]
    return lower, upper


def bootstrapRegressionCoefs(X, Y, reg_coef, n_resamples):

    def fit(X, Y):
        I = np.eye(X.shape[1])
        coefs = np.linalg.solve(np.matmul(X.T, X) + reg_coef * I,
                                np.matmul(X.T, Y))
        return coefs

    N = X.shape[0]
    n_coefs = X.shape[1]
    resampled_coefs = np.empty(shape=(n_resamples, n_coefs), dtype=np.double)
    for i in range(n_resamples):
        indices = np.random.choice(N, size=N, replace=True)
        coefs = fit(X=X[indices, :], Y=Y[indices])
        resampled_coefs[i, :] = coefs
    return resampled_coefs


def buildDataMatrix(px, order, nRDs):
    totalNroCoefs = getTotalNroCoefs(order=order, nRDs=nRDs)
    dataMatrix = np.empty(shape=(px.shape[0], totalNroCoefs), dtype=np.double)
    index = 0

    # zeroth order term
    dataMatrix[:, index] = 1
    index += 1

    # first order terms
    for j in range(nRDs):
        dataMatrix[:, index] = px[:, j]
        index += 1
    if order < 2:
        return dataMatrix

    # second order terms
    items = np.empty(shape=(2,), dtype=np.int)
    for j0 in range(nRDs):
        items[0] = j0
        for j1 in range(j0, nRDs):
            items[1] = j1
            dataMatrix[:, index] = \
                computeSymmetriesCoef(items) *\
                computeProductCols(px, items)
            index += 1
    if order < 3:
        return dataMatrix

    # third order terms
    items = np.empty(shape=(3,), dtype=np.int)
    for j0 in range(nRDs):
        items[0] = j0
        for j1 in (j0, nRDs):
            items[1] = j1
            for j2 in range(j1, nRDs):
                items[2] = j2
                dataMatrix[:, index] = \
                    computeSymmetriesCoef(items) *\
                    computeProductCols(px, items)
                index += 1
    if order < 4:
        return dataMatrix

    # fourth order terms
    items = np.empty(shape=(4,), dtype=np.int)
    for j0 in range(nRDs):
        items[0] = j0
        for j1 in range(j0, nRDs):
            items[1] = j1
            for j2 in range(j1, nRDs):
                items[2] = j2
                for j3 in range(j2, nRDs):
                    items[3] = j3
                    dataMatrix[:, index] = \
                        computeSymmetriesCoef(items) *\
                        computeProductCols(px, items)
                    index += 1
    if order > 4:
        raise ValueError("Order should be less or equal than four")

    return dataMatrix


def computeSymmetriesCoef(indices):
    multiplicities = [1]
    lastDifferentIndex = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == lastDifferentIndex:
            multiplicities[len(multiplicities)-1] += 1
        else:
            lastDifferentIndex = indices[i]
            multiplicities.append(1)
    coef = np.math.factorial(len(indices))
    for multiplicity in multiplicities:
        coef /= np.math.factorial(multiplicity)
    return coef


def computeProductCols(matrix, indices):
    i = 0
    product = matrix[:, indices[i]]
    while i < len(indices)-1:
        i += 1
        product = product * matrix[:, indices[i]]
    return product


def getTotalNroCoefs(order, nRDs):
    nroUnknowns = 1
    for termOrder in range(1, order+1):
        nroUnknowns += getNroCoefsForTerm(order=termOrder, nRDs=nRDs)
    return nroUnknowns


def getNroCoefsForTerm(order, nRDs):
    answer = math.comb(nRDs+order-1, order)
    return answer


def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction.
    From https://en.wikipedia.org/wiki/Gabor_filter#Example_implementation
    sigma: Gaussian variance along x
    theta: cosine orientation
    Lambda: cosine period
    psi: cosine phase
    gamma: proportionality Gaussian variance along y
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(
        abs(nstds * sigma_x * np.cos(theta)),
        abs(nstds * sigma_y * np.sin(theta))
    )
    xmax = np.ceil(max(1, xmax))
    ymax = max(
        abs(nstds * sigma_x * np.sin(theta)),
        abs(nstds * sigma_y * np.cos(theta))
    )
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(
        -0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)
    ) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb
