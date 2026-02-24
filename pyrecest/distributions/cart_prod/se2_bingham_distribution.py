import numpy as np
from numpy.linalg import pinv, eig
from scipy.stats import multivariate_normal
from pyrecest.distributions.cart_prod.abstract_se2_distribution import AbstractSE2Distribution


class SE2BinghamDistribution(AbstractSE2Distribution):

    def __init__(self, C1, C2, C3):
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

    def computeNC(self):
        # Compute normalization constant of the distribution.
        pass

    def mode(self):
        # Computes one of the modes of the distribution.
        pass

    def sampleDeterministic(self):
        # Generates deterministic samples.
        pass

    def sample(self, n=10000):
        # Samples from current distribution.
        pass

    def plotState(self, scalingFactor=1, circleColor=None, angleColor=None, samplesForMatching=10000):
        if circleColor is None:
            circleColor = np.array([0, 0.4470, 0.7410])
        if angleColor is None:
            angleColor = np.array([0.8500, 0.3250, 0.0980])
        pass

    def getBinghamMarginal(self):
        # Computes Bingham marginal of circular part.
        BM = self.C1 - (self.C2.T @ pinv(self.C3) @ self.C2)
        M, Z = eig(BM)
        order = np.argsort(np.diag(Z), axis=-1)
        Z = Z - Z[2]
        M = M[:, order]
        b = BinghamDistribution(Z, M)
        return b

    @staticmethod
    def fit(samples, weights=None):
        if weights is None:
            weights = np.ones((1, samples.shape[1])) / samples.shape[1]
        # Estimates parameters of SE2 distribution from samples.
        pass