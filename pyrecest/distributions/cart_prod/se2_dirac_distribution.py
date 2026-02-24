from .abstract_se2_distribution import AbstractSE2Distribution
from .hypercylindrical_dirac_distribution import HypercylindricalDiracDistribution
import numpy as np

class SE2DiracDistribution(HypercylindricalDiracDistribution, AbstractSE2Distribution):

    def __init__(self, d_, w_=None):
        self.d = d_
        if w_ is None:
            w_ = np.ones((1, d_.shape[1])) / d_.shape[1]
        self.w = w_

    def mean4D(self):
        s = self.d
        S = np.column_stack((np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1:3]))
        mu = np.sum(np.tile(self.w, (4, 1)) * S, axis=0)
        return mu

    def covariance4D(self):
        s = self.d
        S = np.column_stack((np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1:3]))
        C = S @ np.diag(self.w) @ S.T
        return C

    @staticmethod
    def fromDistribution(dist, nParticles):
        raise NotImplementedError()