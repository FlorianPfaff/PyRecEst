from scipy.special import iv
from ..nonperiodic.gaussian_distribution import GaussianDistribution
from scipy.linalg import eigh
from pyrecest import random, mod, linalg, allclose, all, tile, concatenate, isscalar, pi, cos, eye, sqrt, imag, real, hstack, vstack, exp, sum, zeros
from scipy.special import iv

class GaussVMDistribution:
    def __init__(self, mu, P, alpha: float, beta, Gamma, kappa: float):
        # Check parameters
        n = mu.shape[0]
        assert P.shape == (n, n), 'P and mu must have matching size'
        assert allclose(P, P.T), 'P must be symmetric'
        assert all(eigh(P, eigvals_only=True) > 0), 'P must be positive definite'

        assert isscalar(alpha), 'alpha must be a scalar'

        assert beta.shape == mu.shape, 'size of beta must match size of mu'

        assert Gamma.shape == (n, n), 'Gamma and mu must have matching size'
        assert allclose(Gamma, Gamma.T), 'Gamma must be symmetric'

        assert isscalar(kappa) and kappa > 0, 'kappa has to be a positive scalar'
        # Assign parameters
        self.mu = mu
        self.P = P
        self.alpha = mod(alpha, 2 * pi)
        self.beta = beta
        self.Gamma = Gamma
        self.kappa = kappa
        self.A = linalg.cholesky(P)

        self.linD = len(mu)
        self.boundD = 1
        self.dim = self.linD + self.boundD

    def pdf(self, xa):
        assert xa.shape[0] == self.linD + 1
        p = random.multivariate_normal.pdf(xa[1:, :].T, mean=self.mu.ravel(), cov=self.P) * exp(
            self.kappa * cos(xa[0, :] - self.get_theta(xa[1:, :]))
        ) / (2 * pi * iv(0, self.kappa))
        return p

    def get_theta(self, xa):
        z = linalg.solve(self.A, xa - tile(self.mu, (1, xa.shape[1])))
        theta = tile(self.alpha, (1, xa.shape[1])) + self.beta.T @ z + 0.5 * sum(
            (linalg.cholesky(self.Gamma) @ z) ** 2, axis=0
        )
        return theta

    def mode(self):
        return concatenate(([self.alpha], self.mu.ravel()))

    def hybrid_moment(self):
        from ..circle.von_mises_distribution import VonMisesDistribution
        M = eye(self.linD) - 1j * self.Gamma
        eiphi = 1 / sqrt(linalg.det(M)) * VonMisesDistribution.besselratio(0, self.kappa) * exp(
            1j * self.alpha - 0.5 * self.beta.T @ linalg.solve(M, self.beta)
        )
        mu = concatenate((real(eiphi), imag(eiphi), self.mu.ravel()))
        return mu 
    
    def to_gaussian(self):
        # Convert to Gaussian
        mtmp = hstack((self.alpha, self.mu))
        Atmp = vstack((hstack((1 / sqrt(self.kappa), self.beta.T)), hstack((zeros((self.linD, 1)), self.A))))
        Ptmp = Atmp @ Atmp.T
        gauss = GaussianDistribution(mtmp, Ptmp)
        return gauss

    def linear_covariance(self):
        # Computes covariance of linear dimensions

        # Returns:
        #   C (linD x linD)
        #       covariance matrix
        C = self.P
        return C

    def marginalize_circular(self):
        gauss = GaussianDistribution(self.mu, self.P)
        return gauss