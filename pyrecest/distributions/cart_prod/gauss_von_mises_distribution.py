import numpy as np
from scipy.special import iv
from ..nonperiodic.gaussian_distribution import GaussianDistribution
from scipy.linalg import eigh

class GaussVMDistribution:
    def __init__(self, mu: np.ndarray, P: np.ndarray, alpha: float, beta: np.ndarray, Gamma: np.ndarray, kappa: float):
        # Check parameters
        n = mu.shape[0]
        assert P.shape == (n, n), 'P and mu must have matching size'
        assert np.allclose(P, P.T), 'P must be symmetric'
        assert np.all(eigh(P, eigvals_only=True) > 0), 'P must be positive definite'

        assert np.isscalar(alpha), 'alpha must be a scalar'

        assert beta.shape == mu.shape, 'size of beta must match size of mu'

        assert Gamma.shape == (n, n), 'Gamma and mu must have matching size'
        assert np.allclose(Gamma, Gamma.T), 'Gamma must be symmetric'

        assert np.isscalar(kappa) and kappa > 0, 'kappa has to be a positive scalar'
        # Assign parameters
        self.mu = mu
        self.P = P
        self.alpha = np.mod(alpha, 2 * np.pi)
        self.beta = beta
        self.Gamma = Gamma
        self.kappa = kappa
        self.A = np.linalg.cholesky(P)

        self.linD = len(mu)
        self.boundD = 1
        self.dim = self.linD + self.boundD

    def pdf(self, xa: np.ndarray) -> np.ndarray:
        assert xa.shape[0] == self.linD + 1
        p = multivariate_normal.pdf(xa[1:, :].T, mean=self.mu.ravel(), cov=self.P) * np.exp(
            self.kappa * np.cos(xa[0, :] - self.get_theta(xa[1:, :]))
        ) / (2 * np.pi * besseli(0, self.kappa))
        return p

    def get_theta(self, xa: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(self.A, xa - np.tile(self.mu, (1, xa.shape[1])))
        theta = np.tile(self.alpha, (1, xa.shape[1])) + self.beta.T @ z + 0.5 * np.sum(
            (np.linalg.cholesky(self.Gamma) @ z) ** 2, axis=0
        )
        return theta

    def mode(self) -> np.ndarray:
        return np.concatenate(([self.alpha], self.mu.ravel()))

    def hybrid_moment(self) -> np.ndarray:
        from ..circle.von_mises_distribution import VonMisesDistribution
        M = np.eye(self.linD) - 1j * self.Gamma
        eiphi = 1 / np.sqrt(np.linalg.det(M)) * VonMisesDistribution.besselratio(0, self.kappa) * np.exp(
            1j * self.alpha - 0.5 * self.beta.T @ np.linalg.solve(M, self.beta)
        )
        mu = np.concatenate((np.real(eiphi), np.imag(eiphi), self.mu.ravel()))
        return mu 
    
    def sample_deterministic_horwood(self):
        # Horwood 5.1
        dim = self.linD

        # solution for canonical form (mu = 0, P=I, alpha=beta=Gamma=0)
        B = lambda p, kappa: 1 - iv(p, kappa) / iv(0, kappa)
        xi = np.sqrt(3)
        eta = np.arccos(B(2, self.kappa) / 2 / B(1, self.kappa) - 1)
        wxi0 = 1 / 6
        weta0 = B(1, self.kappa)**2 / (4 * B(1, self.kappa) - B(2, self.kappa))
        w00 = 1 - 2 * weta0 - 2 * dim * wxi0
        N00 = np.zeros((dim + 1, 1))
        Neta0 = np.hstack((np.zeros((dim, 2)), np.array([[-eta, eta]]).T))
        Nxi0 = np.zeros((dim + 1, 2 * dim))
        for i in range(dim):
            Nxi0[i, 2 * i - 1] = -xi
            Nxi0[i, 2 * i] = xi

    def to_gaussian(self):
        # Convert to Gaussian

        # Uses approximation only valid for large kappa and small
        # Gamma, see Horwood, 4.6
        mtmp = np.hstack((self.alpha, self.mu))
        Atmp = np.vstack((np.hstack((1 / np.sqrt(self.kappa), self.beta.T)), np.hstack((np.zeros((self.linD, 1)), self.A))))
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