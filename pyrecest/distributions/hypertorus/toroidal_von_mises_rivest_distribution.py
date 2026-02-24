import numpy as np
from scipy.special import comb, iv
from ..circle.custom_circular_distribution import CustomCircularDistribution

class ToroidalVonMisesRivestDistribution:
    def __init__(self, mu_, kappa_, alpha_, beta_):
        assert mu_.shape == (2,)
        assert kappa_.shape == (2,)

        self.mu = mu_
        self.kappa = kappa_
        self.alpha = alpha_
        self.beta = beta_

    def circular_correlation_jammalamadaka(self):
        sinAsinB = lambda m: comb(2 * m, m) * m * (self.beta ** 2 / 4 / self.kappa[0] / self.kappa[1]) ** m * iv(m, self.kappa[0]) * iv(m, self.kappa[1])
        cosAsquared = lambda m: comb(2 * m, m) * (self.beta ** 2 / 4 / self.kappa[1]) ** m * (iv(m + 2, self.kappa[0]) / self.kappa[0] ** m + iv(m + 1, self.kappa[0]) / self.kappa[0] ** (m + 1)) * iv(m, self.kappa[1])
        cosBsquared = lambda m: comb(2 * m, m) * (self.beta ** 2 / 4 / self.kappa[0]) ** m * (iv(m + 2, self.kappa[1]) / self.kappa[1] ** m + iv(m + 1, self.kappa[1]) / self.kappa[1] ** (m + 1)) * iv(m, self.kappa[0])
        
        EsinAsinB = 8 * np.pi ** 2 * self.alpha / self.beta * sum(sinAsinB(m) for m in range(11))
        EsinAsquared = 1 - self.alpha * 4 * np.pi ** 2 * sum(cosAsquared(m) for m in range(11))
        EsinBsquared = 1 - self.alpha * 4 * np.pi ** 2 * sum(cosBsquared(m) for m in range(11))
        
        rho = EsinAsinB / np.sqrt(EsinAsquared * EsinBsquared)
        return rho

    def marginalize_to_1D(self, dimension):
        assert dimension == 1 or dimension == 2
        
        other = 3 - dimension
        f = lambda x: 2 * np.pi * self.alpha * iv(0, np.sqrt(self.kappa[other] ** 2 + self.beta ** 2 * np.sin(x - self.mu[dimension]) ** 2)) * np.exp(self.kappa[dimension] * np.cos(x - self.mu[dimension]))
        dist = CustomCircularDistribution(f)
        return dist

    def shift(self, shift_by):
        assert shift_by.shape == (self.dim,)
        tvm = self
        tvm.mu = np.mod(self.mu + shift_by, 2 * np.pi)
        return tvm
       
