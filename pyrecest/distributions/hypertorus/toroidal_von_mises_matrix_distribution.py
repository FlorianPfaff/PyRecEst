import numpy as np
import scipy
from pyrecest.distributions.circle.custom_circular_distribution import CustomCircularDistribution
from pyrecest.distributions import AbstractToroidalDistribution
from scipy.special import iv

class ToroidalVonMisesMatrixDistribution(AbstractToroidalDistribution):
    def __init__(self, mu, kappa, A):
        AbstractToroidalDistribution.__init__(self)
        assert mu.shape == (2,)
        assert kappa.shape == (2,)
        assert A.shape == (2, 2)
        assert kappa[0] > 0
        assert kappa[1] > 0
        self.mu = np.mod(mu, 2 * np.pi)
        self.kappa = kappa
        self.A = A

        if self.kappa[0] > 1.5 or self.kappa[1] > 1.5 or np.max(np.abs(self.A)) > 1:
            self.C = 1
            self.C = 1 / self.integrate()
        else:
            self.C = self.norm_const_approx()
            
    def norm_const_approx(self, n=8):
        """
            Approximate normalization constant
            See
            Gerhard Kurz, Uwe D. Hanebeck,
            Toroidal Information Fusion Based on the Bivariate von Mises Distribution
            Proceedings of the 2015 IEEE International Conference on Multisensor Fusion and Information Integration (MFI 2015), 
            San Diego, California, USA, September 2015.
        """
        a11 = self.A[0][0]
        a12 = self.A[0][1]
        a21 = self.A[1][0]
        a22 = self.A[1][1]
        k1 = self.kappa[0]
        k2 = self.kappa[1]
        
        total = 4 * np.pi ** 2  # n=0
        
        if n >= 1:
            total += 0 / scipy.special.factorial(1)  # n=1
        
        if n >= 2:
            total += (a11 ** 2 + a12 ** 2 + a21 ** 2 + a22 ** 2 + 2 * k1 ** 2 + 2 * k2 ** 2) * np.pi ** 2 / scipy.special.factorial(2)  # n=2
        
        if n >= 3:
            total += (6 * a11 * k1 * k2 * np.pi ** 2) / scipy.special.factorial(3)  # n=3
        
        if n >= 4:
            total += (3 / 16) * (3 * a11 ** 4 + 3 * a12 ** 4 + 3 * a21 ** 4 + 8 * a11 * a12 * a21 * a22 + 6 * a21 ** 2 * a22 ** 2 + 3 * a22 ** 4 + 8 * a21 ** 2 * k1 ** 2 + 8 * a22 ** 2 * k1 ** 2 + 8 * k1 ** 4 + 8 * (3 * a21 ** 2 + a22 ** 2 + 4 * k1 ** 2) * k2 ** 2 + 8 * k2 ** 4 + 2 * a11 ** 2 * (3 * a12 ** 2 + 3 * a21 ** 2 + a22 ** 2 + 12 * (k1 ** 2 + k2 ** 2)) + 2 * a12 ** 2 * (a21 ** 2 + 3 * a22 ** 2 + 4 * (3 * k1 ** 2 + k2 ** 2))) * np.pi ** 2 / scipy.special.factorial(4)
        if n >= 5:
            total += (15 / 4) * np.pi ** 2 * k1 * k2 * (3 * a11 ** 3 + 3 * a11 * a12 ** 2 + 3 * a11 * a21 ** 2 + a11 * a22 ** 2 + 4 * a11 * k1 ** 2 + 4 * a11 * k2 ** 2 + 2 * a12 * a21 * a22) / scipy.special.factorial(5)

        if n >= 6:
            total += (
                (5 / 64) * np.pi ** 2 * (
                    5 * a11 ** 6 + 15 * a11 ** 4 * a12 ** 2 + 15 * a11 ** 4 * a21 ** 2 + 3 * a11 ** 4 * a22 ** 2 + 90 * a11 ** 4 * k1 ** 2 + 90 * a11 ** 4 * k2 ** 2 +
                    24 * a11 ** 3 * a12 * a21 * a22 + 15 * a11 ** 2 * a12 ** 4 + 18 * a11 ** 2 * a12 ** 2 * a21 ** 2 + 18 * a11 ** 2 * a12 ** 2 * a22 ** 2 + 180 * a11 ** 2 * a12 ** 2 * k1 ** 2 + 108 * a11 ** 2 * a12 ** 2 * k2 ** 2 +
                    15 * a11 ** 2 * a21 ** 4 + 18 * a11 ** 2 * a21 ** 2 * a22 ** 2 + 108 * a11 ** 2 * a21 ** 2 * k1 ** 2 + 180 * a11 ** 2 * a21 ** 2 * k2 ** 2 + 3 * a11 ** 2 * a22 ** 4 + 36 * a11 ** 2 * a22 ** 2 * k1 ** 2 + 36 * a11 ** 2 * a22 ** 2 * k2 ** 2 +
                    120 * a11 ** 2 * k1 ** 4 + 648 * a11 ** 2 * k1 ** 2 * k2 ** 2 + 120 * a11 ** 2 * k2 ** 4 + 24 * a11 * a12 ** 3 * a21 * a22 + 24 * a11 * a12 * a21 ** 3 * a22 + 24 * a11 * a12 * a21 * a22 ** 3 + 144 * a11 * a12 * a21 * a22 * k1 ** 2 + 144 * a11 * a12 * a21 * a22 * k2 ** 2 +
                    5 * a12 ** 6 + 3 * a12 ** 4 * a21 ** 2 + 15 * a12 ** 4 * a22 ** 2 + 90 * a12 ** 4 * k1 ** 2 + 18 * a12 ** 4 * k2 ** 2 + 3 * a12 ** 2 * a21 ** 4 + 18 * a12 ** 2 * a21 ** 2 * a22 ** 2 + 36 * a12 ** 2 * a21 ** 2 * k1 ** 2 + 36 * a12 ** 2 * a21 ** 2 * k2 ** 2 +
                    15 * a12 ** 2 * a22 ** 4 + 108 * a12 ** 2 * a22 ** 2 * k1 ** 2 + 36 * a12 ** 2 * a22 ** 2 * k2 ** 2 + 120 * a12 ** 2 * k1 ** 4 + 216 * a12 ** 2 * k1 ** 2 * k2 ** 2 + 24 * a12 ** 2 * k2 ** 4 + 5 * a21 ** 6 + 15 * a21 ** 4 * a22 ** 2 + 18 * a21 ** 4 * k1 ** 2 +
                    90 * a21 ** 4 * k2 ** 2 + 15 * a21 ** 2 * a22 ** 4 + 36 * a21 ** 2 * a22 ** 2 * k1 ** 2 + 108 * a21 ** 2 * a22 ** 2 * k2 ** 2 + 24 * a21 ** 2 * k1 ** 4 + 216 * a21 ** 2 * k1 ** 2 * k2 ** 2 + 120 * a21 ** 2 * k2 ** 4 + 5 * a22 ** 6 + 18 * a22 ** 4 * k1 ** 2 +
                    18 * a22 ** 4 * k2 ** 2 + 24 * a22 ** 2 * k1 ** 4 + 72 * a22 ** 2 * k1 ** 2 * k2 ** 2 + 24 * a22 ** 2 * k2 ** 4 + 16 * k1 ** 6 + 144 * k1 ** 4 * k2 ** 2 + 144 * k1 ** 2 * k2 ** 4 + 16 * k2 ** 6
                ) / scipy.special.factorial(6)
            )

        if n >= 7:
            total += (
                (105 / 32) * k1 * k2 * np.pi ** 2 * (
                    5 * a11 ** 5 + 10 * a11 ** 3 * a12 ** 2 + 10 * a11 ** 3 * a21 ** 2 + 2 * a11 ** 3 * a22 ** 2 + 20 * a11 ** 3 * k1 ** 2 + 20 * a11 ** 3 * k2 ** 2 +
                    12 * a11 ** 2 * a12 * a21 * a22 + 5 * a11 * a12 ** 4 + 6 * a11 * a12 ** 2 * a21 ** 2 + 6 * a11 * a12 ** 2 * a22 ** 2 + 20 * a11 * a12 ** 2 * k1 ** 2 + 12 * a11 * a12 ** 2 * k2 ** 2 +
                    5 * a11 * a21 ** 4 + 6 * a11 * a21 ** 2 * a22 ** 2 + 12 * a11 * a21 ** 2 * k1 ** 2 + 20 * a11 * a21 ** 2 * k2 ** 2 + a11 * a22 ** 4 + 4 * a11 * a22 ** 2 * k1 ** 2 + 4 * a11 * a22 ** 2 * k2 ** 2 +
                    8 * a11 * k1 ** 4 + 24 * a11 * k1 ** 2 * k2 ** 2 + 8 * a11 * k2 ** 4 + 4 * a12 ** 3 * a21 * a22 + 4 * a12 * a21 ** 3 * a22 + 4 * a12 * a21 * a22 ** 3 + 8 * a12 * a21 * a22 * k1 ** 2 + 8 * a12 * a21 * a22 * k2 ** 2
                ) / scipy.special.factorial(7)
            )

        if n >= 8:
            part1 = (
                35 * a11 ** 8 + 140 * a11 ** 6 * a12 ** 2 + 140 * a11 ** 6 * a21 ** 2 + 20 * a11 ** 6 * a22 ** 2 + 
                1120 * a11 ** 6 * k1 ** 2 + 1120 * a11 ** 6 * k2 ** 2 + 240 * a11 ** 5 * a12 * a21 * a22 + 
                210 * a11 ** 4 * a12 ** 4 + 300 * a11 ** 4 * a12 ** 2 * a21 ** 2 + 180 * a11 ** 4 * a12 ** 2 * a22 ** 2 + 
                3360 * a11 ** 4 * a12 ** 2 * k1 ** 2 + 2400 * a11 ** 4 * a12 ** 2 * k2 ** 2 + 210 * a11 ** 4 * a21 ** 4 + 
                180 * a11 ** 4 * a21 ** 2 * a22 ** 2 + 2400 * a11 ** 4 * a21 ** 2 * k1 ** 2 + 3360 * a11 ** 4 * a21 ** 2 * k2 ** 2 + 
                18 * a11 ** 4 * a22 ** 4 + 480 * a11 ** 4 * a22 ** 2 * k1 ** 2 + 480 * a11 ** 4 * a22 ** 2 * k2 ** 2 + 
                3360 * a11 ** 4 * k1 ** 4 + 19200 * a11 ** 4 * k1 ** 2 * k2 ** 2 + 3360 * a11 ** 4 * k2 ** 4 + 
                480 * a11 ** 3 * a12 ** 3 * a21 * a22 + 480 * a11 ** 3 * a12 * a21 ** 3 * a22 + 288 * a11 ** 3 * a12 * a21 * a22 ** 3 + 
                3840 * a11 ** 3 * a12 * a21 * a22 * k1 ** 2 + 3840 * a11 ** 3 * a12 * a21 * a22 * k2 ** 2 + 
                140 * a11 ** 2 * a12 ** 6 + 180 * a11 ** 2 * a12 ** 4 * a21 ** 2 + 300 * a11 ** 2 * a12 ** 4 * a22 ** 2 + 
                3360 * a11 ** 2 * a12 ** 4 * k1 ** 2 + 1440 * a11 ** 2 * a12 ** 4 * k2 ** 2 + 180 * a11 ** 2 * a12 ** 2 * a21 ** 4
            )

            part2 = (
                648 * a11 ** 2 * a12 ** 2 * a21 ** 2 * a22 ** 2 + 2880 * a11 ** 2 * a12 ** 2 * a21 ** 2 * k1 ** 2 + 
                2880 * a11 ** 2 * a12 ** 2 * a21 ** 2 * k2 ** 2 + 180 * a11 ** 2 * a12 ** 2 * a22 ** 4 + 
                2880 * a11 ** 2 * a12 ** 2 * a22 ** 2 * k1 ** 2 + 1728 * a11 ** 2 * a12 ** 2 * a22 ** 2 * k2 ** 2 + 
                6720 * a11 ** 2 * a12 ** 2 * k1 ** 4 + 23040 * a11 ** 2 * a12 ** 2 * k1 ** 2 * k2 ** 2 + 2880 * a11 ** 2 * a12 ** 2 * k2 ** 4 + 
                140 * a11 ** 2 * a21 ** 6 + 300 * a11 ** 2 * a21 ** 4 * a22 ** 2 + 1440 * a11 ** 2 * a21 ** 4 * k1 ** 2 + 
                3360 * a11 ** 2 * a21 ** 4 * k2 ** 2 + 180 * a11 ** 2 * a21 ** 2 * a22 ** 4 + 1728 * a11 ** 2 * a21 ** 2 * a22 ** 2 * k1 ** 2 + 
                2880 * a11 ** 2 * a21 ** 2 * a22 ** 2 * k2 ** 2 + 2880 * a11 ** 2 * a21 ** 2 * k1 ** 4 + 
                23040 * a11 ** 2 * a21 ** 2 * k1 ** 2 * k2 ** 2 + 6720 * a11 ** 2 * a21 ** 2 * k2 ** 4 + 20 * a11 ** 2 * a22 ** 6 + 
                288 * a11 ** 2 * a22 ** 4 * k1 ** 2 + 288 * a11 ** 2 * a22 ** 4 * k2 ** 2 + 960 * a11 ** 2 * a22 ** 2 * k1 ** 4 + 
                4608 * a11 ** 2 * a22 ** 2 * k1 ** 2 * k2 ** 2 + 960 * a11 ** 2 * a22 ** 2 * k2 ** 4 + 1792 * a11 ** 2 * k1 ** 6 + 
                23040 * a11 ** 2 * k1 ** 4 * k2 ** 2 + 23040 * a11 ** 2 * k1 ** 2 * k2 ** 4 + 1792 * a11 ** 2 * k2 ** 6
            )
            
            part3 = (
                480 * a11 * a12 * a21 ** 3 * a22 ** 3 + 2304 * a11 * a12 * a21 ** 3 * a22 * k1 ** 2 + 
                3840 * a11 * a12 * a21 ** 3 * a22 * k2 ** 2 + 240 * a11 * a12 * a21 * a22 ** 5 + 
                2304 * a11 * a12 * a21 * a22 ** 3 * k1 ** 2 + 2304 * a11 * a12 * a21 * a22 ** 3 * k2 ** 2 + 
                3840 * a11 * a12 * a21 * a22 * k1 ** 4 + 18432 * a11 * a12 * a21 * a22 * k1 ** 2 * k2 ** 2 + 
                3840 * a11 * a12 * a21 * a22 * k2 ** 4 + 35 * a12 ** 8 + 20 * a12 ** 6 * a21 ** 2 + 140 * a12 ** 6 * a22 ** 2 + 
                1120 * a12 ** 6 * k1 ** 2 + 160 * a12 ** 6 * k2 ** 2 + 18 * a12 ** 4 * a21 ** 4 + 180 * a12 ** 4 * a21 ** 2 * a22 ** 2 + 
                480 * a12 ** 4 * a21 ** 2 * k1 ** 2 + 288 * a12 ** 4 * a21 ** 2 * k2 ** 2 + 210 * a12 ** 4 * a22 ** 4 + 
                2400 * a12 ** 4 * a22 ** 2 * k1 ** 2 + 480 * a12 ** 4 * a22 ** 2 * k2 ** 2 + 3360 * a12 ** 4 * k1 ** 4 + 
                3840 * a12 ** 4 * k1 ** 2 * k2 ** 2 + 288 * a12 ** 4 * k2 ** 4
            )

            part4 = (
                20 * a12 ** 2 * a21 ** 6 + 180 * a12 ** 2 * a21 ** 4 * a22 ** 2 + 288 * a12 ** 2 * a21 ** 4 * k1 ** 2 + 
                480 * a12 ** 2 * a21 ** 4 * k2 ** 2 + 300 * a12 ** 2 * a21 ** 2 * a22 ** 4 + 1728 * a12 ** 2 * a21 ** 2 * a22 ** 2 * k1 ** 2 + 
                1728 * a12 ** 2 * a21 ** 2 * a22 ** 2 * k2 ** 2 + 960 * a12 ** 2 * a21 ** 2 * k1 ** 4 + 
                4608 * a12 ** 2 * a21 ** 2 * k1 ** 2 * k2 ** 2 + 960 * a12 ** 2 * a21 ** 2 * k2 ** 4 + 140 * a12 ** 2 * a22 ** 6 + 
                1440 * a12 ** 2 * a22 ** 4 * k1 ** 2 + 480 * a12 ** 2 * a22 ** 4 * k2 ** 2 + 2880 * a12 ** 2 * a22 ** 2 * k1 ** 4 + 
                4608 * a12 ** 2 * a22 ** 2 * k1 ** 2 * k2 ** 2 + 576 * a22 ** 2 * a12 ** 2 * k2 ** 4 + 1792 * a12 ** 2 * k1 ** 6 + 
                7680 * a12 ** 2 * k1 ** 4 * k2 ** 2 + 4608 * k1 ** 2 * a12 ** 2 * k2 ** 4 + 256 * a12 ** 2 * k2 ** 6 + 35 * a21 ** 8 + 
                140 * a21 ** 6 * a22 ** 2 + 160 * a21 ** 6 * k1 ** 2 + 1120 * a21 ** 6 * k2 ** 2 + 210 * a21 ** 4 * a22 ** 4 + 
                480 * a21 ** 4 * a22 ** 2 * k1 ** 2 + 2400 * a21 ** 4 * a22 ** 2 * k2 ** 2 + 288 * a21 ** 4 * k1 ** 4 + 
                3840 * a21 ** 4 * k1 ** 2 * k2 ** 2 + 3360 * a21 ** 4 * k2 ** 4
            )
            
            part5 = (
                1536 * a22 ** 2 * k1 ** 4 * k2 ** 2 + 1536 * a22 ** 2 * k1 ** 2 * k2 ** 4 + 
                256 * a22 ** 2 * k2 ** 6 + 1792 * k1 ** 8 + 4608 * k1 ** 6 * k2 ** 2 + 
                4608 * k1 ** 4 * k2 ** 4 + 1792 * k1 ** 2 * k2 ** 6 + 512 * k2 ** 8
            )
            
            total += 35 / 4096 * np.pi ** 2 * (part1 + part2 + part3 + part4 + part5)

        if n >= 9:
            raise NotImplementedError("Not implemented for n > 8")

        return total

    def pdf(self, xs):
        assert xs.shape[-1] == 2
        xs = np.reshape(xs, (-1, self.input_dim))
        exponent = (self.kappa[0] * np.cos(xs[:, 0] - self.mu[0]) +
                    self.kappa[1] * np.cos(xs[:, 1] - self.mu[1]) +
                    np.cos(xs[:, 0] - self.mu[0]) * self.A[0, 0] * np.cos(xs[:, 1] - self.mu[1]) +
                    np.cos(xs[:, 0] - self.mu[0]) * self.A[0, 1] * np.sin(xs[:, 1] - self.mu[1]) +
                    np.sin(xs[:, 0] - self.mu[0]) * self.A[1, 0] * np.cos(xs[:, 1] - self.mu[1]) +
                    np.sin(xs[:, 0] - self.mu[0]) * self.A[1, 1] * np.sin(xs[:, 1] - self.mu[1]))

        p = self.C * np.exp(exponent)
        return p
    
    def multiply(self, tvm2):
        assert isinstance(tvm2, ToroidalVonMisesMatrixDistribution)

        C1 = self.kappa[0] * np.cos(self.mu[0]) + tvm2.kappa[0] * np.cos(tvm2.mu[0])
        S1 = self.kappa[0] * np.sin(self.mu[0]) + tvm2.kappa[0] * np.sin(tvm2.mu[0])
        C2 = self.kappa[1] * np.cos(self.mu[1]) + tvm2.kappa[1] * np.cos(tvm2.mu[1])
        S2 = self.kappa[1] * np.sin(self.mu[1]) + tvm2.kappa[1] * np.sin(tvm2.mu[1])
        mu_ = np.array([np.mod(np.arctan2(S1, C1), 2 * np.pi), np.mod(np.arctan2(S2, C2), 2 * np.pi)])
        kappa_ = np.array([np.sqrt(C1**2 + S1**2), np.sqrt(C2**2 + S2**2)])

        def M(mu):
            return np.array([
                [np.cos(mu[0]) * np.cos(mu[1]), -np.sin(mu[0]) * np.cos(mu[1]), -np.cos(mu[0]) * np.sin(mu[1]), np.sin(mu[0]) * np.sin(mu[1])],
                [np.sin(mu[0]) * np.cos(mu[1]),  np.cos(mu[0]) * np.cos(mu[1]), -np.sin(mu[0]) * np.sin(mu[1]), -np.cos(mu[0]) * np.sin(mu[1])],
                [np.cos(mu[0]) * np.sin(mu[1]), -np.sin(mu[0]) * np.sin(mu[1]),  np.cos(mu[0]) * np.cos(mu[1]), -np.sin(mu[0]) * np.cos(mu[1])],
                [np.sin(mu[0]) * np.sin(mu[1]),  np.cos(mu[0]) * np.sin(mu[1]),  np.sin(mu[0]) * np.cos(mu[1]),  np.cos(mu[0]) * np.cos(mu[1])]
            ])

        b = M(self.mu) @ np.array([self.A[0, 0], self.A[1, 0], self.A[0, 1], self.A[1, 1]]) + M(tvm2.mu) @ np.array([tvm2.A[0, 0], tvm2.A[1, 0], tvm2.A[0, 1], tvm2.A[1, 1]])
        a = np.linalg.solve(M(mu_), b)
        A_ = np.array([[a[0], a[2]], [a[1], a[3]]])

        return ToroidalVonMisesMatrixDistribution(mu_, kappa_, A_)
    
    def marginalize_to_1D(self, dimension):
        assert dimension == 1 or dimension == 2

        other = 3 - dimension
        alpha = lambda x: self.kappa[other - 1] + np.cos(x - self.mu[dimension - 1]) * self.A[0, 0] + np.sin(
            x - self.mu[dimension - 1]) * self.A[1, 0]
        beta = lambda x: np.sin(x - self.mu[dimension - 1]) * self.A[1, 1] + np.cos(
            x - self.mu[dimension - 1]) * self.A[0, 1]
        f = lambda x: 2 * np.pi * self.C * iv(0, np.sqrt(alpha(x) ** 2 + beta(x) ** 2)) * np.exp(
            self.kappa[dimension - 1] * np.cos(x - self.mu[dimension - 1]))

        return CustomCircularDistribution(f)

    def shift(self, shift_angles):
        assert np.all(np.array(shift_angles.shape) == np.array([self.dim, 1]))

        tvm = ToroidalVonMisesMatrixDistribution(self.mu, self.kappa, self.A)
        tvm.mu = np.mod(self.mu + shift_angles, 2 * np.pi)
        return tvm