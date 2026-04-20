# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    arctan2,
    array,
    column_stack,
    cos,
    cov,
    exp,
    log,
    mean,
    mod,
    pi,
    sin,
    sqrt,
)

from .abstract_se2_distribution import AbstractSE2Distribution
from .cart_prod.partially_wrapped_normal_distribution import PartiallyWrappedNormalDistribution


class SE2PartiallyWrappedNormalDistribution(PartiallyWrappedNormalDistribution, AbstractSE2Distribution):
    """Partially wrapped normal distribution for SE(2).

    The first component is the rotation angle (periodic, wrapped), the second
    and third components are the 2-D translation (linear).

    Reference:
        Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
        "The Partially Wrapped Normal Distribution for SE(2) Estimation",
        Proc. IEEE MFI 2014.
    """

    def __init__(self, mu, C):
        AbstractSE2Distribution.__init__(self)
        PartiallyWrappedNormalDistribution.__init__(self, mu, C, bound_dim=self.bound_dim)

    def mean_4d(self):
        """Return the 4-D moment E[cos(x1), sin(x1), x2, x3].

        Returns
        -------
        array of shape (4,)
        """
        return self.hybrid_moment()

    def covariance_4d(self):  # pylint: disable=too-many-locals
        """Return the analytical 4-D covariance of [cos(x1), sin(x1), x2, x3].

        Based on the formula from:
            Gerhard Kurz, "Directional Estimation for Robotic Beating Heart Surgery",
            Karlsruhe Institute of Technology, 2015.

        Returns
        -------
        array of shape (4, 4)
        """
        mu0 = self.mu[0]
        c11 = self.C[0, 0]
        c12 = self.C[0, 1]
        c13 = self.C[0, 2]

        a = 1 - exp(-c11)
        half_a = 0.5 * a
        e1 = exp(-c11)
        e_half = exp(-0.5 * c11)

        si11 = half_a * (1 - e1 * cos(2 * mu0))
        si22 = half_a * (1 + e1 * cos(2 * mu0))
        si33 = self.C[1, 1]
        si44 = self.C[2, 2]

        si12 = -half_a * e1 * sin(2 * mu0)
        si13 = -e_half * c12 * sin(mu0)
        si23 = e_half * c12 * cos(mu0)
        si14 = -e_half * c13 * sin(mu0)
        si24 = e_half * c13 * cos(mu0)
        si34 = self.C[1, 2]

        return array(
            [
                [si11, si12, si13, si14],
                [si12, si22, si23, si24],
                [si13, si23, si33, si34],
                [si14, si24, si34, si44],
            ]
        )

    def covariance_4d_numerical(self, n_samples=10000):
        """Estimate the 4-D covariance of [cos(x1), sin(x1), x2, x3] from samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw for the estimate.

        Returns
        -------
        array of shape (4, 4)
        """
        s = self.sample(n_samples)
        big_s = column_stack([cos(s[:, 0]), sin(s[:, 0]), s[:, 1], s[:, 2]])
        return cov(big_s.T)

    @staticmethod
    def from_samples(samples):
        """Fit an SE2PartiallyWrappedNormalDistribution from samples via moment matching.

        Parameters
        ----------
        samples : array of shape (n, 3)
            Samples on [0, 2*pi) x R^2.

        Returns
        -------
        SE2PartiallyWrappedNormalDistribution
        """
        big_s = column_stack(
            [
                cos(samples[:, 0]),
                sin(samples[:, 0]),
                samples[:, 1],
                samples[:, 2],
            ]
        )
        mu4 = mean(big_s, axis=0)

        mu0 = mod(arctan2(mu4[1], mu4[0]), 2.0 * pi)
        m1abs = sqrt(mu4[0] ** 2 + mu4[1] ** 2)
        mu = array([mu0, mu4[2], mu4[3]])

        c4 = cov(big_s.T)

        c00 = -2.0 * log(m1abs)
        factor = exp(0.5 * c00)
        c01 = (-c4[0, 2] * sin(mu[0]) + c4[1, 2] * cos(mu[0])) * factor
        c02 = (-c4[0, 3] * sin(mu[0]) + c4[1, 3] * cos(mu[0])) * factor
        c = array(
            [
                [c00, c01, c02],
                [c01, c4[2, 2], c4[2, 3]],
                [c02, c4[3, 2], c4[3, 3]],
            ]
        )

        return SE2PartiallyWrappedNormalDistribution(mu, c)
