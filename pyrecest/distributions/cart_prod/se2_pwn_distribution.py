# pylint: disable=no-name-in-module,no-member
from ..se2_partially_wrapped_normal_distribution import SE2PartiallyWrappedNormalDistribution


class SE2PWNDistribution(SE2PartiallyWrappedNormalDistribution):
    """Partially wrapped normal distribution for SE(2).

    The first component is the rotation angle (periodic), the second and
    third components are the 2-D translation (linear).

    Based on:
        Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
        "The Partially Wrapped Normal Distribution for SE(2) Estimation",
        Proc. IEEE MFI 2014.

    This class is a backward-compatible alias for SE2PartiallyWrappedNormalDistribution.
    """

    def mean4D(self):
        """Return the 4-D moment E[cos(x1), sin(x1), x2, x3].

        Backward-compatible alias for mean_4d().
        """
        return self.mean_4d()

    def covariance4D(self):
        """Return the analytical 4-D covariance of [cos(x1), sin(x1), x2, x3].

        Backward-compatible alias for covariance_4d().
        """
        return self.covariance_4d()

    def covariance4D_numerical(self, n_samples=10000):
        """Estimate the 4-D covariance numerically.

        Backward-compatible alias for covariance_4d_numerical().
        """
        return self.covariance_4d_numerical(n_samples)

    @staticmethod
    def from_samples(samples):
        """Fit an SE2PWNDistribution from samples via moment matching."""
        result = SE2PartiallyWrappedNormalDistribution.from_samples(samples)
        return SE2PWNDistribution(result.mu, result.C)
