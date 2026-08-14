# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import column_stack, cos, sin, sqrt, sum
from pyrecest.backend import max as backend_max

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class ToroidalDiracDistribution(
    HypertoroidalDiracDistribution, AbstractToroidalDistribution
):
    def __init__(self, d, w):
        """
        Initialize ToroidalDiracDistribution.

        :param d: Array of positions.
        :param w: Array of weights.
        """
        AbstractToroidalDistribution.__init__(self)
        HypertoroidalDiracDistribution.__init__(self, d, w)

    def circular_correlation_jammalamadaka(self) -> float:
        """
        Calculate the circular correlation according to Jammalamadaka's definition

        :returns: Correlation coefficient.
        """
        m = self.mean_direction()

        first_sines = sin(self.d[:, 0] - m[0])
        second_sines = sin(self.d[:, 1] - m[1])

        first_scale = backend_max(backend_abs(first_sines))
        second_scale = backend_max(backend_abs(second_sines))
        if bool(first_scale == 0.0) or bool(second_scale == 0.0):
            raise ValueError(
                "Circular correlation is undefined for zero circular variance."
            )

        first_sines = first_sines / first_scale
        second_sines = second_sines / second_scale
        first_energy = sum(self.w * first_sines**2)
        second_energy = sum(self.w * second_sines**2)
        if bool(first_energy == 0.0) or bool(second_energy == 0.0):
            raise ValueError(
                "Circular correlation is undefined for zero circular variance."
            )

        numerator = sum(self.w * first_sines * second_sines)
        denominator = sqrt(first_energy * second_energy)
        return numerator / denominator

    def covariance_4D(self):
        """
        Compute the 4D covariance matrix.

        :returns: 4D covariance matrix.
        """
        dbar = column_stack(
            [
                cos(self.d[:, 0]),
                sin(self.d[:, 0]),
                cos(self.d[:, 1]),
                sin(self.d[:, 1]),
            ]
        )
        mu = sum(self.w[:, None] * dbar, axis=0)
        centered = dbar - mu
        C = centered.T @ (self.w[:, None] * centered)
        return C
