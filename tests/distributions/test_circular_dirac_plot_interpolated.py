import pytest
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)


def test_plot_interpolated_without_arguments_raises_supported_error():
    distribution = CircularDiracDistribution([0.0])

    with pytest.raises(
        NotImplementedError,
        match="Interpolation is not available for CircularDiracDistribution",
    ):
        distribution.plot_interpolated()
