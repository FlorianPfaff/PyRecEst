import pytest
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


def test_gaussian_mixture_rejects_matrix_weights():
    components = [
        GaussianDistribution([0.0], [[1.0]], check_validity=False),
        GaussianDistribution([1.0], [[1.0]], check_validity=False),
    ]

    with pytest.raises(ValueError, match="one-dimensional"):
        GaussianMixture(components, [[0.25, 0.75]])
