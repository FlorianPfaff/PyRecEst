import pytest
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


def _components():
    return [
        GaussianDistribution([0.0], [[1.0]], check_validity=False),
        GaussianDistribution([1.0], [[1.0]], check_validity=False),
    ]


def test_gaussian_mixture_rejects_matrix_weights():
    with pytest.raises(ValueError, match="one-dimensional"):
        GaussianMixture(_components(), [[0.25, 0.75]])


def test_gaussian_mixture_rejects_boolean_weights():
    with pytest.raises(ValueError, match="not boolean"):
        GaussianMixture(_components(), [True, False])


def test_gaussian_mixture_rejects_complex_weights():
    with pytest.raises(ValueError, match="real-valued numeric"):
        GaussianMixture(_components(), [0.5 + 0.0j, 0.5])


def test_gaussian_mixture_rejects_text_weights():
    with pytest.raises(ValueError, match="real-valued numeric"):
        GaussianMixture(_components(), ["0.5", 0.5])
