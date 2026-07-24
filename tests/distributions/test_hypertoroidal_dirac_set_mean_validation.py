import numpy.testing as npt
import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, zeros_like
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
)


def _three_dimensional_distribution():
    samples = array([[0.5, 2.0, 0.5], [3.0, 2.0, 0.2], [4.0, 5.0, 5.8]])
    weights = array([0.2, 0.3, 0.5])
    return HypertoroidalDiracDistribution(samples, weights), samples


def test_set_mean_accepts_matching_vector_and_preserves_original():
    dist, original_samples = _three_dimensional_distribution()
    target_mean = array([0.25, 1.0, 2.5])

    shifted = dist.set_mean(target_mean)

    npt.assert_allclose(
        AbstractHypertoroidalDistribution.angular_error(
            shifted.mean_direction(), target_mean
        ),
        zeros_like(target_mean),
        atol=1e-6,
    )
    npt.assert_allclose(dist.d, original_samples)


def test_set_mean_rejects_broadcasting_mean_shapes_for_multidimensional_distribution():
    dist, original_samples = _three_dimensional_distribution()

    for malformed_mean in (0.25, [0.25], [0.25, 1.0], [[0.25, 1.0, 2.5]]):
        with pytest.raises(ValueError, match=r"mean must have shape \(3,\)"):
            dist.set_mean(malformed_mean)
        npt.assert_allclose(dist.d, original_samples)


def test_set_mean_accepts_scalar_for_one_dimensional_distribution():
    dist = HypertoroidalDiracDistribution([0.1, 0.2, 0.3], dim=1)

    shifted = dist.set_mean(0.25)

    npt.assert_allclose(shifted.mean_direction(), array([0.25]), atol=1e-6)
