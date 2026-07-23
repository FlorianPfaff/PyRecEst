import pytest

from pyrecest.backend import array, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture


class _MalformedSamplingGaussian(GaussianDistribution):
    def sample(self, n):
        del n
        return zeros((1, self.dim))


def test_mixture_rejects_component_that_returns_too_few_samples():
    component = _MalformedSamplingGaussian(
        array([0.0, 0.0]),
        array([[1.0, 0.0], [0.0, 1.0]]),
    )
    mixture = LinearMixture([component], array([1.0]))

    with pytest.raises(
        ValueError,
        match=r"Mixture component sample output must have shape \(3, 2\), got \(1, 2\)",
    ):
        mixture.sample(3)


def test_mixture_accepts_component_sample_matrix_with_exact_shape():
    component = GaussianDistribution(
        array([0.0, 0.0]),
        array([[1.0, 0.0], [0.0, 1.0]]),
    )
    mixture = LinearMixture([component], array([1.0]))

    assert mixture.sample(3).shape == (3, 2)
