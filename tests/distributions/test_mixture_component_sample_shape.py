import pytest
from pyrecest.backend import array, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture


class _MalformedSamplingGaussian(GaussianDistribution):
    def sample(self, n):
        # Non-JAX mixture sampling requests all samples from a selected component
        # in one batch, while the JAX path requests one sample at a time. Violate
        # both contracts so this regression test exercises the shape validation
        # consistently across backends.
        returned_count = 0 if n == 1 else 1
        return zeros((returned_count, self.dim))


def test_mixture_rejects_component_that_returns_too_few_samples():
    component = _MalformedSamplingGaussian(
        array([0.0, 0.0]),
        array([[1.0, 0.0], [0.0, 1.0]]),
    )
    mixture = LinearMixture([component], array([1.0]))

    with pytest.raises(
        ValueError,
        match=r"Mixture component sample output must have shape",
    ):
        mixture.sample(3)


def test_mixture_accepts_component_sample_matrix_with_exact_shape():
    component = GaussianDistribution(
        array([0.0, 0.0]),
        array([[1.0, 0.0], [0.0, 1.0]]),
    )
    mixture = LinearMixture([component], array([1.0]))

    assert mixture.sample(3).shape == (3, 2)
