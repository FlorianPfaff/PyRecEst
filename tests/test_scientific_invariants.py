import numpy as np
from pyrecest.backend import array, diag, pi, to_numpy
from pyrecest.distributions import CircularUniformDistribution, GaussianDistribution


def _as_numpy(value):
    value = to_numpy(value)
    return np.asarray(value, dtype=float)


def test_gaussian_product_covariance_is_symmetric_positive_definite():
    first = GaussianDistribution(array([0.0, 1.0]), diag(array([4.0, 1.0])))
    second = GaussianDistribution(array([1.0, -0.5]), diag(array([1.0, 2.25])))

    product = first.multiply(second)
    covariance = _as_numpy(product.C)

    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_circular_uniform_integrates_to_one_and_has_constant_density():
    distribution = CircularUniformDistribution()

    assert abs(float(distribution.integrate()) - 1.0) < 1e-12

    xs = array([0.0, float(pi) / 2.0, float(pi)])
    values = _as_numpy(distribution.pdf(xs))
    expected = np.full(values.shape, 1.0 / (2.0 * np.pi))

    np.testing.assert_allclose(values, expected, atol=1e-12)
