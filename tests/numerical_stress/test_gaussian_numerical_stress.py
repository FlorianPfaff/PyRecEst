import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution


@pytest.mark.numerical_stress
def test_gaussian_product_remains_symmetric_for_ill_conditioned_covariances():
    first = GaussianDistribution(
        array([0.0, 0.0]),
        array([[1.0e-8, 0.0], [0.0, 1.0]]),
        check_validity=False,
    )
    second = GaussianDistribution(
        array([1.0, -1.0]),
        array([[2.0e-8, 0.0], [0.0, 2.0]]),
        check_validity=False,
    )

    product = first.multiply(second)
    covariance = to_numpy(product.C)

    npt.assert_allclose(covariance, covariance.T, rtol=1e-12, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
