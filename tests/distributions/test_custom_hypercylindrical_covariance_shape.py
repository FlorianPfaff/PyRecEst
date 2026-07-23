from unittest.mock import patch

import numpy.testing as npt
from pyrecest.backend import array, ones, pi, to_numpy
from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import (
    CustomHypercylindricalDistribution,
)


def _constant_pdf(xs):
    xs = array(xs)
    return ones(1 if xs.ndim == 1 else xs.shape[0])


def test_custom_numerical_covariance_preserves_matrix_shape_for_boundaries():
    distribution = CustomHypercylindricalDistribution(_constant_pdf, 1, 1)
    quadrature_path = (
        "pyrecest.distributions.cart_prod."
        "abstract_hypercylindrical_distribution.nquad"
    )

    with (
        patch.object(distribution, "linear_mean_numerical", return_value=array([0.0])),
        patch.object(distribution, "mode", return_value=array([0.0, 2.0])),
        patch(quadrature_path, return_value=(4.0, 0.0)),
    ):
        covariance = distribution.linear_covariance()
        boundaries = distribution.get_reasonable_integration_boundaries(scalingFactor=3)

    assert covariance.shape == (1, 1)
    npt.assert_allclose(to_numpy(covariance), [[4.0]])
    npt.assert_allclose(
        to_numpy(array(boundaries)),
        [[0.0, 2.0 * pi], [-4.0, 8.0]],
    )
