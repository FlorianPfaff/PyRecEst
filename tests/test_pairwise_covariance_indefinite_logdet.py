import math

import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.utils import pairwise_covariance_shape_components


def test_pairwise_covariance_shape_components_floor_indefinite_logdet():
    epsilon = 1.0e-6
    negative_definite = array(
        [
            [[-2.0], [0.0]],
            [[0.0], [-3.0]],
        ]
    )
    positive_definite = array(
        [
            [[2.0], [0.0]],
            [[0.0], [3.0]],
        ]
    )

    _, logdet_cost, _ = pairwise_covariance_shape_components(
        negative_definite,
        positive_definite,
        epsilon=epsilon,
    )

    expected = math.log(6.0) - math.log(epsilon)
    npt.assert_allclose(logdet_cost, array([[expected]]))
