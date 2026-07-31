import math

import numpy as np
import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.utils import pairwise_covariance_shape_components


def test_shape_components_preserve_extreme_finite_covariances():
    largest = np.finfo(np.float64).max
    covariance_along_first_axis = array(
        [
            [[largest], [0.0]],
            [[0.0], [0.0]],
        ]
    )
    covariance_along_second_axis = array(
        [
            [[0.0], [0.0]],
            [[0.0], [largest]],
        ]
    )

    shape_cost, logdet_cost, shape_similarity = (
        pairwise_covariance_shape_components(
            covariance_along_first_axis,
            covariance_along_second_axis,
        )
    )

    npt.assert_allclose(shape_cost, array([[1.0]]))
    npt.assert_allclose(logdet_cost, array([[0.0]]))
    npt.assert_allclose(shape_similarity, array([[math.exp(-1.0)]]))
