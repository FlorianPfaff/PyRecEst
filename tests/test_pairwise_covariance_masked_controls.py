import warnings

import numpy as np
import numpy.testing as npt
import pytest

from pyrecest.backend import array, zeros
from pyrecest.utils import (
    pairwise_covariance_shape_components,
    pairwise_mahalanobis_distances,
)


def test_pairwise_covariance_controls_reject_masked_scalars_before_coercion():
    means_a = array([[0.0], [0.0]])
    means_b = array([[2.0], [0.0]])
    covariance = zeros((2, 2, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="regularization.*masked"):
            pairwise_mahalanobis_distances(
                means_a,
                covariance,
                means_b,
                covariance,
                regularization=np.ma.array(1.0, mask=True),
            )

        with pytest.raises(ValueError, match="epsilon.*masked"):
            pairwise_covariance_shape_components(
                covariance,
                covariance,
                epsilon=np.ma.array(1.0e-6, mask=True),
            )


def test_pairwise_covariance_controls_preserve_clear_mask_wrappers():
    means_a = array([[0.0], [0.0]])
    means_b = array([[2.0], [0.0]])
    covariance = zeros((2, 2, 1))

    distances = pairwise_mahalanobis_distances(
        means_a,
        covariance,
        means_b,
        covariance,
        regularization=np.ma.array(1.0, mask=False),
    )
    npt.assert_allclose(distances, array([[2.0]]))

    shape_cost, logdet_cost, shape_similarity = (
        pairwise_covariance_shape_components(
            covariance,
            covariance,
            epsilon=np.ma.array(1.0e-6, mask=False),
        )
    )
    npt.assert_allclose(shape_cost, array([[0.0]]))
    npt.assert_allclose(logdet_cost, array([[0.0]]))
    npt.assert_allclose(shape_similarity, array([[1.0]]))
