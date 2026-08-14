"""Regression tests for masked mixture weights."""

import numpy as np
import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


def _components():
    return [
        GaussianDistribution(array([0.0]), array([[1.0]])),
        GaussianDistribution(array([1.0]), array([[1.0]])),
    ]


@pytest.mark.parametrize(
    "weights",
    [
        np.ma.masked,
        np.ma.array([0.25, 0.75], mask=[False, True]),
    ],
)
def test_mixture_rejects_masked_weights(weights):
    with pytest.raises(ValueError, match="masked"):
        GaussianMixture(_components(), weights)


def test_unmasked_masked_array_weights_remain_supported():
    mixture = GaussianMixture(_components(), np.ma.array([0.25, 0.75], mask=False))

    np.testing.assert_allclose(to_numpy(mixture.w), [0.25, 0.75])
