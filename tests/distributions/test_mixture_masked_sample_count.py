"""Regression tests for masked mixture sample counts."""

import numpy as np
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions.abstract_mixture import _validate_positive_sample_count
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


def _make_mixture():
    component = ToroidalWrappedNormalDistribution(array([1.0, 0.0]), eye(2))
    return HypertoroidalMixture(
        [component, component.shift(array([1.0, 1.0]))],
        array([0.5, 0.5]),
    )


@pytest.mark.parametrize(
    "invalid_count",
    [np.ma.masked, np.ma.array(3, mask=True)],
)
def test_mixture_sample_rejects_masked_count(invalid_count):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        _make_mixture().sample(invalid_count)


def test_unmasked_numeric_masked_array_count_remains_supported():
    assert _validate_positive_sample_count(np.ma.array(3, mask=False)) == 3
