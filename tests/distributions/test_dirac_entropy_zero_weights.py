"""Regression coverage for zero-weight Dirac entropy terms."""

import warnings

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution


def test_dirac_entropy_treats_zero_log_zero_as_zero():
    distribution = LinearDiracDistribution(
        array([[0.0], [1.0], [2.0]]),
        array([0.25, 0.75, 0.0]),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entropy = distribution.entropy()

    expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    npt.assert_allclose(entropy, expected)
    assert len(caught) == 1
    assert "not defined in a continuous sense" in str(caught[0].message)
