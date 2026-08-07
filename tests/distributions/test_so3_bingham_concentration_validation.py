"""Validation regressions for SO(3) Bingham concentration controls."""

import numpy as np
import pytest

from pyrecest.backend import array
from pyrecest.distributions import SO3BinghamDistribution


@pytest.mark.parametrize(
    "invalid_concentration",
    [
        True,
        np.bool_(False),
        np.complex64(2.0 + 3.0j),
        np.asarray(2.0 + 0.0j),
        np.asarray([2.0]),
        np.ma.array(2.0, mask=True),
        np.datetime64("2026-08-04"),
        "2.0",
    ],
)
def test_from_mode_rejects_lossy_concentration_coercions(invalid_concentration):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SO3BinghamDistribution.from_mode_and_concentration(
            array([0.0, 0.0, 0.0, 1.0]),
            invalid_concentration,
        )


def test_from_mode_accepts_numpy_real_scalar_concentration():
    distribution = SO3BinghamDistribution.from_mode_and_concentration(
        array([0.0, 0.0, 0.0, 1.0]),
        np.float64(2.0),
    )

    assert distribution.is_valid()
