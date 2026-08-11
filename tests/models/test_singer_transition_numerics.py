"""Regression tests for numerically stable Singer transition matrices."""

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.models import singer_transition_matrix
from scipy.linalg import expm


@pytest.mark.parametrize(
    ("dt", "tau"),
    [
        (1.0, 5.0),
        (1.0, 1.0e12),
        (0.25, 1.0e10),
        (0.0, 1.0e12),
        (-0.5, 1.0e9),
    ],
)
def test_singer_transition_matches_matrix_exponential(dt, tau):
    alpha = 1.0 / tau
    generator = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -alpha],
        ]
    )
    expected = expm(generator * dt)

    actual = np.asarray(
        singer_transition_matrix(dt, spatial_dim=1, tau=tau),
        dtype=float,
    )

    npt.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-14)
