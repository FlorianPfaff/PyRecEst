"""Regression tests for extreme finite DVS normal-flow magnitudes."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pyrecest.experimental.dvs.event_likelihood as event_likelihood_module
from pyrecest.experimental.dvs import normal_flow_activities


def test_normal_flow_activities_preserves_extreme_finite_directions() -> None:
    magnitude = 1.0e308
    normals = np.array(
        [
            [magnitude, 0.0],
            [0.0, magnitude],
            [magnitude, magnitude],
            [-magnitude, 0.0],
        ]
    )
    velocity = np.array([magnitude, magnitude])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        activities = normal_flow_activities(normals, velocity)

    root_half = 1.0 / np.sqrt(2.0)
    npt.assert_allclose(
        activities,
        np.array([root_half, root_half, 1.0, root_half]),
        rtol=1.0e-15,
        atol=0.0,
    )
    assert normal_flow_activities is event_likelihood_module.normal_flow_activities
