"""Regression tests for extreme finite scalar DVS normal-flow inputs."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pyrecest.experimental.dvs.active_contour import signed_normal_flow


def test_signed_normal_flow_preserves_extreme_finite_directions() -> None:
    magnitude = np.finfo(np.float64).max
    normal = np.array([magnitude, magnitude])
    velocity = np.array([magnitude, 0.0])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        flow = signed_normal_flow(normal, velocity)

    npt.assert_allclose(flow, 1.0 / np.sqrt(2.0), rtol=1.0e-15, atol=0.0)
