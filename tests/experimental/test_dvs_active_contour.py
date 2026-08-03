import numpy as np
import pytest
from pyrecest.experimental.dvs import (
    activity_profile,
    normal_flow_activity,
    rectangle_contour_samples,
    signed_normal_flow,
    signed_normal_flow_profile,
)


def test_normal_flow_activity_uses_normal_component():
    assert normal_flow_activity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
    assert normal_flow_activity(np.array([0.0, 1.0]), np.array([2.0, 0.0])) == 0.0


def test_signed_normal_flow_preserves_extreme_finite_directions():
    velocity = np.array([1e308, 1e308])
    normals = np.array(
        [
            [1e308, 0.0],
            [0.0, 1e308],
            [-1e308, 0.0],
        ]
    )
    expected = np.sqrt(0.5)

    with np.errstate(over="raise", invalid="raise"):
        scalar_flow = signed_normal_flow(normals[0], velocity)
        profile = signed_normal_flow_profile(normals, velocity)

    assert scalar_flow == pytest.approx(expected)
    np.testing.assert_allclose(profile, [expected, expected, -expected])


def test_rectangle_activity_matches_horizontal_translation():
    contour = rectangle_contour_samples(samples_per_edge=4)
    activities = activity_profile(contour.normals, np.array([1.0, 0.0]))
    by_edge = {
        edge: float(np.mean(activities[np.array(contour.edge_labels) == edge]))
        for edge in set(contour.edge_labels)
    }

    assert by_edge["left"] == 1.0
    assert by_edge["right"] == 1.0
    assert by_edge["top"] == 0.0
    assert by_edge["bottom"] == 0.0


def test_rectangle_contour_covers_each_corner_once():
    contour = rectangle_contour_samples(width=2.0, height=1.0, samples_per_edge=4)
    corners = np.array(
        [
            [-1.0, 0.5],
            [1.0, 0.5],
            [1.0, -0.5],
            [-1.0, -0.5],
        ]
    )

    assert np.unique(contour.points, axis=0).shape == contour.points.shape
    for corner in corners:
        matches = np.all(np.isclose(contour.points, corner), axis=1)
        assert np.count_nonzero(matches) == 1
