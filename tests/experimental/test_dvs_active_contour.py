import numpy as np
from pyrecest.experimental.dvs import (
    activity_profile,
    normal_flow_activity,
    rectangle_contour_samples,
)


def test_normal_flow_activity_uses_normal_component():
    assert normal_flow_activity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == 1.0
    assert normal_flow_activity(np.array([0.0, 1.0]), np.array([2.0, 0.0])) == 0.0


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
