import numpy as np
import pytest

from pyrecest.experimental.dvs.event_likelihood import (
    ContourSample,
    EventLikelihoodConfig,
    PointProcessUpdateConfig,
    contour_event_intensity,
    event_batch_log_likelihood,
    expected_event_count,
    normal_flow_activities,
)


def _rectangle_contour(width: float, height: float) -> ContourSample:
    half_width = 0.5 * width
    half_height = 0.5 * height
    return ContourSample(
        points=np.array(
            [
                [-half_width, 0.0],
                [half_width, 0.0],
                [0.0, half_height],
                [0.0, -half_height],
            ],
            dtype=float,
        ),
        normals=np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=float,
        ),
        weights=np.array([height, height, width, width], dtype=float),
    )


def test_contour_sample_converts_sequence_inputs_to_arrays():
    contour = ContourSample(points=[[0, 0], [1, 0]], normals=[[0, 1], [0, 1]], weights=[1, 2], angles=[0, np.pi])

    assert isinstance(contour.points, np.ndarray)
    assert isinstance(contour.normals, np.ndarray)
    assert isinstance(contour.weights, np.ndarray)
    assert isinstance(contour.angles, np.ndarray)
    assert contour.points.dtype.kind == "f"
    assert contour.normals.dtype.kind == "f"
    assert contour.weights.dtype.kind == "f"
    assert contour.angles.dtype.kind == "f"
    np.testing.assert_allclose(contour.points, np.array([[0.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_allclose(contour.normals, np.array([[0.0, 1.0], [0.0, 1.0]]))
    np.testing.assert_allclose(contour.weights, np.array([1.0, 2.0]))
    np.testing.assert_allclose(contour.angles, np.array([0.0, np.pi]))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("spatial_sigma_px", float("nan")),
        ("spatial_sigma_px", float("inf")),
        ("foreground_rate", float("nan")),
        ("background_rate", float("inf")),
        ("activity_floor", float("nan")),
        ("min_intensity", float("inf")),
        ("batch_duration", float("nan")),
    ],
)
def test_event_likelihood_config_rejects_nonfinite_numeric_values(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        EventLikelihoodConfig(**{keyword: value})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("finite_difference_eps", float("nan")),
        ("finite_difference_eps", float("inf")),
        ("map_step_size", float("nan")),
        ("covariance_damping", float("inf")),
        ("max_state_update_norm", float("nan")),
    ],
)
def test_point_process_update_config_rejects_nonfinite_numeric_values(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        PointProcessUpdateConfig(**{keyword: value})


def test_horizontal_motion_activates_only_vertical_sides():
    contour = _rectangle_contour(width=4.0, height=2.0)

    activities = normal_flow_activities(contour.normals, np.array([1.0, 0.0]))

    np.testing.assert_allclose(activities, np.array([1.0, 1.0, 0.0, 0.0]))


def test_activity_floor_applies_without_motion():
    contour = _rectangle_contour(width=4.0, height=2.0)

    activities = normal_flow_activities(contour.normals, np.zeros(2, dtype=float), activity_floor=0.05)

    np.testing.assert_allclose(activities, np.full(4, 0.05))


def test_active_edge_events_have_higher_intensity():
    contour = _rectangle_contour(width=4.0, height=2.0)
    config = EventLikelihoodConfig(spatial_sigma_px=0.4, foreground_rate=10.0, background_rate=1e-3)

    intensities = contour_event_intensity(
        np.array(
            [
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        contour,
        np.array([1.0, 0.0]),
        config,
    )

    assert intensities[0] > 100.0 * intensities[1]


def test_inactive_side_length_does_not_change_expected_count():
    config = EventLikelihoodConfig(foreground_rate=2.0, background_rate=0.0)
    narrow = _rectangle_contour(width=1.0, height=2.0)
    wide = _rectangle_contour(width=8.0, height=2.0)

    narrow_foreground, _ = expected_event_count(narrow, np.array([1.0, 0.0]), config)
    wide_foreground, _ = expected_event_count(wide, np.array([1.0, 0.0]), config)

    assert narrow_foreground == pytest.approx(wide_foreground)


def test_likelihood_prefers_correct_width_for_two_sided_support():
    config = EventLikelihoodConfig(
        spatial_sigma_px=0.5,
        foreground_rate=10.0,
        background_rate=1e-3,
        include_expected_count=False,
    )
    events = np.array(
        [
            [-2.0, -0.2],
            [-2.0, 0.2],
            [2.0, -0.2],
            [2.0, 0.2],
        ],
        dtype=float,
    )

    correct = event_batch_log_likelihood(events, _rectangle_contour(width=4.0, height=2.0), np.array([1.0, 0.0]), config)
    collapsed = event_batch_log_likelihood(events, _rectangle_contour(width=1.0, height=2.0), np.array([1.0, 0.0]), config)

    assert correct > collapsed
