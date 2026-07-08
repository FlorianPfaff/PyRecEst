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
    scgp_event_batch_log_likelihood_terms,
)


class _DummySCGPTracker:
    def __init__(self, contour: ContourSample):
        self.contour = contour
        self.last_sample_count = None

    def sample_contour(self, n=100):
        self.last_sample_count = int(n)
        return self.contour


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
    contour = ContourSample(
        points=[[0, 0], [1, 0]],
        normals=[[0, 1], [0, 1]],
        weights=[1, 2],
        angles=[0, np.pi],
    )

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
        ("points", np.array([[0.0, 0.0], [np.nan, 0.0]])),
        ("normals", np.array([[0.0, 1.0], [np.inf, 0.0]])),
        ("weights", np.array([1.0, np.nan])),
        ("angles", np.array([0.0, -np.inf])),
    ],
)
def test_contour_sample_rejects_nonfinite_geometry(keyword, value):
    kwargs = {
        "points": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "normals": np.array([[0.0, 1.0], [0.0, 1.0]]),
        "weights": np.array([1.0, 2.0]),
        "angles": np.array([0.0, np.pi]),
    }
    kwargs[keyword] = value

    with pytest.raises(ValueError, match=f"{keyword} must contain only finite values"):
        ContourSample(**kwargs)


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
        ("spatial_sigma_px", "1.0"),
        ("foreground_rate", True),
        ("background_rate", b"0.1"),
        ("activity_floor", np.array(True, dtype=object)),
        ("min_intensity", "1e-12"),
        ("batch_duration", np.array("1.0")),
    ],
)
def test_event_likelihood_config_rejects_text_and_boolean_scalars(keyword, value):
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


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("finite_difference_eps", "0.01"),
        ("map_step_size", True),
        ("covariance_damping", "0.98"),
        ("max_state_update_norm", np.array(True, dtype=object)),
    ],
)
def test_point_process_update_config_rejects_text_and_boolean_scalars(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        PointProcessUpdateConfig(**{keyword: value})


def test_horizontal_motion_activates_only_vertical_sides():
    contour = _rectangle_contour(width=4.0, height=2.0)

    activities = normal_flow_activities(contour.normals, np.array([1.0, 0.0]))

    np.testing.assert_allclose(activities, np.array([1.0, 1.0, 0.0, 0.0]))


def test_activity_floor_applies_without_motion():
    contour = _rectangle_contour(width=4.0, height=2.0)

    activities = normal_flow_activities(
        contour.normals, np.zeros(2, dtype=float), activity_floor=0.05
    )

    np.testing.assert_allclose(activities, np.full(4, 0.05))


def test_normal_flow_activities_rejects_nonfinite_inputs():
    normals = np.array([[1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="normals must contain only finite values"):
        normal_flow_activities(np.array([[np.nan, 0.0]]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="velocity must contain only finite values"):
        normal_flow_activities(normals, np.array([np.inf, 0.0]))
    with pytest.raises(ValueError, match="activity_floor"):
        normal_flow_activities(normals, np.array([0.0, 0.0]), activity_floor=np.nan)


@pytest.mark.parametrize(
    "event_xy",
    [
        np.array([[np.nan, 0.0]]),
        np.array([[0.0, np.inf]]),
        np.array([[0.0, -np.inf]]),
    ],
)
def test_event_likelihood_rejects_nonfinite_event_coordinates(event_xy):
    contour = _rectangle_contour(width=1.0, height=2.0)
    velocity = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="event_xy must contain only finite values"):
        contour_event_intensity(event_xy, contour, velocity)
    with pytest.raises(ValueError, match="event_xy must contain only finite values"):
        event_batch_log_likelihood(event_xy, contour, velocity)


def test_active_edge_events_have_higher_intensity():
    contour = _rectangle_contour(width=4.0, height=2.0)
    config = EventLikelihoodConfig(
        spatial_sigma_px=0.4, foreground_rate=10.0, background_rate=1e-3
    )

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


@pytest.mark.parametrize("image_area", [np.nan, np.inf, -1.0, "4.0", True])
def test_expected_event_count_rejects_invalid_image_area(image_area):
    contour = _rectangle_contour(width=1.0, height=2.0)

    with pytest.raises(ValueError, match="image_area"):
        expected_event_count(contour, np.array([1.0, 0.0]), image_area=image_area)


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

    correct = event_batch_log_likelihood(
        events, _rectangle_contour(width=4.0, height=2.0), np.array([1.0, 0.0]), config
    )
    collapsed = event_batch_log_likelihood(
        events, _rectangle_contour(width=1.0, height=2.0), np.array([1.0, 0.0]), config
    )

    assert correct > collapsed


def test_scgp_event_batch_log_likelihood_terms_uses_tracker_contour_sampler():
    contour = _rectangle_contour(width=4.0, height=2.0)
    tracker = _DummySCGPTracker(contour)
    events = np.array(
        [
            [-2.0, -0.2],
            [2.0, 0.2],
        ],
        dtype=float,
    )
    update_config = PointProcessUpdateConfig(
        likelihood=EventLikelihoodConfig(
            spatial_sigma_px=0.5,
            foreground_rate=10.0,
            background_rate=1e-3,
            include_expected_count=False,
        ),
        contour_samples=17,
    )

    terms = scgp_event_batch_log_likelihood_terms(
        tracker,
        events,
        np.array([1.0, 0.0]),
        update_config,
    )

    assert tracker.last_sample_count == 17
    assert terms.log_likelihood == pytest.approx(
        event_batch_log_likelihood(
            events, contour, np.array([1.0, 0.0]), update_config.likelihood
        )
    )
