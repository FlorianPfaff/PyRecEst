# pylint: disable=protected-access
import numpy as np
import pytest
from pyrecest.backend import array, diag, eye
from pyrecest.filters.velocity_aided_mem_qkf_tracker import VelocityAidedMEMQKFTracker


def _make_tracker(**kwargs):
    config = {
        "kinematic_state": array([0.0, 0.0, 3.0, 4.0]),
        "covariance": diag(array([1.0, 1.0, 0.04, 0.04])),
        "shape_state": array([0.0, 2.0, 1.0]),
        "shape_covariance": diag(array([0.5, 0.2, 0.2])),
        "default_meas_noise_cov": 0.05 * eye(2),
        "heading_noise_variance": 0.25,
    }
    config.update(kwargs)
    return VelocityAidedMEMQKFTracker(**config)


@pytest.mark.parametrize(
    "parameter",
    (
        "speed_threshold",
        "orientation_offset",
        "heading_noise_variance",
        "minimum_heading_variance",
    ),
)
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_heading_scalar_controls_reject_nonfinite_values(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        _make_tracker(**{parameter: value})


@pytest.mark.parametrize(
    "parameter",
    ("apply_heading_on_prediction", "use_heading_constraint"),
)
@pytest.mark.parametrize("value", ("False", 0, 1, None, np.array(False)))
def test_heading_boolean_controls_reject_non_boolean_values(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        _make_tracker(**{parameter: value})


def test_heading_boolean_controls_accept_numpy_booleans():
    tracker = _make_tracker(
        apply_heading_on_prediction=np.bool_(False),
        use_heading_constraint=np.bool_(True),
    )

    assert tracker.apply_heading_on_prediction is False
    assert tracker.use_heading_constraint is True


def test_per_update_heading_override_rejects_non_boolean_without_mutating_state():
    tracker = _make_tracker()

    with pytest.raises(ValueError, match="use_heading_constraint"):
        tracker.update(
            array([[1.0, 0.2]]),
            use_heading_constraint="False",
        )

    assert tracker.use_heading_constraint is True
    assert tracker._heading_update_pending is False
