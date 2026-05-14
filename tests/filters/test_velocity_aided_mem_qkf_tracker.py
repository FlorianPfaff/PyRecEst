import numpy as np
from pyrecest.backend import array, diag, eye
from pyrecest.filters.velocity_aided_mem_qkf_tracker import (
    VelocityAidedMEMQKFTracker,
    VelocityAidedMemQkfTracker,
)


def _axial_error(angle):
    return 0.5 * np.arctan2(np.sin(2.0 * angle), np.cos(2.0 * angle))


def _make_tracker(**kwargs):
    return VelocityAidedMEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 3.0, 4.0]),
        covariance=diag(array([1.0, 1.0, 0.04, 0.04])),
        shape_state=array([0.0, 2.0, 1.0]),
        shape_covariance=diag(array([0.5, 0.2, 0.2])),
        default_meas_noise_cov=0.05 * eye(2),
        heading_noise_variance=0.25,
        **kwargs,
    )


def test_velocity_aided_alias_points_to_tracker_class():
    assert VelocityAidedMemQkfTracker is VelocityAidedMEMQKFTracker


def test_heading_update_is_soft_not_locked():
    tracker = _make_tracker()
    initial_orientation = float(tracker.shape_state[0])
    heading = np.arctan2(4.0, 3.0)

    assert tracker._update_orientation_from_heading_state()

    posterior_orientation = float(tracker.shape_state[0])
    assert initial_orientation < posterior_orientation < heading
    assert not np.isclose(posterior_orientation, heading)
    assert float(tracker.shape_covariance[0, 0]) < 0.5


def test_heading_update_respects_axial_periodicity():
    tracker = VelocityAidedMEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 1.0, 0.0]),
        covariance=diag(array([1.0, 1.0, 0.01, 0.01])),
        shape_state=array([np.pi - 0.2, 2.0, 1.0]),
        shape_covariance=diag(array([0.5, 0.2, 0.2])),
        default_meas_noise_cov=0.05 * eye(2),
        heading_noise_variance=0.25,
    )
    before_error = abs(_axial_error(float(tracker.shape_state[0])))

    assert tracker._update_orientation_from_heading_state()

    after_error = abs(_axial_error(float(tracker.shape_state[0])))
    assert after_error < before_error
    assert float(tracker.shape_state[0]) > np.pi - 0.2


def test_stationary_heading_update_is_inactive():
    tracker = VelocityAidedMEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 0.0, 0.0]),
        covariance=diag(array([1.0, 1.0, 0.04, 0.04])),
        shape_state=array([0.2, 2.0, 1.0]),
        shape_covariance=diag(array([0.5, 0.2, 0.2])),
        default_meas_noise_cov=0.05 * eye(2),
        speed_threshold=1e-3,
    )
    before_orientation = float(tracker.shape_state[0])
    before_variance = float(tracker.shape_covariance[0, 0])

    assert not tracker._update_orientation_from_heading_state()

    assert np.isclose(float(tracker.shape_state[0]), before_orientation)
    assert np.isclose(float(tracker.shape_covariance[0, 0]), before_variance)


def test_update_fuses_heading_once_and_keeps_shape_valid():
    tracker = _make_tracker()
    measurements = array(
        [
            [1.0, 0.2],
            [-0.1, 0.7],
            [0.4, -0.2],
        ]
    )

    tracker.update(measurements)

    assert np.isfinite(float(tracker.shape_state[0]))
    assert tracker.shape_state[1] > 0.0
    assert tracker.shape_state[2] > 0.0
    assert not tracker._heading_update_pending
