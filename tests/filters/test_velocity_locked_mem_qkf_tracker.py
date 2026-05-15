import numpy as np
from pyrecest.backend import array, diag, eye
from pyrecest.filters.velocity_locked_mem_qkf_tracker import (
    VelocityLockedMEMQKFTracker,
)


def _make_tracker(speed_threshold=1e-9, **kwargs):
    return VelocityLockedMEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 3.0, 4.0]),
        covariance=diag(array([1.0, 1.0, 0.04, 0.04])),
        shape_state=array([0.0, 2.0, 1.0]),
        shape_covariance=diag(array([0.5, 0.2, 0.2])),
        default_meas_noise_cov=0.05 * eye(2),
        speed_threshold=speed_threshold,
        **kwargs,
    )


def test_velocity_locked_orientation_is_initialized_from_heading():
    tracker = _make_tracker()

    assert np.isclose(float(tracker.shape_state[0]), np.arctan2(4.0, 3.0))
    assert float(tracker.shape_covariance[0, 0]) > 0.0


def test_velocity_locked_orientation_tracks_prediction_heading():
    tracker = _make_tracker()
    system_matrix = array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    tracker.predict_linear(system_matrix, sys_noise=0.0 * eye(4))

    expected_heading = np.arctan2(3.0, -4.0)
    assert np.isclose(float(tracker.shape_state[0]), expected_heading)


def test_velocity_locked_update_keeps_orientation_coupled_to_velocity():
    tracker = _make_tracker()
    measurements = array(
        [
            [1.0, 0.2],
            [-0.1, 0.7],
            [0.4, -0.2],
        ]
    )
    tracker.update(measurements)

    vx, vy = tracker.kinematic_state[2], tracker.kinematic_state[3]
    expected_heading = np.arctan2(float(vy), float(vx))
    assert np.isclose(float(tracker.shape_state[0]), expected_heading)
    assert tracker.shape_state[1] > 0.0
    assert tracker.shape_state[2] > 0.0


def test_velocity_locked_batch_update_keeps_orientation_coupled_to_velocity():
    tracker = _make_tracker(update_mode="batch")
    measurements = array(
        [
            [1.0, 0.2],
            [-0.1, 0.7],
            [0.4, -0.2],
        ]
    )
    tracker.update(measurements)

    vx, vy = tracker.kinematic_state[2], tracker.kinematic_state[3]
    expected_heading = np.arctan2(float(vy), float(vx))
    assert np.isclose(float(tracker.shape_state[0]), expected_heading)
    assert tracker.shape_state[1] > 0.0
    assert tracker.shape_state[2] > 0.0


def test_stationary_case_falls_back_to_standard_mem_qkf_update():
    tracker = VelocityLockedMEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 0.0, 0.0]),
        covariance=diag(array([1.0, 1.0, 0.04, 0.04])),
        shape_state=array([0.2, 2.0, 1.0]),
        shape_covariance=diag(array([0.5, 0.2, 0.2])),
        default_meas_noise_cov=0.05 * eye(2),
        speed_threshold=1e-3,
    )
    tracker.update(array([[1.0, 0.0], [0.0, 1.0]]))

    assert np.isfinite(float(tracker.shape_state[0]))
    assert tracker.shape_state[1] > 0.0
    assert tracker.shape_state[2] > 0.0
