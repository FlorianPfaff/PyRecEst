from __future__ import annotations

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.backend import array, diag, eye
from pyrecest.filters.mem_qkf_tracker import MEMQKFTracker
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker
from pyrecest.filters.vbrm_tracker import VBRMTracker
from pyrecest.filters.velocity_aided_mem_qkf_tracker import VelocityAidedMEMQKFTracker
from pyrecest.filters.velocity_locked_mem_qkf_tracker import VelocityLockedMEMQKFTracker

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Empty-measurement regression tests compare NumPy snapshots directly.",
)


def _assert_snapshot_unchanged(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> None:
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_allclose(after[key], before[key], err_msg=key)


def _measurement_matrix():
    return array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )


def _kinematic_state():
    return array([1.0, 2.0, 3.0, 4.0])


def _kinematic_covariance():
    return diag(array([0.2, 0.3, 0.04, 0.05]))


def _shape_state():
    return array([0.25, 2.0, 1.0])


def _shape_covariance():
    return diag(array([0.1, 0.2, 0.3]))


def _snapshot_mem_qkf(tracker: MEMQKFTracker) -> dict[str, np.ndarray]:
    return {
        "kinematic_state": np.asarray(tracker.kinematic_state).copy(),
        "covariance": np.asarray(tracker.covariance).copy(),
        "shape_state": np.asarray(tracker.shape_state).copy(),
        "shape_covariance": np.asarray(tracker.shape_covariance).copy(),
    }


def _snapshot_mem_rbpf(tracker: MEMRBPFTracker) -> dict[str, np.ndarray]:
    return {
        "kinematic_state": np.asarray(tracker.kinematic_state).copy(),
        "covariance": np.asarray(tracker.covariance).copy(),
        "theta": np.asarray(tracker.theta).copy(),
        "axis": np.asarray(tracker.axis).copy(),
        "axis_covariances": np.asarray(tracker.axis_covariances).copy(),
        "weights": np.asarray(tracker.weights).copy(),
    }


def _snapshot_vbrm(tracker: VBRMTracker) -> dict[str, np.ndarray]:
    return {
        "kinematic_state": np.asarray(tracker.kinematic_state).copy(),
        "covariance": np.asarray(tracker.covariance).copy(),
        "orientation": np.asarray([tracker.orientation]).copy(),
        "orientation_variance": np.asarray([tracker.orientation_variance]).copy(),
        "alpha": np.asarray(tracker.alpha).copy(),
        "beta": np.asarray(tracker.beta).copy(),
    }


def _make_mem_qkf() -> MEMQKFTracker:
    return MEMQKFTracker(
        _kinematic_state(),
        _kinematic_covariance(),
        _shape_state(),
        _shape_covariance(),
        measurement_matrix=_measurement_matrix(),
        default_meas_noise_cov=0.05 * eye(2),
    )


def test_mem_qkf_empty_measurement_set_is_noop() -> None:
    tracker = _make_mem_qkf()
    before = _snapshot_mem_qkf(tracker)

    tracker.update(np.empty((0, 2)), meas_noise_cov=0.1 * eye(2))

    _assert_snapshot_unchanged(before, _snapshot_mem_qkf(tracker))


def test_velocity_aided_mem_qkf_empty_measurement_set_does_not_apply_heading_aid() -> (
    None
):
    tracker = VelocityAidedMEMQKFTracker(
        _kinematic_state(),
        _kinematic_covariance(),
        _shape_state(),
        _shape_covariance(),
        measurement_matrix=_measurement_matrix(),
        default_meas_noise_cov=0.05 * eye(2),
        heading_noise_variance=0.2,
    )
    before = _snapshot_mem_qkf(tracker)

    tracker.update(np.empty((0, 2)))

    _assert_snapshot_unchanged(before, _snapshot_mem_qkf(tracker))
    assert not tracker._heading_update_pending  # pylint: disable=protected-access


def test_velocity_locked_mem_qkf_empty_measurement_set_does_not_relock_orientation() -> (
    None
):
    tracker = VelocityLockedMEMQKFTracker(
        _kinematic_state(),
        _kinematic_covariance(),
        _shape_state(),
        _shape_covariance(),
        measurement_matrix=_measurement_matrix(),
        default_meas_noise_cov=0.05 * eye(2),
        speed_threshold=0.1,
    )
    tracker.shape_state = array([1.25, 2.0, 1.0])
    tracker.shape_covariance = diag(array([0.4, 0.2, 0.3]))
    before = _snapshot_mem_qkf(tracker)

    tracker.update(np.empty((0, 2)))

    _assert_snapshot_unchanged(before, _snapshot_mem_qkf(tracker))


def test_mem_rbpf_empty_measurement_set_is_noop() -> None:
    tracker = MEMRBPFTracker(
        _kinematic_state(),
        _kinematic_covariance(),
        _shape_state(),
        _shape_covariance(),
        meas_noise_cov=0.05 * eye(2),
        measurement_matrix=_measurement_matrix(),
        multiplicative_noise_cov=0.25 * eye(2),
        n_particles=8,
        rng=17,
        resampling_threshold=None,
    )
    before = _snapshot_mem_rbpf(tracker)

    tracker.update(np.empty((0, 2)))

    _assert_snapshot_unchanged(before, _snapshot_mem_rbpf(tracker))


def test_vbrm_empty_measurement_set_is_noop() -> None:
    tracker = VBRMTracker(
        _kinematic_state(),
        _kinematic_covariance(),
        _shape_state(),
        orientation_variance=0.15,
        inverse_gamma_shape=array([8.0, 9.0]),
        measurement_noise_cov=0.05 * eye(2),
        measurement_matrix=_measurement_matrix(),
    )
    before = _snapshot_vbrm(tracker)

    tracker.update(np.empty((0, 2)), meas_noise_cov=0.1 * eye(2))

    _assert_snapshot_unchanged(before, _snapshot_vbrm(tracker))
