import numpy as np
import numpy.testing as npt
import pytest
from pyrecest import backend
from pyrecest.backend import array, diag, eye, random
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker
from pyrecest.smoothers import (
    MEMRBPF_FFBSiSmoother,
    MEMRBPFFFBSiSmoother,
    MEMRBPFForwardRecord,
    RBFFBSiSmoother,
)

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="MEM-RBPF FFBSi tests use NumPy sampling paths",
)


def _record(
    x,
    p,
    theta,
    axis,
    axis_covariance,
    weights=None,
    *,
    system_matrix=None,
    sys_noise=None,
    axis_system_matrix=None,
    axis_sys_noise=None,
    orientation_process_variance=0.1,
):
    theta = np.asarray(theta, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis_covariance = np.asarray(axis_covariance, dtype=float)
    if weights is None:
        weights = np.full(theta.shape[0], 1.0 / theta.shape[0])
    return MEMRBPFForwardRecord(
        kinematic_state=np.asarray(x, dtype=float).reshape(-1),
        covariance=np.asarray(p, dtype=float),
        theta=theta,
        axis_mean=axis,
        axis_covariance=axis_covariance,
        weights=np.asarray(weights, dtype=float),
        system_matrix=system_matrix,
        sys_noise=sys_noise,
        axis_system_matrix=axis_system_matrix,
        axis_sys_noise=axis_sys_noise,
        orientation_process_variance=orientation_process_variance,
    )


def test_aliases():
    assert MEMRBPF_FFBSiSmoother is MEMRBPFFFBSiSmoother
    assert RBFFBSiSmoother is MEMRBPFFFBSiSmoother


def test_two_record_deterministic_backward_kernel_matches_rts_moments():
    axis_cov = np.repeat(np.eye(2)[np.newaxis, :, :] * 0.5, 1, axis=0)
    records = [
        _record(
            [0.0],
            [[0.5]],
            [0.1],
            [[2.0, 1.0]],
            axis_cov,
            system_matrix=np.array([[1.0]]),
            sys_noise=np.array([[0.5]]),
            axis_system_matrix=np.eye(2),
            axis_sys_noise=0.5 * np.eye(2),
        ),
        _record([1.0], [[0.4]], [0.2], [[3.0, 2.0]], axis_cov),
    ]

    smoother = MEMRBPFFFBSiSmoother(n_trajectories=1, sample_axis=False)
    result = smoother.smooth(records, rng=0, full_axis_lengths=False)

    npt.assert_allclose(result.kinematic_mean[0], np.array([0.5]))
    npt.assert_allclose(result.axis_samples[0, 0], np.array([2.5, 1.5]))
    npt.assert_allclose(result.states[0, 0], 0.5)
    npt.assert_allclose(result.states[0, 2:4], np.array([2.5, 1.5]))


def test_smoother_returns_finite_shapes_for_multiple_particles():
    axis_cov = np.repeat(np.eye(2)[np.newaxis, :, :] * 0.1, 3, axis=0)
    records = [
        _record(
            [0.0, 0.0, 1.0, 0.0],
            np.eye(4),
            [0.0, 0.2, 0.4],
            [[2.0, 1.0], [2.2, 1.1], [1.8, 0.9]],
            axis_cov,
            weights=[0.2, 0.5, 0.3],
            system_matrix=np.eye(4),
            sys_noise=0.1 * np.eye(4),
            axis_sys_noise=0.05 * np.eye(2),
        ),
        _record(
            [1.0, 0.0, 1.0, 0.0],
            0.5 * np.eye(4),
            [0.1, 0.3, 0.5],
            [[2.1, 1.0], [2.3, 1.0], [1.9, 0.8]],
            axis_cov,
            weights=[0.1, 0.7, 0.2],
            system_matrix=np.eye(4),
            sys_noise=0.1 * np.eye(4),
            axis_sys_noise=0.05 * np.eye(2),
        ),
        _record(
            [2.0, 0.0, 1.0, 0.0],
            0.4 * np.eye(4),
            [0.2, 0.4, 0.6],
            [[2.2, 1.1], [2.4, 1.1], [2.0, 0.9]],
            axis_cov,
            weights=[0.15, 0.7, 0.15],
        ),
    ]

    smoother = MEMRBPFFFBSiSmoother(n_trajectories=16, sample_axis=False)
    result = smoother.smooth(records, rng=np.random.default_rng(1))

    assert result.states.shape == (3, 7)
    assert result.sample_states.shape == (16, 3, 7)
    assert result.theta_samples.shape == (16, 3)
    assert result.axis_samples.shape == (16, 3, 2)
    assert result.index_samples.shape == (16, 3)
    assert np.all(np.isfinite(result.states))
    assert np.all(result.states[:, -2:] > 0.0)


def test_forward_record_from_tracker_smoke_after_update():
    random.seed(0)
    tracker = MEMRBPFTracker(
        kinematic_state=array([0.0, 0.0, 1.0, -0.5]),
        covariance=eye(4),
        shape_state=array([0.2, 2.0, 1.0]),
        shape_covariance=diag(array([0.05, 0.1, 0.1])),
        meas_noise_cov=0.05 * eye(2),
        sys_noise=0.01 * eye(4),
        shape_sys_noise=diag(array([0.02, 0.01, 0.01])),
        n_particles=16,
        resampling_threshold=None,
        axis_floor=1e-3,
    )

    tracker.predict()
    tracker.update(np.array([[1.2, 0.1], [0.8, -0.2], [1.0, 0.2]]))
    record = MEMRBPFForwardRecord.from_tracker(tracker)

    assert record.theta.shape == (16,)
    assert record.axis_mean.shape == (16, 2)
    assert record.axis_covariance.shape == (16, 2, 2)
    npt.assert_allclose(np.sum(record.weights), 1.0)
    assert np.all(np.isfinite(record.weights))
    assert np.all(np.isfinite(record.axis_mean))


def test_smooth_accepts_mapping_records():
    axis_cov = np.repeat(np.eye(2)[np.newaxis, :, :] * 0.2, 2, axis=0)
    record = _record(
        [0.0],
        [[1.0]],
        [0.0, 0.2],
        [[2.0, 1.0], [1.8, 0.8]],
        axis_cov,
        weights=[0.75, 0.25],
    )
    mapping = {
        "kinematic_state": record.kinematic_state,
        "covariance": record.covariance,
        "theta": record.theta,
        "axis": record.axis_mean,
        "axis_covariances": record.axis_covariance,
        "weights": record.weights,
    }

    result = MEMRBPFFFBSiSmoother(n_trajectories=4, sample_axis=False).smooth(
        [mapping], rng=2, full_axis_lengths=False
    )

    assert result.states.shape == (1, 4)
    assert np.all(np.isfinite(result.states))
