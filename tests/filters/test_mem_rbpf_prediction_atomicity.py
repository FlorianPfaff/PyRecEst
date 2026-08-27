import numpy as np
import numpy.testing as npt
import pytest
from pyrecest import backend
from pyrecest.backend import array, diag, eye
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ == "jax",
    reason="MEMRBPFTracker is unsupported on JAX.",
)


def _make_tracker():
    return MEMRBPFTracker(
        kinematic_state=array([0.0, 0.0, 1.0, -0.5]),
        covariance=eye(4),
        shape_state=array([0.2, 2.0, 1.0]),
        shape_covariance=diag(array([0.05, 0.1, 0.1])),
        meas_noise_cov=0.05 * eye(2),
        sys_noise=0.01 * eye(4),
        shape_sys_noise=diag(array([0.01, 0.01, 0.01])),
        n_particles=8,
        resampling_threshold=0,
        rng=7,
    )


def _to_numpy_copy(value):
    return np.asarray(backend.to_numpy(value)).copy()


def _snapshot(tracker):
    return {
        "kinematic_state": _to_numpy_copy(tracker.kinematic_state),
        "covariance": _to_numpy_copy(tracker.covariance),
        "system_matrix": _to_numpy_copy(tracker.system_matrix),
        "sys_noise": _to_numpy_copy(tracker.sys_noise),
        "axis": _to_numpy_copy(tracker.axis),
        "axis_covariances": _to_numpy_copy(tracker.axis_covariances),
        "axis_sys_noise": _to_numpy_copy(tracker.axis_sys_noise),
        "orientation_process_variance": tracker.orientation_process_variance,
    }


def _assert_snapshot_equal(tracker, snapshot):
    npt.assert_array_equal(
        _to_numpy_copy(tracker.kinematic_state), snapshot["kinematic_state"]
    )
    npt.assert_array_equal(_to_numpy_copy(tracker.covariance), snapshot["covariance"])
    npt.assert_array_equal(
        _to_numpy_copy(tracker.system_matrix), snapshot["system_matrix"]
    )
    npt.assert_array_equal(_to_numpy_copy(tracker.sys_noise), snapshot["sys_noise"])
    npt.assert_array_equal(_to_numpy_copy(tracker.axis), snapshot["axis"])
    npt.assert_array_equal(
        _to_numpy_copy(tracker.axis_covariances), snapshot["axis_covariances"]
    )
    npt.assert_array_equal(
        _to_numpy_copy(tracker.axis_sys_noise), snapshot["axis_sys_noise"]
    )
    assert (
        tracker.orientation_process_variance == snapshot["orientation_process_variance"]
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"system_matrix": eye(3)}, "system_matrix"),
        ({"shape_system_matrix": eye(4)}, "shape_system_matrix"),
        (
            {"shape_sys_noise": array([[1.0, 2.0], [2.0, 1.0]])},
            "shape_sys_noise",
        ),
    ],
)
def test_predict_validation_failures_leave_tracker_unchanged(override, message):
    tracker = _make_tracker()
    snapshot = _snapshot(tracker)

    with pytest.raises(ValueError, match=message):
        tracker.predict_linear(**override)

    _assert_snapshot_equal(tracker, snapshot)
