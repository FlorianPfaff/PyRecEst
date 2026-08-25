import numpy as np
import pytest
from pyrecest import backend
from pyrecest.backend import array, diag, eye
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ == "jax",
    reason="MEMRBPFTracker is unsupported on JAX.",
)


def _make_tracker(**kwargs):
    parameters = {
        "kinematic_state": array([0.0, 0.0, 1.0, -0.5]),
        "covariance": eye(4),
        "shape_state": array([0.2, 2.0, 1.0]),
        "shape_covariance": diag(array([0.05, 0.1, 0.1])),
        "meas_noise_cov": 0.05 * eye(2),
        "sys_noise": 0.01 * eye(4),
        "shape_sys_noise": diag(array([0.01, 0.01, 0.01])),
        "n_particles": 8,
        "resampling_threshold": 0,
    }
    parameters.update(kwargs)
    return MEMRBPFTracker(**parameters)


@pytest.mark.parametrize(
    ("name", "invalid_covariance"),
    [
        ("meas_noise_cov", array([[1.0, 2.0], [2.0, 1.0]])),
        ("sys_noise", diag(array([0.1, 0.1, -0.1, 0.1]))),
        ("shape_sys_noise", diag(array([0.1, -0.1, 0.1]))),
    ],
)
def test_constructor_rejects_indefinite_semidefinite_covariances(
    name, invalid_covariance
):
    with pytest.raises(ValueError, match=name):
        _make_tracker(**{name: invalid_covariance})


def test_constructor_rejects_nonfinite_covariance():
    with pytest.raises(ValueError, match="meas_noise_cov"):
        _make_tracker(meas_noise_cov=array([[1.0, 0.0], [0.0, np.nan]]))


def test_constructor_rejects_asymmetric_covariance_instead_of_silently_repairing():
    with pytest.raises(ValueError, match="meas_noise_cov"):
        _make_tracker(meas_noise_cov=array([[1.0, 0.5], [0.0, 1.0]]))


def test_constructor_accepts_positive_semidefinite_zero_noise():
    tracker = _make_tracker(
        meas_noise_cov=array([[1.0, 0.0], [0.0, 0.0]]),
        sys_noise=0.0 * eye(4),
        shape_sys_noise=0.0 * eye(3),
    )

    assert tracker.meas_noise_cov.shape == (2, 2)
    assert tracker.sys_noise.shape == (4, 4)
    assert tracker.shape_sys_noise.shape == (3, 3)


def test_predict_rejects_indefinite_process_noise_overrides():
    tracker = _make_tracker()

    with pytest.raises(ValueError, match="sys_noise"):
        tracker.predict_linear(sys_noise=diag(array([0.1, 0.1, -0.1, 0.1])))

    with pytest.raises(ValueError, match="shape_sys_noise"):
        tracker.predict_linear(shape_sys_noise=diag(array([0.1, -0.1, 0.1])))

    with pytest.raises(ValueError, match="shape_sys_noise"):
        tracker.predict_linear(shape_sys_noise=array([[1.0, 2.0], [2.0, 1.0]]))


def test_update_rejects_indefinite_measurement_noise_override():
    tracker = _make_tracker()

    with pytest.raises(ValueError, match="meas_noise_cov"):
        tracker.update(
            array([[0.0, 0.0]]),
            meas_noise_cov=array([[1.0, 2.0], [2.0, 1.0]]),
        )
