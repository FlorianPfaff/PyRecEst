import numpy as np
import pytest
import pyrecest.backend
from pyrecest.backend import array, diag, eye
from pyrecest.filters import MEMEKFTracker, MEMQKFTracker


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM covariance validation regressions use the NumPy-backed MEM trackers",
)


def _make_mem_ekf(**kwargs):
    parameters = {
        "kinematic_state": array([0.0, 0.0, 1.0, -0.5]),
        "covariance": eye(4),
        "shape_state": array([0.2, 2.0, 1.0]),
        "shape_covariance": diag(array([0.05, 0.1, 0.1])),
    }
    parameters.update(kwargs)
    return MEMEKFTracker(**parameters)


@pytest.mark.parametrize(
    ("name", "invalid_covariance"),
    [
        ("covariance", diag(array([1.0, 1.0, -1.0, 1.0]))),
        ("shape_covariance", diag(array([0.1, -0.1, 0.1]))),
        ("multiplicative_noise_cov", array([[1.0, 2.0], [2.0, 1.0]])),
    ],
)
def test_constructor_rejects_indefinite_covariances(name, invalid_covariance):
    with pytest.raises(ValueError, match=name):
        _make_mem_ekf(**{name: invalid_covariance})


def test_constructor_rejects_nonfinite_covariance():
    with pytest.raises(ValueError, match="covariance"):
        _make_mem_ekf(covariance=diag(array([1.0, 1.0, np.nan, 1.0])))


def test_constructor_rejects_asymmetric_covariance_instead_of_silently_repairing():
    asymmetric = eye(4)
    asymmetric[0, 1] = 0.5

    with pytest.raises(ValueError, match="covariance"):
        _make_mem_ekf(covariance=asymmetric)


def test_semidefinite_measurement_and_process_noise_remain_supported():
    tracker = _make_mem_ekf()
    system_matrix = eye(4)

    tracker.predict_linear(
        system_matrix,
        sys_noise=diag(array([0.1, 0.0, 0.1, 0.0])),
        shape_sys_noise=diag(array([0.1, 0.0, 0.1])),
    )
    tracker.update(
        array([[0.0, 0.0]]),
        meas_noise_cov=array([[0.1, 0.0], [0.0, 0.0]]),
    )


def test_predict_rejects_indefinite_noise_before_using_it():
    tracker = _make_mem_ekf()

    with pytest.raises(ValueError, match="sys_noise"):
        tracker.predict_linear(
            eye(4),
            sys_noise=diag(array([0.1, 0.1, -0.1, 0.1])),
        )

    tracker = _make_mem_ekf()
    with pytest.raises(ValueError, match="shape_sys_noise"):
        tracker.predict_linear(
            eye(4),
            shape_sys_noise=diag(array([0.1, -0.1, 0.1])),
        )


def test_update_rejects_indefinite_measurement_noise():
    tracker = _make_mem_ekf()

    with pytest.raises(ValueError, match="meas_noise_cov"):
        tracker.update(
            array([[0.0, 0.0]]),
            meas_noise_cov=array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_mem_qkf_inherits_measurement_noise_validation():
    tracker = MEMQKFTracker(
        array([0.0, 0.0, 1.0, -0.5]),
        eye(4),
        array([0.2, 2.0, 1.0]),
        diag(array([0.05, 0.1, 0.1])),
    )

    with pytest.raises(ValueError, match="meas_noise_cov"):
        tracker.set_default_measurement_noise_cov(
            array([[1.0, 2.0], [2.0, 1.0]])
        )
