import numpy as np
import pyrecest.backend
import pytest
from pyrecest.backend import array, diag, eye
from pyrecest.filters import GGIWTracker

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="GGIW validation regressions use the NumPy-backed tracker",
)


def _make_tracker(**kwargs):
    parameters = {
        "kinematic_state": array([0.0, 0.0, 1.0, -1.0]),
        "covariance": diag(array([1.0, 1.0, 0.25, 0.25])),
        "extent": diag(array([4.0, 1.0])),
        "extent_degrees_of_freedom": 12.0,
        "gamma_shape": 4.0,
        "gamma_rate": 2.0,
    }
    parameters.update(kwargs)
    return GGIWTracker(**parameters)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("gamma_shape", np.nan),
        ("gamma_shape", np.inf),
        ("gamma_rate", np.nan),
        ("gamma_rate", np.inf),
        ("extent_innovation_weight", np.nan),
        ("extent_innovation_weight", np.inf),
        ("extent_degrees_of_freedom", np.nan),
        ("extent_degrees_of_freedom", np.inf),
    ],
)
def test_constructor_rejects_nonfinite_scalar_hyperparameters(name, value):
    with pytest.raises(ValueError, match=name):
        _make_tracker(**{name: value})


def test_constructor_rejects_non_boolean_extent_is_scale():
    with pytest.raises(TypeError, match="extent_is_scale"):
        _make_tracker(extent_is_scale="False")


def test_constructor_rejects_asymmetric_covariance_and_extent():
    asymmetric_covariance = eye(4)
    asymmetric_covariance[0, 1] = 0.5
    with pytest.raises(ValueError, match="covariance"):
        _make_tracker(covariance=asymmetric_covariance)

    with pytest.raises(ValueError, match="extent"):
        _make_tracker(extent=array([[4.0, 1.0], [0.0, 1.0]]))


def test_constructor_rejects_nonfinite_covariance_and_extent():
    with pytest.raises(ValueError, match="covariance"):
        _make_tracker(covariance=diag(array([1.0, 1.0, np.nan, 1.0])))

    with pytest.raises(ValueError, match="extent"):
        _make_tracker(extent=array([[4.0, 0.0], [0.0, np.nan]]))


def test_update_rejects_nonfinite_extent_innovation_weight():
    tracker = _make_tracker()

    with pytest.raises(ValueError, match="extent_innovation_weight"):
        tracker.update(
            array([[0.0], [0.0]]),
            extent_innovation_weight=np.nan,
        )


def test_finite_valid_inputs_remain_supported():
    tracker = _make_tracker(
        extent_is_scale=np.bool_(False),
        extent_innovation_weight=0.0,
    )

    tracker.predict_linear(eye(4), 0.01 * eye(4))
    tracker.update(array([[0.0], [0.0]]), meas_noise_cov=0.1 * eye(2))

    assert np.isfinite(float(tracker.get_measurement_rate_estimate()))
    assert np.all(np.isfinite(np.asarray(tracker.get_point_estimate_extent())))
