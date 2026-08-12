import numpy as np
import pytest
from pyrecest.calibration.bias import SensorBiasCorrectionModel


def _model_kwargs():
    return {
        "target_dim": 1,
        "feature_dim": 1,
        "intercept": np.array([0.0]),
        "coefficients": np.array([[1.0]]),
        "feature_mean": np.array([0.0]),
        "feature_scale": np.array([1.0]),
        "residual_std": np.array([0.0]),
        "training_count": 2,
        "ridge_alpha": 0.0,
    }


@pytest.mark.parametrize("invalid_scale", [np.nan, np.inf, 0.0, -1.0])
def test_model_rejects_invalid_feature_scale(invalid_scale):
    kwargs = _model_kwargs()
    kwargs["feature_scale"] = np.array([invalid_scale])

    with pytest.raises(ValueError, match="feature_scale"):
        SensorBiasCorrectionModel(**kwargs)


def test_model_preserves_valid_feature_scale_and_prediction():
    kwargs = _model_kwargs()
    kwargs["feature_scale"] = np.array([2.0])
    model = SensorBiasCorrectionModel(**kwargs)

    np.testing.assert_array_equal(model.feature_scale, np.array([2.0]))
    np.testing.assert_allclose(model.predict(np.array([[4.0]])), np.array([[2.0]]))
