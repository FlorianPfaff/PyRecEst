import numpy as np
import pytest
from pyrecest.calibration.bias import SensorBiasCorrectionModel


def make_valid_model(**overrides):
    kwargs = {
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
    kwargs.update(overrides)
    return SensorBiasCorrectionModel(**kwargs)


def test_bias_model_rejects_fractional_dimension_fields():
    with pytest.raises(ValueError, match="target_dim"):
        make_valid_model(target_dim=1.5)

    with pytest.raises(ValueError, match="feature_dim"):
        make_valid_model(feature_dim=1.5)


def test_bias_model_rejects_boolean_scalar_fields():
    for field_name in ("target_dim", "feature_dim", "training_count", "ridge_alpha"):
        with pytest.raises(ValueError):
            make_valid_model(**{field_name: True})


def test_bias_model_rejects_invalid_training_metadata():
    with pytest.raises(ValueError, match="training_count"):
        make_valid_model(training_count=-1)

    with pytest.raises(ValueError, match="ridge_alpha"):
        make_valid_model(ridge_alpha=np.nan)

    with pytest.raises(ValueError, match="ridge_alpha"):
        make_valid_model(ridge_alpha=-1.0)


def test_bias_model_accepts_integer_like_scalar_metadata():
    model = make_valid_model(
        target_dim=np.float64(1.0),
        feature_dim=np.array(1),
        training_count=np.float64(2.0),
        ridge_alpha=np.float64(0.0),
    )

    assert model.target_dim == 1
    assert model.feature_dim == 1
    assert model.training_count == 2
    assert model.ridge_alpha == 0.0
