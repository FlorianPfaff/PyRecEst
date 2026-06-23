import numpy as np
import pytest
from pyrecest.calibration.bias import SensorBiasCorrectionModel


def test_apply_rejects_feature_row_count_mismatch():
    model = SensorBiasCorrectionModel(
        target_dim=1,
        feature_dim=1,
        intercept=np.array([0.0]),
        coefficients=np.array([[1.0]]),
        feature_mean=np.array([0.0]),
        feature_scale=np.array([1.0]),
        residual_std=np.array([0.0]),
        training_count=2,
        ridge_alpha=0.0,
    )

    measurements = np.array([[10.0], [20.0]])
    features = np.array([[1.0]])

    with pytest.raises(ValueError, match="one predicted bias row per measurement"):
        model.apply(measurements, features)


def test_apply_accepts_matching_feature_rows():
    model = SensorBiasCorrectionModel(
        target_dim=1,
        feature_dim=1,
        intercept=np.array([0.5]),
        coefficients=np.array([[2.0]]),
        feature_mean=np.array([1.0]),
        feature_scale=np.array([2.0]),
        residual_std=np.array([0.0]),
        training_count=3,
        ridge_alpha=0.0,
    )

    measurements = np.array([[10.0], [20.0]])
    features = np.array([[1.0], [3.0]])

    corrected = model.apply(measurements, features)

    np.testing.assert_allclose(corrected, np.array([[9.5], [17.5]]))


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "match"),
    [
        ("target_dim", 1.5, "target_dim"),
        ("feature_dim", 1.5, "feature_dim"),
        ("training_count", 2.5, "training_count"),
        ("ridge_alpha", np.nan, "ridge_alpha"),
    ],
)
def test_model_rejects_invalid_scalar_metadata(field_name, invalid_value, match):
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
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=match):
        SensorBiasCorrectionModel(**kwargs)
