import numpy as np
import pytest
from pyrecest.calibration.bias import (
    SensorBiasCorrectionModel,
    make_bias_training_examples,
)


def _unit_feature_model_kwargs():
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


def test_model_rejects_text_array_field_before_float_coercion():
    kwargs = _unit_feature_model_kwargs()
    kwargs["intercept"] = np.array(["0.0"])

    with pytest.raises(ValueError) as exc_info:
        SensorBiasCorrectionModel(**kwargs)

    assert "intercept must contain numeric values" in str(exc_info.value)


def test_model_rejects_boolean_array_field_before_float_coercion():
    kwargs = _unit_feature_model_kwargs()
    kwargs["coefficients"] = np.array([[True]])

    with pytest.raises(ValueError) as exc_info:
        SensorBiasCorrectionModel(**kwargs)

    assert "coefficients must contain numeric values" in str(exc_info.value)


def test_make_bias_training_examples_rejects_text_times_before_float_coercion():
    with pytest.raises(ValueError) as exc_info:
        make_bias_training_examples(
            measurement_times_s=np.array(["0.0", "1.0"]),
            measurement_values=np.array([[1.0], [2.0]]),
            reference_times_s=np.array([0.0, 1.0]),
            reference_values=np.array([[1.5], [2.5]]),
        )

    assert "measurement_times_s must contain numeric values" in str(exc_info.value)
