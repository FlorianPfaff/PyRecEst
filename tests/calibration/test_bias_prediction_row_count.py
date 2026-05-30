import numpy as np
import pytest
from pyrecest.calibration.bias import SensorBiasCorrectionModel


def _constant_bias_model():
    return SensorBiasCorrectionModel(
        target_dim=1,
        feature_dim=0,
        intercept=np.array([2.0]),
        coefficients=np.empty((0, 1)),
        feature_mean=np.empty(0),
        feature_scale=np.empty(0),
        residual_std=np.array([0.0]),
        training_count=3,
        ridge_alpha=0.0,
    )


def _feature_bias_model():
    return SensorBiasCorrectionModel(
        target_dim=1,
        feature_dim=1,
        intercept=np.array([0.0]),
        coefficients=np.array([[1.0]]),
        feature_mean=np.array([0.0]),
        feature_scale=np.array([1.0]),
        residual_std=np.array([0.0]),
        training_count=3,
        ridge_alpha=0.0,
    )


@pytest.mark.parametrize("n_rows", [-1, 1.5, np.nan, np.inf, True, np.array([1])])
def test_constant_bias_predict_rejects_invalid_n_rows(n_rows):
    model = _constant_bias_model()

    with pytest.raises(ValueError, match="n_rows must be a nonnegative integer"):
        model.predict(n_rows=n_rows)


def test_constant_bias_predict_accepts_integer_like_n_rows():
    model = _constant_bias_model()

    prediction = model.predict(n_rows=np.array(2.0))

    assert prediction.shape == (2, 1)
    np.testing.assert_allclose(prediction, np.array([[2.0], [2.0]]))


def test_constant_bias_predict_uses_zero_feature_row_count():
    model = _constant_bias_model()

    prediction = model.predict(np.empty((3, 0)))

    assert prediction.shape == (3, 1)
    np.testing.assert_allclose(prediction, np.full((3, 1), 2.0))


def test_constant_bias_predict_rejects_mismatched_zero_feature_row_count():
    model = _constant_bias_model()

    with pytest.raises(
        ValueError, match="features rows must match requested row count"
    ):
        model.predict(np.empty((3, 0)), n_rows=2)


def test_constant_bias_predict_rejects_nonzero_feature_columns():
    model = _constant_bias_model()

    with pytest.raises(
        ValueError, match="features have incompatible feature dimension"
    ):
        model.predict(np.ones((3, 1)))


@pytest.mark.parametrize("n_rows", [0.5, np.nan, np.inf, False])
def test_feature_bias_predict_validates_explicit_n_rows_before_row_comparison(n_rows):
    model = _feature_bias_model()

    with pytest.raises(ValueError, match="n_rows must be a nonnegative integer"):
        model.predict(np.array([[1.0]]), n_rows=n_rows)
