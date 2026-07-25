import numpy as np
from pyrecest.calibration.bias import SensorBiasCorrectionModel


def test_bias_model_copies_mutable_parameter_arrays():
    intercept = np.array([1.0])
    coefficients = np.array([[2.0]])
    feature_mean = np.array([3.0])
    feature_scale = np.array([4.0])
    residual_std = np.array([5.0])
    model = SensorBiasCorrectionModel(
        target_dim=1,
        feature_dim=1,
        intercept=intercept,
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        residual_std=residual_std,
        training_count=1,
        ridge_alpha=0.0,
    )
    features = np.array([[7.0]])
    expected_prediction = model.predict(features).copy()

    intercept[:] = 10.0
    coefficients[:] = 20.0
    feature_mean[:] = 30.0
    feature_scale[:] = 40.0
    residual_std[:] = 50.0

    np.testing.assert_allclose(model.intercept, [1.0])
    np.testing.assert_allclose(model.coefficients, [[2.0]])
    np.testing.assert_allclose(model.feature_mean, [3.0])
    np.testing.assert_allclose(model.feature_scale, [4.0])
    np.testing.assert_allclose(model.residual_std, [5.0])
    np.testing.assert_allclose(model.predict(features), expected_prediction)
