import unittest

import numpy as np
from pyrecest.calibration.bias import (
    SensorBiasCorrectionModel,
    make_bias_training_examples,
)


class BiasComplexInputValidationTest(unittest.TestCase):
    def test_make_bias_training_examples_rejects_complex_inputs(self):
        valid_times = np.array([0.0, 1.0])
        valid_values = np.array([[0.0], [1.0]])
        cases = (
            (
                "measurement_times_s",
                np.array([0.0 + 1.0j, 1.0]),
                valid_values,
                valid_times,
                valid_values,
                None,
            ),
            (
                "measurement_values",
                valid_times,
                np.array([[0.0 + 1.0j], [1.0]]),
                valid_times,
                valid_values,
                None,
            ),
            (
                "reference_times_s",
                valid_times,
                valid_values,
                np.array([0.0, np.complex64(1.0 + 1.0j)], dtype=object),
                valid_values,
                None,
            ),
            (
                "reference_values",
                valid_times,
                valid_values,
                valid_times,
                np.array([[0.0], [np.complex64(1.0 + 1.0j)]], dtype=object),
                None,
            ),
            (
                "feature_values",
                valid_times,
                valid_values,
                valid_times,
                valid_values,
                np.array([[0.0], [np.complex64(1.0 + 1.0j)]], dtype=object),
            ),
        )

        for (
            expected_name,
            measurement_times,
            measurement_values,
            reference_times,
            reference_values,
            feature_values,
        ) in cases:
            with self.subTest(expected_name=expected_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{expected_name} must contain numeric values",
                ):
                    make_bias_training_examples(
                        measurement_times,
                        measurement_values,
                        reference_times,
                        reference_values,
                        feature_values=feature_values,
                    )

    def test_model_rejects_complex_parameters(self):
        with self.assertRaisesRegex(
            ValueError,
            "intercept must contain numeric values",
        ):
            SensorBiasCorrectionModel(
                target_dim=1,
                feature_dim=0,
                intercept=np.array([1.0 + 1.0j]),
                coefficients=np.empty((0, 1)),
                feature_mean=np.empty(0),
                feature_scale=np.empty(0),
                residual_std=np.array([0.0]),
                training_count=1,
                ridge_alpha=0.0,
            )

    def test_model_rejects_complex_predict_and_apply_inputs(self):
        model = SensorBiasCorrectionModel(
            target_dim=1,
            feature_dim=1,
            intercept=np.array([0.0]),
            coefficients=np.array([[1.0]]),
            feature_mean=np.array([0.0]),
            feature_scale=np.array([1.0]),
            residual_std=np.array([0.0]),
            training_count=1,
            ridge_alpha=0.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "features must contain numeric values",
        ):
            model.predict(np.array([[np.complex64(1.0 + 1.0j)]], dtype=object))

        with self.assertRaisesRegex(
            ValueError,
            "measurements must contain numeric values",
        ):
            model.apply(np.array([[1.0 + 1.0j]]), np.array([[0.0]]))


if __name__ == "__main__":
    unittest.main()
