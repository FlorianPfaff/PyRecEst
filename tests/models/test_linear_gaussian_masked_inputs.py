import unittest

import numpy as np
from pyrecest.models import (
    IdentityGaussianMeasurementModel,
    IdentityGaussianTransitionModel,
    LinearGaussianMeasurementModel,
    LinearGaussianTransitionModel,
)


class LinearGaussianMaskedInputsTest(unittest.TestCase):
    def test_rejects_masked_constructor_inputs(self):
        masked_matrix = np.ma.array(
            [[1.0, 4.0], [0.0, 1.0]],
            mask=[[False, True], [False, False]],
        )
        masked_covariance = np.ma.array(
            [[1.0, 0.0], [0.0, 1.0]],
            mask=[[False, False], [False, True]],
        )
        masked_offset = np.ma.array([0.0, 3.0], mask=[False, True])

        with self.assertRaisesRegex(ValueError, "masked"):
            LinearGaussianTransitionModel(masked_matrix, np.eye(2))
        with self.assertRaisesRegex(ValueError, "masked"):
            LinearGaussianTransitionModel(np.eye(2), masked_covariance)
        with self.assertRaisesRegex(ValueError, "masked"):
            LinearGaussianTransitionModel(
                np.eye(2),
                np.eye(2),
                offset=masked_offset,
            )
        with self.assertRaisesRegex(ValueError, "masked"):
            LinearGaussianMeasurementModel(masked_matrix, np.eye(2))
        with self.assertRaisesRegex(ValueError, "masked"):
            LinearGaussianMeasurementModel(np.eye(2), masked_covariance)

    def test_rejects_masked_identity_controls(self):
        masked_dim = np.ma.array(2, mask=True)
        masked_scalar_covariance = np.ma.array(0.25, mask=True)

        with self.assertRaises(ValueError):
            IdentityGaussianTransitionModel(masked_dim, np.eye(2))
        with self.assertRaises(ValueError):
            IdentityGaussianMeasurementModel(masked_dim, np.eye(2))
        with self.assertRaisesRegex(ValueError, "masked"):
            IdentityGaussianTransitionModel(2, masked_scalar_covariance)
        with self.assertRaisesRegex(ValueError, "masked"):
            IdentityGaussianMeasurementModel(2, masked_scalar_covariance)

    def test_rejects_masked_prediction_inputs(self):
        transition_model = LinearGaussianTransitionModel(np.eye(2), np.eye(2))
        measurement_model = LinearGaussianMeasurementModel(np.eye(2), np.eye(2))
        masked_state = np.ma.array([1.0, 9.0], mask=[False, True])
        masked_covariance = np.ma.array(
            [[1.0, 0.0], [0.0, 9.0]],
            mask=[[False, False], [False, True]],
        )

        for model in (transition_model, measurement_model):
            with self.subTest(model=type(model).__name__, value="state"):
                with self.assertRaisesRegex(ValueError, "masked"):
                    model.predict_mean(masked_state)
            with self.subTest(model=type(model).__name__, value="covariance"):
                with self.assertRaisesRegex(ValueError, "masked"):
                    if isinstance(model, LinearGaussianTransitionModel):
                        model.predict_covariance(masked_covariance)
                    else:
                        model.innovation_covariance(masked_covariance)

    def test_accepts_clear_mask_wrappers(self):
        matrix = np.ma.array(np.eye(2), mask=False)
        covariance = np.ma.array(np.eye(2), mask=False)
        offset = np.ma.array([0.5, -0.5], mask=False)
        state = np.ma.array([1.0, 2.0], mask=False)

        transition_model = LinearGaussianTransitionModel(
            matrix,
            covariance,
            offset=offset,
        )
        measurement_model = LinearGaussianMeasurementModel(matrix, covariance)
        IdentityGaussianTransitionModel(np.ma.array(2, mask=False), covariance)
        IdentityGaussianMeasurementModel(np.ma.array(2, mask=False), covariance)

        transition_model.predict_mean(state)
        transition_model.predict_covariance(covariance)
        measurement_model.predict_mean(state)
        measurement_model.innovation_covariance(covariance)


if __name__ == "__main__":
    unittest.main()
