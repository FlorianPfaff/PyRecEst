import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.utils.metrics import nees, nis


class TestConsistencyMetricCovarianceValidation(unittest.TestCase):
    def test_nees_and_nis_reject_asymmetric_covariances(self):
        residual = np.array([1.0, 1.0])
        asymmetric_covariance = np.array([[1.0, 10.0], [0.0, 1.0]])

        with self.assertRaisesRegex(
            ValueError,
            "uncertainties must contain symmetric covariance matrices",
        ):
            nees(residual, asymmetric_covariance)

        with self.assertRaisesRegex(
            ValueError,
            "innovation_covariances must contain symmetric covariance matrices",
        ):
            nis(residual, asymmetric_covariance)

    def test_asymmetry_validation_respects_strict_numpy_error_policy(self):
        maximum_float = np.finfo(float).max
        asymmetric_covariance = np.array(
            [[1.0, maximum_float], [-maximum_float, 1.0]]
        )
        previous_settings = np.seterr(all="raise")
        try:
            with self.assertRaisesRegex(
                ValueError,
                "uncertainties must contain symmetric covariance matrices",
            ):
                nees(np.array([1.0, 1.0]), asymmetric_covariance)
        finally:
            np.seterr(**previous_settings)

    def test_nees_and_nis_reject_non_positive_definite_covariances(self):
        residuals = np.array([[1.0, 0.0], [0.0, 1.0]])
        covariance_stack = np.array(
            [
                np.eye(2),
                [[1.0, 2.0], [2.0, 1.0]],
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "uncertainties must contain positive-definite covariance matrices",
        ):
            nees(residuals, covariance_stack)

        with self.assertRaisesRegex(
            ValueError,
            "innovation_covariances must contain positive-definite covariance matrices",
        ):
            nis(residuals, covariance_stack)

    def test_valid_covariance_stacks_remain_supported(self):
        residuals = np.array([[1.0, 0.0], [0.0, 2.0]])
        covariance_stack = np.array(
            [
                np.eye(2),
                [[2.0, 0.0], [0.0, 4.0]],
            ]
        )

        npt.assert_allclose(nees(residuals, covariance_stack), [1.0, 1.0])
        npt.assert_allclose(nis(residuals, covariance_stack), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
