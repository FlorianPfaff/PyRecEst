import unittest

import numpy as np
from pyrecest.filters.candidate_mixture import GaussianMixtureMeasurementFactor


class GaussianMixtureMeasurementCovarianceSymmetryTest(unittest.TestCase):
    def test_rejects_nonsymmetric_shared_covariance(self):
        nonsymmetric = np.array([[2.0, 1.0], [0.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "covariances must be symmetric"):
            GaussianMixtureMeasurementFactor(
                means=np.zeros((2, 2)),
                covariances=nonsymmetric,
            )

    def test_rejects_nonsymmetric_component_covariance(self):
        covariances = np.array(
            [
                np.eye(2),
                [[2.0, 1.0], [0.0, 2.0]],
            ]
        )

        with self.assertRaisesRegex(ValueError, "covariances must be symmetric"):
            GaussianMixtureMeasurementFactor(
                means=np.zeros((2, 2)),
                covariances=covariances,
            )

    def test_tolerates_roundoff_scale_asymmetry(self):
        covariance = np.array([[2.0, 1.0 + 1e-13], [1.0, 2.0]])

        factor = GaussianMixtureMeasurementFactor(
            means=np.zeros((1, 2)),
            covariances=covariance,
        )

        np.testing.assert_allclose(
            factor.covariances[0],
            0.5 * (covariance + covariance.T),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
