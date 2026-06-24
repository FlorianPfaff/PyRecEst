import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


@unittest.skipIf(
    __backend_name__ == "pytorch",
    reason="Sigma-point tests use NumPy assertions and the PyTorch backend is unsupported",
)
class TestSigmaPoints(unittest.TestCase):
    def test_merwe_scaled_sigma_points_match_moments(self):
        mean = np.array([1.0, -2.0])
        covariance = np.array([[2.0, 0.4], [0.4, 1.0]])
        points = MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0)

        sigmas = points.sigma_points(asarray(mean), asarray(covariance))
        weighted_mean = points.Wm @ sigmas
        deviations = sigmas - weighted_mean
        weighted_covariance = deviations.T @ (deviations * points.Wc[:, None])

        self.assertEqual(sigmas.shape, (5, 2))
        npt.assert_allclose(to_numpy(weighted_mean), mean)
        npt.assert_allclose(to_numpy(weighted_covariance), covariance)

    def test_julier_sigma_points_match_moments(self):
        mean = np.array([0.5, 1.5])
        covariance = np.array([[1.5, 0.2], [0.2, 0.5]])
        points = JulierSigmaPoints(n=2, kappa=1.0)

        sigmas = points.sigma_points(asarray(mean), asarray(covariance))
        weighted_mean = points.Wm @ sigmas
        deviations = sigmas - weighted_mean
        weighted_covariance = deviations.T @ (deviations * points.Wc[:, None])

        self.assertEqual(sigmas.shape, (5, 2))
        npt.assert_allclose(to_numpy(weighted_mean), mean)
        npt.assert_allclose(to_numpy(weighted_covariance), covariance)

    def test_sigma_point_constructors_reject_invalid_parameters(self):
        invalid_cases = (
            (
                MerweScaledSigmaPoints,
                {"n": 0, "alpha": 0.5, "beta": 2.0, "kappa": 0.0},
                "n must be a positive integer",
            ),
            (
                MerweScaledSigmaPoints,
                {"n": True, "alpha": 0.5, "beta": 2.0, "kappa": 0.0},
                "n must be a scalar",
            ),
            (
                MerweScaledSigmaPoints,
                {"n": 2, "alpha": 0.0, "beta": 2.0, "kappa": 0.0},
                "alpha must be positive",
            ),
            (
                MerweScaledSigmaPoints,
                {
                    "n": 2,
                    "alpha": np.array(True, dtype=object),
                    "beta": 2.0,
                    "kappa": 0.0,
                },
                "alpha must be a scalar",
            ),
            (
                MerweScaledSigmaPoints,
                {"n": 2, "alpha": 0.5, "beta": np.inf, "kappa": 0.0},
                "beta must be finite",
            ),
            (
                MerweScaledSigmaPoints,
                {"n": 2, "alpha": 0.5, "beta": 2.0, "kappa": -2.0},
                r"n \+ kappa must be positive",
            ),
            (
                JulierSigmaPoints,
                {"n": False, "kappa": 0.0},
                "n must be a scalar",
            ),
            (
                JulierSigmaPoints,
                {"n": 2, "kappa": np.nan},
                "kappa must be finite",
            ),
            (
                JulierSigmaPoints,
                {"n": 2, "kappa": -2.0},
                r"n \+ kappa must be positive",
            ),
        )

        for cls, kwargs, expected_message in invalid_cases:
            with self.subTest(cls=cls.__name__, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, expected_message):
                    cls(**kwargs)

    def test_sigma_points_reject_dimension_mismatches(self):
        points = JulierSigmaPoints(n=2, kappa=1.0)

        with self.assertRaisesRegex(ValueError, "x must have shape"):
            points.sigma_points(asarray(np.array([0.0, 1.0, 2.0])), asarray(np.eye(2)))

        with self.assertRaisesRegex(ValueError, "P must have shape"):
            points.sigma_points(asarray(np.array([0.0, 1.0])), asarray(np.eye(3)))


if __name__ == "__main__":
    unittest.main()
