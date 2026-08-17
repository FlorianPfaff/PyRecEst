"""Regression tests for catalog motion-model process-noise validation."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.models import (
    coordinated_turn_model,
    nearly_constant_speed_model,
    se2_unicycle_model,
    se3_pose_twist_model,
)
from pyrecest.models import motion_models


class TestMotionModelNoiseCovarianceValidation(unittest.TestCase):
    def _constructors(self):
        return (
            (coordinated_turn_model, motion_models.coordinated_turn_model, 5),
            (
                nearly_constant_speed_model,
                motion_models.nearly_constant_speed_model,
                4,
            ),
            (se2_unicycle_model, motion_models.se2_unicycle_model, 5),
            (se3_pose_twist_model, motion_models.se3_pose_twist_model, 12),
        )

    def test_rejects_malformed_process_noise_covariances(self) -> None:
        for package_constructor, module_constructor, dim in self._constructors():
            invalid_covariances = []

            wrong_size = np.eye(max(1, dim - 1))
            invalid_covariances.append((wrong_size, "noise_covariance"))

            nonfinite = np.eye(dim)
            nonfinite[0, 0] = np.nan
            invalid_covariances.append((nonfinite, "finite"))

            nonsymmetric = np.eye(dim)
            nonsymmetric[0, 1] = 1.0
            invalid_covariances.append((nonsymmetric, "symmetric"))

            complex_covariance = np.eye(dim, dtype=complex)
            invalid_covariances.append((complex_covariance, "real values"))

            for constructor in (package_constructor, module_constructor):
                for covariance, message in invalid_covariances:
                    with self.subTest(
                        constructor=constructor.__name__,
                        dim=dim,
                        message=message,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            constructor(noise_covariance=covariance)

    def test_rejects_nonfinite_or_boolean_dt(self) -> None:
        for package_constructor, module_constructor, _ in self._constructors():
            for constructor in (package_constructor, module_constructor):
                for dt in (np.nan, np.inf, True):
                    with self.subTest(constructor=constructor.__name__, dt=dt):
                        with self.assertRaisesRegex(ValueError, "dt"):
                            constructor(dt=dt)

    def test_accepts_matching_symmetric_covariance(self) -> None:
        for package_constructor, module_constructor, dim in self._constructors():
            covariance = np.eye(dim) * 0.25
            for constructor in (package_constructor, module_constructor):
                with self.subTest(constructor=constructor.__name__, dim=dim):
                    model = constructor(noise_covariance=covariance)
                    np.testing.assert_allclose(model.noise_covariance, covariance)


if __name__ == "__main__":
    unittest.main()
