"""Regression tests for scale-stable SE(2) rotation normalization."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,protected-access
from pyrecest.backend import array, to_numpy
from pyrecest.filters.se2_ukf import _normalize_rotation_columns


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ in {"autograd", "numpy"},
    reason="Strict NumPy floating-point handling is NumPy-family specific",
)
class TestSE2UKFExtremeRotationNormalization(unittest.TestCase):
    def test_large_finite_rotation_columns_remain_normalized(self):
        max_float = np.finfo(np.float64).max
        rotation_samples = array(
            [
                [0.5 * max_float, 0.5 * max_float, 0.0],
                [0.5 * max_float, -0.5 * max_float, 0.0],
            ]
        )
        fallback = array([1.0, 0.0])

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = np.asarray(
                to_numpy(_normalize_rotation_columns(rotation_samples, fallback)),
                dtype=float,
            )

        expected = np.array(
            [
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 1.0],
                [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
            ]
        )
        npt.assert_allclose(normalized, expected, rtol=1e-14, atol=0.0)
        npt.assert_allclose(np.linalg.norm(normalized, axis=0), 1.0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
