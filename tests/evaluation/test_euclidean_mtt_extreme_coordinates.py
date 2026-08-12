"""Regression tests for overflow-safe Euclidean MTT distances."""

import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.evaluation.get_distance_function import get_distance_function


class EuclideanMttExtremeCoordinatesTest(unittest.TestCase):
    def test_opposite_extreme_coordinates_are_capped_without_overflow(self):
        max_float = np.finfo(float).max
        distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": 7.0},
        )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = distance(
                np.array([[-max_float]]),
                np.array([[max_float]]),
            )

        self.assertEqual(result, 7.0)

    def test_large_subcutoff_norm_preserves_distance(self):
        max_float = np.finfo(float).max
        distance = get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": max_float},
        )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = distance(
                np.array([[max_float / 2.0, max_float / 2.0]]),
                np.array([[0.0, 0.0]]),
            )

        npt.assert_allclose(
            result,
            max_float / np.sqrt(2.0),
            rtol=1e-15,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
