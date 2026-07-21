import unittest
from math import sqrt

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions import SO3ProductDiracDistribution


class SO3ProductDiracMarginalIndexValidationTest(unittest.TestCase):
    def setUp(self):
        self.identity = array([0.0, 0.0, 0.0, 1.0])
        self.z_ninety = array([0.0, 0.0, sqrt(0.5), sqrt(0.5)])
        self.x_ninety = array([sqrt(0.5), 0.0, 0.0, sqrt(0.5)])

    def _make_distribution(self):
        return SO3ProductDiracDistribution(
            array(
                [
                    [self.identity, self.z_ninety, self.x_ninety],
                    [self.z_ninety, self.x_ninety, self.identity],
                ]
            ),
            array([0.25, 0.75]),
        )

    def test_marginalize_rotation_accepts_numpy_integer_scalar(self):
        dist = self._make_distribution()

        marginal = dist.marginalize_rotation(np.int64(2))

        npt.assert_allclose(marginal.d, dist.d[:, 2, :])
        npt.assert_allclose(marginal.w, dist.w)

    def test_marginalize_rotations_preserves_valid_order(self):
        dist = self._make_distribution()

        marginal = dist.marginalize_rotations(np.array([2, 0]))

        self.assertEqual(marginal.d.shape, (2, 2, 4))
        npt.assert_allclose(marginal.d[:, 0, :], dist.d[:, 2, :])
        npt.assert_allclose(marginal.d[:, 1, :], dist.d[:, 0, :])
        npt.assert_allclose(marginal.w, dist.w)

    def test_marginalize_rotations_accepts_scalar_selection(self):
        dist = self._make_distribution()

        marginal = dist.marginalize_rotations(np.array(1))

        self.assertEqual(marginal.d.shape, (2, 1, 4))
        npt.assert_allclose(marginal.d[:, 0, :], dist.d[:, 1, :])

    def test_marginalize_rotation_rejects_invalid_scalar_index(self):
        dist = self._make_distribution()

        for invalid in (True, 1.0, -1, 3, np.array([0])):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "rotation_index"):
                    dist.marginalize_rotation(invalid)

    def test_marginalize_rotations_rejects_invalid_index_sequences(self):
        dist = self._make_distribution()

        invalid_cases = ([], [True], [1.0], [-1], [3], [0, 0], [[0]], ["1"])
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "rotation_"):
                    dist.marginalize_rotations(invalid)


if __name__ == "__main__":
    unittest.main()
