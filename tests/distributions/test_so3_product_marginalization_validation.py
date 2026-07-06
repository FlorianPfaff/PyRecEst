import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import SO3ProductTangentGaussianDistribution
from tests.distributions.so3_test_helpers import ATOL, x_quaternion, z_quaternion


def _two_rotation_distribution():
    covariance = diag(array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    return SO3ProductTangentGaussianDistribution(
        array([z_quaternion(0.0), x_quaternion(np.pi / 2.0)]),
        covariance,
    )


class SO3ProductTangentGaussianMarginalizationValidationTest(unittest.TestCase):
    def test_marginalize_rotations_accepts_numpy_scalar_integer(self):
        dist = _two_rotation_distribution()

        marginal = dist.marginalize_rotations(np.int64(1))

        self.assertEqual(marginal.num_rotations, 1)
        npt.assert_allclose(marginal.mean()[0], x_quaternion(np.pi / 2.0), atol=ATOL)
        npt.assert_allclose(
            marginal.covariance(),
            diag(array([0.4, 0.5, 0.6])),
            atol=ATOL,
        )

    def test_marginalize_rotations_rejects_invalid_indices(self):
        dist = _two_rotation_distribution()

        invalid_indices = (
            True,
            [True],
            [],
            [0, 0],
            [-1],
            [2],
            [1.0],
            [[0]],
        )

        for rotation_indices in invalid_indices:
            with self.subTest(rotation_indices=rotation_indices):
                with self.assertRaisesRegex(ValueError, "rotation_indices"):
                    dist.marginalize_rotations(rotation_indices)


if __name__ == "__main__":
    unittest.main()
