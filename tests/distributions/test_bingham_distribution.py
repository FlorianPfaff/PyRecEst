import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, to_numpy
from pyrecest.distributions import BinghamDistribution

from .test_von_mises_fisher_distribution import vectors_to_test_2d


class TestBinghamDistribution(unittest.TestCase):
    def setUp(self):
        """Setup BinghamDistribution instance for testing."""
        M = array(
            [[1 / 3, 2 / 3, -2 / 3], [-2 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]]
        )
        Z = array([-5.0, -3.0, 0.0])
        self.bd = BinghamDistribution(Z, M)

    def test_constructor_accepts_list_inputs(self):
        """Z and M can be supplied as ordinary Python lists."""
        M = [
            [1 / 3, 2 / 3, -2 / 3],
            [-2 / 3, 2 / 3, 1 / 3],
            [2 / 3, 1 / 3, 2 / 3],
        ]
        Z = [-5.0, -3.0, 0.0]

        bd = BinghamDistribution(Z, M)

        npt.assert_allclose(bd.Z, array(Z))
        npt.assert_allclose(bd.M, array(M))

    def test_constructor_rejects_invalid_parameters(self):
        """Invalid concentration vectors and basis matrices fail explicitly."""
        valid_Z = array([-5.0, -3.0, 0.0])
        valid_M = self.bd.M
        invalid_cases = [
            (valid_Z, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "square"),
            (array([[-5.0, -3.0, 0.0]]), valid_M, "1-D"),
            (array([-5.0, 0.0]), valid_M, "wrong length"),
            (array([-5.0, -3.0, 1.0]), valid_M, "zero"),
            (array([-3.0, -5.0, 0.0]), valid_M, "ascending"),
            (array([float("nan"), -3.0, 0.0]), valid_M, "finite"),
            (
                valid_Z,
                array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
                "orthogonal",
            ),
        ]

        for Z, M, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    BinghamDistribution(Z, M)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf(self):
        """Test pdf method with a fixed set of values."""
        expected_values = array(
            [
                0.0767812166360095,
                0.0145020985787277,
                0.0394207910410773,
                0.0267197897401937,
                0.0298598745474396,
                0.0298598745474396,
            ],
        )
        computed_values = self.bd.pdf(vectors_to_test_2d)
        npt.assert_array_almost_equal(
            computed_values,
            expected_values,
            err_msg="Expected and computed pdf values do not match.",
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf_accepts_list_inputs(self):
        """PDF evaluation points can be supplied as ordinary Python lists."""
        x = [
            [float(v) for v in vectors_to_test_2d[0]],
            [float(v) for v in vectors_to_test_2d[1]],
        ]

        npt.assert_allclose(self.bd.pdf(x), self.bd.pdf(array(x)))
        npt.assert_allclose(self.bd.pdf(x[0]), self.bd.pdf(array(x[0])))

    def test_pdf_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            self.bd.pdf([1.0, 0.0])

    def test_multiply_rejects_invalid_partner(self):
        lower_dim = BinghamDistribution(
            array([-1.0, 0.0]), array([[1.0, 0.0], [0.0, 1.0]])
        )

        with self.assertRaisesRegex(ValueError, "BinghamDistribution"):
            self.bd.multiply(object())
        with self.assertRaisesRegex(ValueError, "Dimensions"):
            self.bd.multiply(lower_dim)

    def test_compose_rejects_invalid_partner_or_dimension(self):
        lower_dim = BinghamDistribution(
            array([-1.0, 0.0]), array([[1.0, 0.0], [0.0, 1.0]])
        )

        with self.assertRaisesRegex(ValueError, "BinghamDistribution"):
            self.bd.compose(object())
        with self.assertRaisesRegex(ValueError, "Dimensions"):
            self.bd.compose(lower_dim)
        with self.assertRaisesRegex(ValueError, "Compose"):
            self.bd.compose(self.bd)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_spherical_moment_axis_and_sigma_points(self):
        """S2 Bingham derivatives should support moment-based methods."""
        expected_diagonal = array(
            [
                0.11516984173035563,
                0.1952701622411956,
                0.6895599960284488,
            ]
        )

        moment = self.bd.moment()
        moment_principal_frame = self.bd.M.T @ moment @ self.bd.M

        npt.assert_allclose(
            to_numpy(moment_principal_frame),
            to_numpy(diag(expected_diagonal)),
            rtol=1e-6,
            atol=1e-7,
        )

        axis = self.bd.mean_axis()
        expected_axis = self.bd.M[:, -1]
        axis_alignment = float(to_numpy(axis @ expected_axis))
        self.assertAlmostEqual(abs(axis_alignment), 1.0, places=6)

        samples, weights = self.bd.sample_deterministic()
        self.assertEqual(samples.shape, (3, 6))
        npt.assert_allclose(to_numpy(weights).sum(), 1.0, rtol=0, atol=1e-7)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sample_accepts_integer_like_count(self):
        """Scalar integer-like counts should be normalized before sampling."""
        samples = self.bd.sample(np.array(4.0))
        self.assertEqual(samples.shape, (4, 3))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sample_rejects_invalid_count(self):
        """Invalid counts should fail before Metropolis-Hastings allocation."""
        for invalid_n in (0, -1, 1.5, True, [3]):
            with self.subTest(n=invalid_n), self.assertRaises(ValueError):
                self.bd.sample(invalid_n)


if __name__ == "__main__":
    unittest.main()
