import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, exp, mod, pi
from pyrecest.distributions import HypertoroidalWNDistribution


class TestHypertoroidalWNDistribution(unittest.TestCase):
    def test_pdf(self):
        mu = array([1, 2])
        C = array([[0.5, 0.1], [0.1, 0.3]])

        hwn = HypertoroidalWNDistribution(mu, C)

        xa = array([[0, 1, 2], [1, 2, 3]]).T
        pdf_values = hwn.pdf(xa)

        expected_values = array(
            [0.0499028191873498, 0.425359477472412, 0.0499028191873498]
        )
        npt.assert_allclose(pdf_values, expected_values, rtol=2e-6)

    def test_pdf_accepts_scalar_and_1d_batch_for_one_dimensional_distribution(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        scalar_value = dist.pdf(0.3)
        batch_values = dist.pdf([0.3, 0.4])

        self.assertEqual(scalar_value.shape, (1,))
        self.assertEqual(batch_values.shape, (2,))
        npt.assert_allclose(scalar_value, dist.pdf(array([[0.3]])))
        npt.assert_allclose(batch_values, dist.pdf(array([[0.3], [0.4]])))

    def test_pdf_accepts_list_single_point_for_multidimensional_distribution(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.3]])

        from_list = dist.pdf([1.0, 2.0])
        from_matrix = dist.pdf([[1.0, 2.0]])

        self.assertEqual(from_list.shape, (1,))
        npt.assert_allclose(from_list, from_matrix)

    def test_pdf_accepts_python_sequence_parameters_and_query_points(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.3]])

        query_points = [[1.0, 2.0], [1.1, 2.1]]

        npt.assert_allclose(dist.pdf(query_points), dist.pdf(array(query_points)))

    def test_pdf_interprets_one_dimensional_sequences_as_multiple_scalar_points(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        query_points = [0.1, 0.2, 0.3]
        expected_points = array([[0.1], [0.2], [0.3]])

        values = dist.pdf(query_points)

        self.assertEqual(values.shape, (3,))
        npt.assert_allclose(values, dist.pdf(expected_points))

    def test_scalar_pdf_accepts_scalar_and_sequence_inputs(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        scalar_pdf = dist.pdf(0.2)
        list_pdf = dist.pdf([0.2, 0.4])
        matrix_pdf = dist.pdf(array([[0.2], [0.4]]))

        self.assertEqual(scalar_pdf.shape, (1,))
        self.assertEqual(list_pdf.shape, (2,))
        npt.assert_allclose(list_pdf, matrix_pdf)
        npt.assert_allclose(scalar_pdf, matrix_pdf[:1])

    def test_pdf_accepts_numpy_integer_series_order(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        values = dist.pdf(0.2, m=np.int64(2))

        self.assertEqual(values.shape, (1,))
        self.assertGreater(float(values[0]), 0.0)

    def test_pdf_rejects_invalid_series_order(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        for m in (True, -1, 1.5, [1]):
            with self.subTest(m=m):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    dist.pdf(0.2, m=m)

    def test_constructor_rejects_invalid_parameters(self):
        invalid_cases = [
            ([[0.0, 1.0]], [[1.0]], "one-dimensional"),
            ([0.0], [1.0, 2.0], "shape"),
            ([0.0], [[float("inf")]], "finite"),
            ([0.0, 1.0], [[1.0, 0.2], [0.1, 1.0]], "symmetric"),
            ([0.0, 1.0], [[1.0, 2.0], [2.0, 1.0]], "positive definite"),
            ([0.0], [[1.0, 0.0], [0.0, 1.0]], "shape"),
        ]

        for mu, C, message in invalid_cases:
            with self.subTest(mu=mu, C=C):
                with self.assertRaisesRegex(ValueError, message):
                    HypertoroidalWNDistribution(mu, C)

    def test_vector_pdf_accepts_single_point_sequence(self):
        dist = HypertoroidalWNDistribution(
            array([0.3, 0.4]), array([[0.7, 0.0], [0.0, 0.5]])
        )

        one_point_pdf = dist.pdf([0.2, 0.5])
        matrix_pdf = dist.pdf(array([[0.2, 0.5]]))

        self.assertEqual(one_point_pdf.shape, (1,))
        npt.assert_allclose(one_point_pdf, matrix_pdf)

    def test_scalar_parameters_are_stored_as_vector_and_matrix(self):
        dist = HypertoroidalWNDistribution(array(0.3), array(0.7))

        self.assertEqual(dist.dim, 1)
        self.assertEqual(dist.mu.shape, (1,))
        self.assertEqual(dist.C.shape, (1, 1))
        npt.assert_allclose(dist.mu, array([0.3]))
        npt.assert_allclose(dist.C, array([[0.7]]))
        npt.assert_allclose(
            dist.trigonometric_moment(1), exp(1j * array([0.3]) - 0.7 / 2)
        )

    def test_sample_validates_count_before_backend_call(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.6]])

        samples = dist.sample(np.int64(4))

        self.assertEqual(samples.shape, (4, 2))
        npt.assert_allclose(samples, mod(samples, 2.0 * pi))

        for n in (True, 1.5, 0, -1):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)

    def test_operations_reject_dimension_mismatches(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.6]])
        one_dimensional = HypertoroidalWNDistribution(0.3, 0.7)

        with self.assertRaisesRegex(ValueError, "shape"):
            dist.set_mean([0.1])
        with self.assertRaisesRegex(ValueError, "shape"):
            dist.set_mode([0.1])
        with self.assertRaisesRegex(ValueError, "Dimensions"):
            dist.convolve(one_dimensional)

    def test_trigonometric_moment_rejects_invalid_order(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.6]])

        npt.assert_allclose(
            dist.trigonometric_moment(np.int64(1)), dist.trigonometric_moment(1)
        )

        for n in (True, 1.5, [1]):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "integer"):
                    dist.trigonometric_moment(n)

    def test_shift_accepts_scalar_for_one_dimensional_distribution(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        shifted = dist.shift(0.25)

        self.assertEqual(shifted.dim, 1)
        npt.assert_allclose(shifted.mu, array([0.55]))
        npt.assert_allclose(dist.mu, array([0.3]))

    def test_shift_accepts_list_for_multidimensional_distribution(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.6]])

        shifted = dist.shift([0.25, -0.5])

        npt.assert_allclose(shifted.mu, mod(array([1.25, 1.5]), 2.0 * pi))
        npt.assert_allclose(dist.mu, array([1.0, 2.0]))

    def test_shift_returns_copy_without_mutating_original(self):
        mu = array([1.0, 2.0])
        C = array([[0.5, 0.1], [0.1, 0.6]])
        shift_by = array([0.25, 2.0 * pi - 0.5])
        dist = HypertoroidalWNDistribution(mu, C)

        shifted = dist.shift(shift_by)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, mod(mu + shift_by, 2.0 * pi))
        npt.assert_allclose(shifted.C, dist.C)

    def test_shift_accepts_plain_python_scalars_and_sequences(self):
        scalar_dist = HypertoroidalWNDistribution(0.3, 0.7)

        scalar_shifted = scalar_dist.shift(0.5)

        npt.assert_allclose(scalar_shifted.mu, array([0.8]))
        npt.assert_allclose(scalar_dist.mu, array([0.3]))

        mu = array([1.0, 2.0])
        C = array([[0.5, 0.1], [0.1, 0.6]])
        dist = HypertoroidalWNDistribution(mu, C)

        shifted = dist.shift([0.25, -0.5])

        npt.assert_allclose(shifted.mu, mod(mu + array([0.25, -0.5]), 2.0 * pi))
        npt.assert_allclose(dist.mu, mu)

    def test_set_mode_wraps_to_fundamental_domain(self):
        dist = HypertoroidalWNDistribution(
            array([0.3, 0.4]), array([[0.7, 0.0], [0.0, 0.5]])
        )

        updated = dist.set_mode(array([2.0 * pi + 0.1, -0.2]))

        npt.assert_allclose(updated.mu, mod(array([0.1, -0.2]), 2.0 * pi))
        npt.assert_allclose(dist.mu, array([0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()
