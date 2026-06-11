import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos
from pyrecest.distributions import (
    CustomHypertoroidalDistribution,
    CustomToroidalDistribution,
)


class CustomHypertoroidalDistributionTest(unittest.TestCase):
    def test_pdf_accepts_list_inputs_with_list_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs * 0.0 + 1.0, 1, shift_by=[0.1]
        )

        list_pdf = dist.pdf([0.1, 0.2])
        array_pdf = dist.pdf(array([0.1, 0.2]))

        self.assertEqual(list_pdf.shape, (2,))
        npt.assert_allclose(list_pdf, array_pdf)

    def test_pdf_accepts_multidimensional_list_inputs_with_list_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs[:, 0], 2, shift_by=[0.1, 0.2]
        )

        list_pdf = dist.pdf([[0.1, 0.2], [0.3, 0.4]])
        array_pdf = dist.pdf(array([[0.1, 0.2], [0.3, 0.4]]))

        npt.assert_allclose(list_pdf, array_pdf)

    def test_constructor_rejects_wrong_shift_shape(self):
        with self.assertRaisesRegex(ValueError, "shift_by"):
            CustomHypertoroidalDistribution(lambda xs: xs, 2, shift_by=[0.1])

    def test_shift_accepts_scalar_for_one_dimension(self):
        dist = CustomHypertoroidalDistribution(cos, 1)

        shifted = dist.shift(0.1)

        npt.assert_allclose(shifted.pdf(array([0.3])), dist.pdf(array([0.2])))

    def test_shift_accepts_list_vector(self):
        dist = CustomHypertoroidalDistribution(lambda xs: xs[:, 0] + xs[:, 1], 2)

        shifted = dist.shift([0.1, 0.2])

        npt.assert_allclose(
            shifted.pdf(array([[0.4, 0.6]])), dist.pdf(array([[0.3, 0.4]]))
        )

    def test_shift_rejects_wrong_shape(self):
        dist = CustomHypertoroidalDistribution(lambda xs: xs[:, 0], 2)

        with self.assertRaisesRegex(ValueError, "shift_by"):
            dist.shift([0.1])

    def test_to_custom_circular_preserves_scale_and_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs, 1, shift_by=[0.4], scale_by=2.5
        )

        circular = dist.to_custom_circular()
        xs = array([0.1, 0.2, 0.3])

        npt.assert_allclose(circular.pdf(xs), dist.pdf(xs))

    def test_to_custom_circular_rejects_multidimensional_distribution(self):
        dist = CustomHypertoroidalDistribution(lambda xs: xs[:, 0], 2)

        with self.assertRaisesRegex(ValueError, "dim == 1"):
            dist.to_custom_circular()

    def test_to_custom_toroidal_preserves_scale_and_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs[:, 0] + 2.0 * xs[:, 1],
            2,
            shift_by=[0.3, 0.4],
            scale_by=0.5,
        )

        toroidal = dist.to_custom_toroidal()
        xs = array([[0.1, 0.2], [0.5, 0.6]])

        self.assertIsInstance(toroidal, CustomToroidalDistribution)
        npt.assert_allclose(toroidal.pdf(xs), dist.pdf(xs))

    def test_to_custom_toroidal_rejects_wrong_dimension(self):
        dist = CustomHypertoroidalDistribution(cos, 1)

        with self.assertRaisesRegex(ValueError, "dim == 2"):
            dist.to_custom_toroidal()


if __name__ == "__main__":
    unittest.main()
