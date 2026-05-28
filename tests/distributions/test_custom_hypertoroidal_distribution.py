import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
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

    def test_to_custom_circular_preserves_scale_and_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs, 1, shift_by=[0.4], scale_by=2.5
        )

        circular = dist.to_custom_circular()
        xs = array([0.1, 0.2, 0.3])

        npt.assert_allclose(circular.pdf(xs), dist.pdf(xs))

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


if __name__ == "__main__":
    unittest.main()
