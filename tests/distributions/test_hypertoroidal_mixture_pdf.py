import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import (
    HypertoroidalMixture,
)
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


class HypertoroidalMixturePdfTest(unittest.TestCase):
    def test_pdf_accepts_scalar_for_one_dimensional_mixture(self):
        first = HypertoroidalWrappedNormalDistribution(array([0.0]), array([[0.4]]))
        second = HypertoroidalWrappedNormalDistribution(array([0.8]), array([[0.7]]))
        mixture = HypertoroidalMixture([first, second], array([0.25, 0.75]))

        result = mixture.pdf(0.3)
        expected = 0.25 * first.pdf(array([0.3])) + 0.75 * second.pdf(array([0.3]))

        self.assertEqual(result.shape, (1,))
        self.assertTrue(allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
