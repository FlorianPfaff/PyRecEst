import math
import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution


def _scalar(value):
    return float(np.asarray(to_numpy(value)).reshape(-1)[0])


class GaussianDistributionRegressionTest(unittest.TestCase):
    def test_standard_normal_pdf_and_log_pdf_match_reference_values(self):
        distribution = GaussianDistribution(array([0.0]), array([[1.0]]))

        log_pdf = _scalar(distribution.ln_pdf(array([0.0])))
        pdf = _scalar(distribution.pdf(array([0.0])))

        self.assertAlmostEqual(log_pdf, -0.5 * math.log(2.0 * math.pi), places=12)
        self.assertAlmostEqual(pdf, 1.0 / math.sqrt(2.0 * math.pi), places=12)

    def test_gaussian_product_matches_closed_form_scalar_reference(self):
        first = GaussianDistribution(array([0.0]), array([[1.0]]))
        second = GaussianDistribution(array([1.0]), array([[4.0]]))

        product = first.multiply(second)

        npt.assert_allclose(to_numpy(product.mu), np.array([0.2]), rtol=1e-12, atol=1e-12)
        npt.assert_allclose(to_numpy(product.C), np.array([[0.8]]), rtol=1e-12, atol=1e-12)

    def test_invalid_covariance_shape_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaussianDistribution(array([0.0, 1.0]), array([[1.0]]))

    def test_invalid_marginal_dimensions_raise_value_error(self):
        distribution = GaussianDistribution(array([0.0, 1.0]), array([[1.0, 0.0], [0.0, 1.0]]))

        with self.assertRaises(ValueError):
            distribution.marginalize_out([2])


if __name__ == "__main__":
    unittest.main()
