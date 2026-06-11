import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones, pi
from pyrecest.distributions import (
    CustomHyperhemisphericalDistribution,
    HypersphericalUniformDistribution,
)


class CustomHyperhemisphericalDistributionTest(unittest.TestCase):
    def test_pdf_accepts_list_inputs(self):
        dist = CustomHyperhemisphericalDistribution(lambda xs: ones(xs.shape[:-1]), 2)
        points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        npt.assert_allclose(dist.pdf(points[0]), dist.pdf(array(points[0])))
        npt.assert_allclose(dist.pdf(points), dist.pdf(array(points)))

    def test_pdf_rejects_wrong_dimension(self):
        dist = CustomHyperhemisphericalDistribution(lambda xs: 1.0, 2)

        with self.assertRaises(ValueError):
            dist.pdf([1.0, 0.0])

    def test_pdf_rejects_matrix_callback_output(self):
        dist = CustomHyperhemisphericalDistribution(lambda xs: ones((2, 2)), 2)

        with self.assertRaisesRegex(ValueError, "Output format"):
            dist.pdf([1.0, 0.0, 0.0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Numerical integration is only supported for the NumPy backend",
    )
    def test_from_full_hypersphere_distribution_normalizes_callable(self):
        source = HypersphericalUniformDistribution(1)

        dist = CustomHyperhemisphericalDistribution.from_distribution(source)

        self.assertEqual(dist.dim, source.dim)
        npt.assert_allclose(dist.pdf(array([1.0, 0.0])), 1.0 / pi)
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
