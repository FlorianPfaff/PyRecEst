import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, linalg, ones, random
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.custom_hyperspherical_distribution import (
    CustomHypersphericalDistribution,
)


class CustomHypersphericalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.vmf = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 10)
        self.custom_hyperspherical_distribution = (
            CustomHypersphericalDistribution.from_distribution(self.vmf)
        )

    def test_simple_distribution(self):
        """Test that pdf function returns the correct size and values for given points."""
        p = self.custom_hyperspherical_distribution.pdf(array([1.0, 0.0, 0.0]))
        numel_p = 1 if p.ndim == 0 else p.shape[0]
        self.assertEqual(numel_p, 1, "PDF size mismatch.")

        random.seed(10)
        points = random.normal(size=(100, 3))
        points /= linalg.norm(points, axis=1).reshape(-1, 1)

        self.assertTrue(
            allclose(
                self.custom_hyperspherical_distribution.pdf(points),
                self.vmf.pdf(points),
                atol=1e-5,
            ),
            "PDF values do not match.",
        )

    def test_pdf_accepts_list_inputs(self):
        dist = CustomHypersphericalDistribution(lambda xs: ones(xs.shape[:-1]), dim=2)
        points = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        self.assertTrue(allclose(dist.pdf(points[0]), dist.pdf(array(points[0]))))
        self.assertTrue(allclose(dist.pdf(points), dist.pdf(array(points))))

    def test_pdf_rejects_wrong_dimension(self):
        dist = CustomHypersphericalDistribution(lambda xs: 1.0, dim=2)

        with self.assertRaises(ValueError):
            dist.pdf([1.0, 0.0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integrate(self):
        """Test that the distribution integrates to 1."""
        self.assertAlmostEqual(
            self.custom_hyperspherical_distribution.integrate_numerically(),
            1,
            delta=1e-4,
            msg="Integration value mismatch.",
        )

    def test_from_distribution(self):
        """Test that the distribution can be created from another hyperspherical distribution."""
        dist = CustomHypersphericalDistribution.from_distribution(self.vmf)
        self.assertIsInstance(
            dist,
            CustomHypersphericalDistribution,
            "Type mismatch when creating from distribution.",
        )


if __name__ == "__main__":
    unittest.main()
