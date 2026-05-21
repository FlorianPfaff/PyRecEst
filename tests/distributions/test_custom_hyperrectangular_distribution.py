import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend as backend
from pyrecest.backend import allclose, array, column_stack, linspace, meshgrid, ones
from pyrecest.distributions.custom_hyperrectangular_distribution import (
    CustomHyperrectangularDistribution,
)
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)


class TestCustomHyperrectangularDistribution(unittest.TestCase):
    def setUp(self):
        self.bounds = array([[1, 3], [2, 5]])
        self.hud = HyperrectangularUniformDistribution(self.bounds)
        self.cd = CustomHyperrectangularDistribution(self.hud.pdf, self.hud.bounds)

    def test_object_creation(self):
        """Test that a CustomHyperrectangularDistribution object is successfully created."""
        self.assertIsInstance(
            self.cd,
            CustomHyperrectangularDistribution,
            "CustomHyperrectangularDistribution object creation failed.",
        )

    def test_pdf_method(self):
        """Test that the pdf method returns correct values."""
        x_mesh, y_mesh = meshgrid(
            linspace(1.0, 3.0, 50), linspace(2.0, 5.0, 50), indexing="ij"
        )
        expected_pdf = 1.0 / 6.0 * ones(50**2)
        calculated_pdf = self.cd.pdf(column_stack((x_mesh.ravel(), y_mesh.ravel())))
        self.assertTrue(
            allclose(calculated_pdf, expected_pdf),
            "PDF calculated values do not match the expected values.",
        )

    def test_manifold_size_uses_one_row_per_dimension(self):
        self.assertEqual(self.hud.dim, 2)
        self.assertTrue(allclose(self.hud.get_manifold_size(), 6.0))

    def test_integrate_defaults_to_full_rectangular_bounds(self):
        self.assertAlmostEqual(float(self.hud.integrate()), 1.0, places=10)

    def test_one_dimensional_bounds_are_supported(self):
        cd = CustomHyperrectangularDistribution(lambda _xs: 0.5, array([0.0, 2.0]))

        self.assertEqual(cd.dim, 1)
        self.assertEqual(tuple(cd.bounds.shape), (1, 2))
        self.assertAlmostEqual(float(cd.get_manifold_size()), 2.0)
        self.assertAlmostEqual(cd.integrate(), 1.0)

    def test_bounds_rows_define_integration_intervals(self):
        cd = CustomHyperrectangularDistribution(
            lambda _xs: 1.0,
            array([[0.0, 2.0], [10.0, 13.0], [20.0, 24.0]]),
        )

        self.assertEqual(cd.dim, 3)
        self.assertAlmostEqual(float(cd.get_manifold_size()), 24.0)
        self.assertAlmostEqual(cd.integrate(), 24.0)

    def test_three_dimensional_bounds_set_dim_and_volume(self):
        dist = HyperrectangularUniformDistribution(
            array(
                [
                    [0.0, 2.0],
                    [1.0, 4.0],
                    [-2.0, 2.0],
                ]
            )
        )

        self.assertEqual(dist.dim, 3)
        self.assertTrue(allclose(dist.get_manifold_size(), 24.0))

    def test_normalize_verify_handles_scalar_integral(self):
        if backend.__backend_name__ != "numpy":  # pylint: disable=no-member
            self.skipTest("normalize currently supports the NumPy backend only")

        dist = CustomHyperrectangularDistribution(
            lambda xs: 2.0 * ones(xs.shape[0]), self.bounds
        )
        normalized = dist.normalize(verify=True)

        self.assertAlmostEqual(float(normalized.integrate()), 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
