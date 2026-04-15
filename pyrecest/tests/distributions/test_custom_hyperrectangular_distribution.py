import unittest

# pylint: disable=no-name-in-module,no-member
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


if __name__ == "__main__":
    unittest.main()
