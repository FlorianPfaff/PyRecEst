import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.so3_dirac_distribution import SO3DiracDistribution
from pyrecest.distributions.so3_product_dirac_distribution import (
    SO3ProductDiracDistribution,
)
from pyrecest.distributions.so3_product_tangent_gaussian_distribution import (
    SO3ProductTangentGaussianDistribution,
)
from pyrecest.distributions.so3_tangent_gaussian_distribution import (
    SO3TangentGaussianDistribution,
)


class SO3ConversionValidationTest(unittest.TestCase):
    def test_so3_tangent_gaussian_rejects_invalid_covariance_regularization(self):
        base = array([0.0, 0.0, 0.0, 1.0])
        rotations = SO3TangentGaussianDistribution.exp_map(
            array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]),
            base=base,
        )
        particles = SO3DiracDistribution(rotations, array([0.5, 0.5]))

        for covariance_regularization in (
            True,
            -1.0e-6,
            float("nan"),
            float("inf"),
            "1e-3",
        ):
            with self.subTest(covariance_regularization=covariance_regularization):
                with self.assertRaisesRegex(
                    ValueError,
                    "covariance_regularization must be a nonnegative finite scalar",
                ):
                    convert_distribution(
                        particles,
                        "so3_tangent_gaussian",
                        covariance_regularization=covariance_regularization,
                    )

    def test_so3_product_tangent_gaussian_rejects_invalid_covariance_regularization(
        self,
    ):
        mean = array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        rotations = SO3ProductTangentGaussianDistribution.exp_map(
            array(
                [
                    [0.01, 0.0, 0.0, 0.0, 0.02, 0.0],
                    [-0.01, 0.0, 0.0, 0.0, -0.02, 0.0],
                ]
            ),
            base=mean,
            num_rotations=2,
        )
        particles = SO3ProductDiracDistribution(rotations, array([0.5, 0.5]))

        for covariance_regularization in (
            True,
            -1.0e-6,
            float("nan"),
            float("inf"),
            "1e-3",
        ):
            with self.subTest(covariance_regularization=covariance_regularization):
                with self.assertRaisesRegex(
                    ValueError,
                    "covariance_regularization must be a nonnegative finite scalar",
                ):
                    convert_distribution(
                        particles,
                        "so3_product_tangent_gaussian",
                        covariance_regularization=covariance_regularization,
                    )


if __name__ == "__main__":
    unittest.main()
