import unittest

import numpy.testing as npt
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    all,
    allclose,
    array,
    column_stack,
    cos,
    diff,
    exp,
    isnan,
    linspace,
    meshgrid,
    ones_like,
    pi,
    random,
    sin,
    sqrt,
    zeros,
)
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.abstract_spherical_distribution import (
    AbstractSphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)


class SphericalHarmonicsDistributionComplexTest(unittest.TestCase):
    def setUp(self):
        random.seed(1)
        coeff_rand = random.uniform(size=9)
        self.unnormalized_coeffs = array(
            [
                [coeff_rand[0], float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                [
                    coeff_rand[1] + 1j * coeff_rand[2],
                    coeff_rand[3],
                    -coeff_rand[1] + 1j * coeff_rand[2],
                    float("NaN"),
                    float("NaN"),
                ],
                [
                    coeff_rand[4] + 1j * coeff_rand[5],
                    coeff_rand[6] + 1j * coeff_rand[7],
                    coeff_rand[8],
                    -coeff_rand[6] + 1j * coeff_rand[7],
                    coeff_rand[4] - 1j * coeff_rand[5],
                ],
            ]
        )

    def test_mormalization_error(self):
        self.assertRaises(ValueError, SphericalHarmonicsDistributionComplex, array(0.0))

    def test_normalization(self):
        with self.assertWarns(Warning):
            shd = SphericalHarmonicsDistributionComplex(self.unnormalized_coeffs)

        self.assertAlmostEqual(shd.integrate(), 1, delta=1e-5)

        # Enforce unnormalized coefficients and compare ratio
        phi, theta = (
            random.uniform(size=(1, 10)) * 2.0 * pi,
            random.uniform(size=(1, 10)) * pi - pi / 2.0,
        )
        x, y, z = array([cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta)])
        vals_normalized = shd.pdf(column_stack([x, y, z]))
        shd.coeff_mat = self.unnormalized_coeffs
        vals_unnormalized = shd.pdf(column_stack([x, y, z]))
        self.assertTrue(
            allclose(
                diff(vals_normalized / vals_unnormalized),
                zeros(vals_normalized.shape[0] - 1),
                atol=1e-6,
            )
        )

    @parameterized.expand([("identity",), ("sqrt",)])
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integral_analytical(self, transformation):
        """Test if the analytical integral is equal to the numerical integral"""
        random.seed(10)
        coeff_rand = random.uniform(size=(1, 9))
        unnormalized_coeffs = array(
            [
                [
                    coeff_rand[0, 0],
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                    float("NaN"),
                ],
                [
                    coeff_rand[0, 1] + 1j * coeff_rand[0, 2],
                    coeff_rand[0, 3],
                    -coeff_rand[0, 1] + 1j * coeff_rand[0, 2],
                    float("NaN"),
                    float("NaN"),
                ],
                [
                    coeff_rand[0, 4] + 1j * coeff_rand[0, 5],
                    coeff_rand[0, 6] + 1j * coeff_rand[0, 7],
                    coeff_rand[0, 8],
                    -coeff_rand[0, 6] + 1j * coeff_rand[0, 7],
                    coeff_rand[0, 4] - 1j * coeff_rand[0, 5],
                ],
            ]
        )
        # First initialize and overwrite afterward to prevent normalization
        shd = SphericalHarmonicsDistributionComplex(
            array([[1.0, float("NaN"), float("NaN")], [0.0, 0.0, 0.0]])
        )
        shd.coeff_mat = unnormalized_coeffs
        shd.transformation = transformation
        int_val_num = shd.integrate_numerically()
        int_val_ana = shd.integrate()
        npt.assert_almost_equal(int_val_ana, int_val_num)

    def test_truncation(self):
        shd = SphericalHarmonicsDistributionComplex(self.unnormalized_coeffs)

        with self.assertWarns(UserWarning):
            shd2 = shd.truncate(4)
        self.assertEqual(shd2.coeff_mat.shape, (5, 9))
        self.assertTrue(all(isnan(shd2.coeff_mat[4, :]) | (shd2.coeff_mat[4, :] == 0)))
        shd3 = shd.truncate(5)
        self.assertEqual(shd3.coeff_mat.shape, (6, 11))
        self.assertTrue(
            all(
                isnan(shd3.coeff_mat[5:6, :]) | (shd3.coeff_mat[5:6, :] == 0),
                axis=(0, 1),
            )
        )
        shd4 = shd2.truncate(3)
        self.assertEqual(shd4.coeff_mat.shape, (4, 7))
        shd5 = shd3.truncate(3)
        self.assertEqual(shd5.coeff_mat.shape, (4, 7))

        phi, theta = (
            random.uniform(size=10) * 2 * pi,
            random.uniform(size=10) * pi,
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        self.assertTrue(
            allclose(
                shd2.pdf(column_stack((x, y, z))),
                shd.pdf(column_stack((x, y, z))),
                atol=1e-6,
            )
        )
        self.assertTrue(
            allclose(
                shd3.pdf(column_stack((x, y, z))),
                shd.pdf(column_stack((x, y, z))),
                atol=1e-6,
            )
        )
        self.assertTrue(
            allclose(
                shd4.pdf(column_stack((x, y, z))),
                shd.pdf(column_stack((x, y, z))),
                atol=1e-6,
            )
        )
        self.assertTrue(
            allclose(
                shd5.pdf(column_stack((x, y, z))),
                shd.pdf(column_stack((x, y, z))),
                atol=1e-6,
            )
        )

    @parameterized.expand(
        [
            # First, the basis functions that only yield real values are tested
            (
                "testl0m0",
                array(
                    [
                        [1.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                lambda _, _1, z: ones_like(z) * sqrt(1.0 / (4.0 * pi)),
            ),
            (
                "testl1m0",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 1.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                lambda _, _1, z: sqrt(3.0 / (4.0 * pi)) * z,
            ),
            (
                "testl2m0",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 1.0, 0.0, 0.0],
                    ]
                ),
                lambda x, y, z: 1.0 / 4.0 * sqrt(5 / pi) * (2.0 * z**2 - x**2 - y**2),
            ),
            (
                "testl3m0",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * sqrt(7 / pi)
                * (z * (2 * z**2 - 3 * x**2 - 3 * y**2)),
            ),
            # For the other basis functions, complex values would be obtained.
            # Hence, combinations of complex basis function are used that are equal
            # to complex basis functions
            (
                "test_l1mneg1real",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [
                            1j * sqrt(1 / 2),
                            0.0,
                            1j * sqrt(1 / 2),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda _, y, _1: sqrt(3 / (4 * pi)) * y,
            ),
            (
                "test_l1m1real",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [sqrt(1 / 2), 0.0, -sqrt(1 / 2), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda x, _, _1: sqrt(3 / (4 * pi)) * x,
            ),
            (
                "test_l2mneg2real",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1j * sqrt(1 / 2), 0.0, 0.0, 0.0, -1j * sqrt(1 / 2)],
                    ]
                ),
                lambda x, y, _: 1 / 2 * sqrt(15 / pi) * x * y,
            ),
            (
                "test_l2mneg1real",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1j * sqrt(1 / 2), 0.0, 1j * sqrt(1 / 2), 0],
                    ]
                ),
                lambda _, y, z: 1 / 2 * sqrt(15 / pi) * y * z,
            ),
            (
                "test_l2m1real",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, sqrt(1 / 2), 0.0, -sqrt(1 / 2), 0],
                    ]
                ),
                lambda x, _, z: 1 / 2 * sqrt(15 / pi) * x * z,
            ),
            (
                "test_l2m2real",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [sqrt(1 / 2), 0.0, 0.0, 0.0, sqrt(1 / 2)],
                    ]
                ),
                lambda x, y, _: 1 / 4 * sqrt(15 / pi) * (x**2 - y**2),
            ),
            (
                "test_l3mneg3real",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1j / sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0, 1j / sqrt(2)],
                    ]
                ),
                lambda x, y, z: 1.0
                / 4.0
                * sqrt(35.0 / (2.0 * pi))
                * y
                * (3.0 * x**2 - y**2),
            ),
            (
                "test_l3mneg2real",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1j / sqrt(2), 0.0, 0.0, 0.0, -1j / sqrt(2), 0],
                    ]
                ),
                lambda x, y, z: 1 / 2 * sqrt(105 / pi) * x * y * z,
            ),
            (
                "test_l3mneg1real",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 1j / sqrt(2), 0.0, 1j / sqrt(2), 0.0, 0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * sqrt(21 / (2 * pi))
                * y
                * (4 * z**2 - x**2 - y**2),
            ),
            (
                "test_l3m1real",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 1 / sqrt(2), 0.0, -1 / sqrt(2), 0.0, 0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * sqrt(21 / (2 * pi))
                * x
                * (4 * z**2 - x**2 - y**2),
            ),
            (
                "test_l3m2real",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1 / sqrt(2), 0.0, 0.0, 0.0, 1 / sqrt(2), 0],
                    ]
                ),
                lambda x, y, z: 1 / 4 * sqrt(105 / pi) * z * (x**2 - y**2),
            ),
            (
                "test_l3m3real",
                array(
                    [
                        [
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1 / sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0, -1 / sqrt(2)],
                    ]
                ),
                lambda x, y, z: 1 / 4 * sqrt(35 / (2 * pi)) * x * (x**2 - 3 * y**2),
            ),
        ]
    )
    def test_basis_function(self, _, coeff_mat, expected_func):
        shd = SphericalHarmonicsDistributionComplex(1.0 / sqrt(4.0 * pi))
        shd.coeff_mat = coeff_mat
        phi, theta = meshgrid(linspace(0.0, 2.0 * pi, 10), linspace(0.0, pi, 10))
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
        npt.assert_allclose(
            shd.pdf(column_stack([x, y, z])), expected_func(x, y, z), atol=1e-6
        )

    @parameterized.expand(
        [
            # Test complex basis functions
            (
                "testl1mneg1_cart",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [1, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda x, y, _: 0.5 * sqrt(3 / (2 * pi)) * (x - 1j * y),
            ),
            (
                "testl1m1_cart",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 1, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda x, y, _: -0.5 * sqrt(3 / (2 * pi)) * (x + 1j * y),
            ),
            (
                "testl2mneg2_cart",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda x, y, _: 0.25 * sqrt(15 / (2 * pi)) * (x - 1j * y) ** 2,
            ),
            (
                "testl2mneg1_cart",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1, 0.0, 0.0, 0],
                    ]
                ),
                lambda x, y, z: 0.5 * sqrt(15 / (2 * pi)) * (x - 1j * y) * z,
            ),
            (
                "testl2m1_cart",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 1, 0],
                    ]
                ),
                lambda x, y, z: -0.5 * sqrt(15 / (2 * pi)) * (x + 1j * y) * z,
            ),
            (
                "testl2m2_cart",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 1],
                    ]
                ),
                lambda x, y, _: 0.25 * sqrt(15 / (2 * pi)) * (x + 1j * y) ** 2,
            ),
            # For spherical coordinates
            (
                "testl1mneg1_sph",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [1, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * sqrt(3 / (2 * pi))
                * sin(theta)
                * exp(-1j * phi),
            ),
            (
                "testl1m1_sph",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 1, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * sqrt(3 / (2 * pi))
                * sin(theta)
                * exp(1j * phi),
            ),
            (
                "testl2mneg2_sph",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: 0.25
                * sqrt(15 / (2 * pi))
                * sin(theta) ** 2
                * exp(-2j * phi),
            ),
            (
                "testl2mneg1_sph",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * sqrt(15 / (2 * pi))
                * sin(theta)
                * cos(theta)
                * exp(-1j * phi),
            ),
            (
                "testl2m1_sph",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 1, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * sqrt(15 / (2 * pi))
                * sin(theta)
                * cos(theta)
                * exp(1j * phi),
            ),
            (
                "testl2m2_sph",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 1],
                    ]
                ),
                lambda phi, theta: 0.25
                * sqrt(15 / (2 * pi))
                * sin(theta) ** 2
                * exp(2j * phi),
            ),
            (
                "testl1mneg1_sphconv_inclination",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [1, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * sqrt(3 / (2 * pi))
                * sin(theta)
                * exp(-1j * phi),
            ),
            (
                "testl1m1_sphconv_inclination",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 1, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * sqrt(3 / (2 * pi))
                * sin(theta)
                * exp(1j * phi),
            ),
            (
                "testl2mneg2_sphconv_inclination",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1, 0.0, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: 0.25
                * sqrt(15 / (2 * pi))
                * sin(theta) ** 2
                * exp(-2j * phi),
            ),
            (
                "testl2mneg1_sphconv_inclination",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1, 0.0, 0.0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * sqrt(15 / (2 * pi))
                * sin(theta)
                * cos(theta)
                * exp(-1j * phi),
            ),
            (
                "testl2m1_sphconv_inclination",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 1, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * sqrt(15 / (2 * pi))
                * sin(theta)
                * cos(theta)
                * exp(1j * phi),
            ),
            (
                "testl2m2_sphconv_inclination",
                array(
                    [
                        [0.0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 1],
                    ]
                ),
                lambda phi, theta: 0.25
                * sqrt(15 / (2 * pi))
                * sin(theta) ** 2
                * exp(2j * phi),
            ),
        ]
    )
    def test_basis_function_complex(self, name, coeff_mat, expected_func):
        shd = SphericalHarmonicsDistributionComplex(1 / sqrt(4 * pi), assert_real=False)
        shd.coeff_mat = coeff_mat
        phi, theta = meshgrid(linspace(0.0, 2 * pi, 10), linspace(-pi / 2, pi / 2, 10))
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())

        vals_to_test = shd.pdf(column_stack([x, y, z]))
        if name.endswith("cart"):
            expected_func_vals = expected_func(x, y, z)
        elif name.endswith("sph"):
            expected_func_vals = expected_func(phi.ravel(), theta.ravel())
        elif name.endswith("inclination"):
            phi, theta = AbstractSphericalDistribution.cart_to_sph(
                x, y, z, mode="inclination"
            )
            expected_func_vals = expected_func(phi, theta)
        else:
            raise ValueError("Unknown test case")

        npt.assert_allclose(vals_to_test, expected_func_vals, atol=1e-6)

    @parameterized.expand(
        [
            (
                "l0m0",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
            ),
            (
                "l1mneg1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [
                            1j * sqrt(1 / 2),
                            0.0,
                            1j * sqrt(1 / 2),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
            ),
            (
                "l1m0",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 1, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
            ),
            (
                "l1m1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [sqrt(1 / 2), 0.0, -sqrt(1 / 2), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 0.0, 0],
                    ]
                ),
            ),
            (
                "l2mneg2",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1j * sqrt(1 / 2), 0.0, 0.0, 0.0, -1j * sqrt(1 / 2)],
                    ]
                ),
            ),
            (
                "l2mneg1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1j * sqrt(1 / 2), 0.0, 1j * sqrt(1 / 2), 0],
                    ]
                ),
            ),
            (
                "l2m0",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 1, 0.0, 0],
                    ]
                ),
            ),
            (
                "l2m1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, sqrt(1 / 2), 0.0, -sqrt(1 / 2), 0],
                    ]
                ),
            ),
            (
                "l2m2",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [sqrt(1 / 2), 0.0, 0.0, 0.0, sqrt(1 / 2)],
                    ]
                ),
            ),
            (
                "l3mneg3",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1j / sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0, 1j / sqrt(2)],
                    ]
                ),
            ),
            (
                "l3mneg2",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1j / sqrt(2), 0.0, 0.0, 0.0, -1j / sqrt(2), 0],
                    ]
                ),
            ),
            (
                "l3mneg1",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 1j / sqrt(2), 0.0, 1j / sqrt(2), 0.0, 0],
                    ]
                ),
            ),
            (
                "l3m0",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 0.0, 1, 0.0, 0.0, 0],
                    ]
                ),
            ),
            (
                "l3m1",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 0.0, 1 / sqrt(2), 0.0, -1 / sqrt(2), 0.0, 0],
                    ]
                ),
            ),
            (
                "l3m2",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [0.0, 1 / sqrt(2), 0.0, 0.0, 0.0, 1 / sqrt(2), 0],
                    ]
                ),
            ),
            (
                "l3m3",
                array(
                    [
                        [
                            1,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [
                            0.0,
                            0.0,
                            0.0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0.0, 0.0, 0.0, 0.0, 0.0, float("NaN"), float("NaN")],
                        [1 / sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0, -1 / sqrt(2)],
                    ]
                ),
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_conversion(self, _, coeff_mat):
        shd = SphericalHarmonicsDistributionComplex(coeff_mat)
        rshd = shd.to_spherical_harmonics_distribution_real()
        phi, theta = meshgrid(linspace(0.0, 2 * pi, 10), linspace(-pi / 2, pi / 2, 10))
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
        npt.assert_allclose(
            rshd.pdf(column_stack((x, y, z))),
            shd.pdf(column_stack((x, y, z))),
            atol=1e-6,
        )

    @parameterized.expand(
        [
            (
                "shd_x",
                array(
                    [[1, float("NaN"), float("NaN")], [sqrt(1 / 2), 0.0, -sqrt(1 / 2)]]
                ),
                array([1, 0.0, 0]),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_y",
                array(
                    [
                        [1, float("NaN"), float("NaN")],
                        [1j * sqrt(1 / 2), 0.0, 1j * sqrt(1 / 2)],
                    ]
                ),
                array([0.0, 1, 0]),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_z",
                array([[1, float("NaN"), float("NaN")], [0.0, 1, 0]]),
                array([0.0, 0.0, 1]),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_xy",
                array(
                    [
                        [1, float("NaN"), float("NaN")],
                        [
                            sqrt(1.0 / 2.0) + 1j * sqrt(1.0 / 2.0),
                            0.0,
                            -sqrt(1.0 / 2.0) + 1j * sqrt(1.0 / 2.0),
                        ],
                    ]
                ),
                array([1.0, 1.0, 0.0]) / sqrt(2.0),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_xz",
                array(
                    [
                        [1.0, float("NaN"), float("NaN")],
                        [sqrt(1 / 2), 1, -sqrt(1.0 / 2.0)],
                    ]
                ),
                array([1.0, 0.0, 1.0]) / sqrt(2.0),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_yz",
                array(
                    [
                        [1, float("NaN"), float("NaN")],
                        [1j * sqrt(1 / 2), 1, 1j * sqrt(1 / 2)],
                    ]
                ),
                array([0.0, 1.0, 1.0]) / sqrt(2.0),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "numerical_shd_x",
                [[1, float("NaN"), float("NaN")], [sqrt(1 / 2), 0.0, -sqrt(1 / 2)]],
                [1, 0.0, 0.0],
                SphericalHarmonicsDistributionComplex.mean_direction_numerical,
            ),
            (
                "numerical_shd_y",
                [
                    [1.0, float("NaN"), float("NaN")],
                    [1j * sqrt(1 / 2), 0.0, 1j * sqrt(1.0 / 2.0)],
                ],
                [0.0, 1.0, 0.0],
                SphericalHarmonicsDistributionComplex.mean_direction_numerical,
            ),
            (
                "numerical_shd_z",
                [[1.0, float("NaN"), float("NaN")], [0.0, 1.0, 0]],
                [0.0, 0.0, 1.0],
                SphericalHarmonicsDistributionComplex.mean_direction_numerical,
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_mean_direction(self, _, input_array, expected_output, fun_to_test):
        shd = SphericalHarmonicsDistributionComplex(array(input_array))
        npt.assert_allclose(fun_to_test(shd), expected_output, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_from_distribution_via_integral_vmf(self):
        # Test approximating a VMF
        dist = VonMisesFisherDistribution(
            array([-1.0, -1.0, 0.0]) / sqrt(2.0), array(1.0)
        )
        shd = SphericalHarmonicsDistributionComplex.from_distribution_via_integral(
            dist, 3
        )
        phi, theta = meshgrid(
            linspace(0.0, 2.0 * pi, 10), linspace(-pi / 2.0, pi / 2.0, 10)
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
        npt.assert_allclose(shd.mean_direction(), dist.mean_direction(), atol=1e-10)
        npt.assert_allclose(
            shd.mean_direction_numerical(), dist.mean_direction(), atol=1e-10
        )
        npt.assert_allclose(
            shd.integrate_numerically(), dist.integrate_numerically(), atol=1e-10
        )
        npt.assert_allclose(
            shd.pdf(column_stack([x, y, z])),
            dist.pdf(column_stack([x, y, z])),
            atol=0.001,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_from_distribution_via_integral_uniform(self):
        shd = SphericalHarmonicsDistributionComplex.from_distribution_via_integral(
            HypersphericalUniformDistribution(2), degree=0
        )
        npt.assert_allclose(shd.coeff_mat, array([[1 / sqrt(4 * pi)]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_transformation_via_integral_shd(self):
        # Test approximating a spherical harmonic distribution
        dist = SphericalHarmonicsDistributionComplex(
            array([[1, float("NaN"), float("NaN")], [0.0, 1, 0]])
        )

        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(
            dist.pdf, 1
        )
        npt.assert_allclose(shd.coeff_mat, dist.coeff_mat, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported or too slow on this backend",
    )
    def test_convergence(self):
        orders = [1, 2]
        dist = VonMisesFisherDistribution(array([0.0, -1.0, 0.0]), 1.0)
        diffs = zeros(len(orders))

        for order in range(1, len(orders) + 1):
            shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(
                dist.pdf, order
            )
            diffs[order - 1] = shd.hellinger_distance_numerical(dist)
        # Check if the deviation from true density is decreasing
        self.assertTrue(all(diff(diffs) < 0.0))

    @parameterized.expand(
        [
            (
                "zplus",
                [[1 / sqrt(4 * pi), float("NaN"), float("NaN")], [0.0, 1, 0]],
                [0.0, 0.0, 1],
            ),
            (
                "zminus",
                [[1 / sqrt(4 * pi), float("NaN"), float("NaN")], [0.0, -1, 0]],
                [0.0, 0.0, -1],
            ),
            (
                "yplus",
                [
                    [1 / sqrt(4 * pi), float("NaN"), float("NaN")],
                    [1j * sqrt(1 / 2), 0.0, 1j * sqrt(1 / 2)],
                ],
                [0.0, 1, 0],
            ),
            (
                "yminus",
                [
                    [1 / sqrt(4 * pi), float("NaN"), float("NaN")],
                    [-1j * sqrt(1 / 2), 0.0, -1j * sqrt(1 / 2)],
                ],
                [0.0, -1, 0],
            ),
            (
                "xplus",
                [
                    [1 / sqrt(4 * pi), float("NaN"), float("NaN")],
                    [sqrt(1 / 2), 0.0, -sqrt(1 / 2)],
                ],
                [1, 0.0, 0],
            ),
            (
                "xminus",
                [
                    [1 / sqrt(4 * pi), float("NaN"), float("NaN")],
                    [-sqrt(1 / 2), 0.0, sqrt(1 / 2)],
                ],
                [-1, 0.0, 0],
            ),
            (
                "xyplus",
                [
                    [1 / sqrt(4 * pi), float("NaN"), float("NaN")],
                    [
                        1j * sqrt(1 / 2) + sqrt(1 / 2),
                        1,
                        1j * sqrt(1 / 2) - sqrt(1 / 2),
                    ],
                ],
                1 / sqrt(3) * array([1, 1, 1]),
            ),
            (
                "xyminus",
                [
                    [1.0 / sqrt(4 * pi), float("NaN"), float("NaN")],
                    [
                        -1j * sqrt(1 / 2) - sqrt(1.0 / 2.0),
                        0.0,
                        -1j * sqrt(1 / 2) + sqrt(1.0 / 2.0),
                    ],
                ],
                1 / sqrt(2) * array([-1.0, -1.0, 0.0]),
            ),
        ]
    )
    def test_mean(self, _, coeff_mat, expected_output):
        shd = SphericalHarmonicsDistributionComplex(array(coeff_mat))
        npt.assert_allclose(shd.mean_direction(), expected_output, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
