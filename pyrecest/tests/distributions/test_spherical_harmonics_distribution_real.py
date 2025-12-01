import unittest
import warnings

import numpy.testing as npt
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    array,
    column_stack,
    diff,
    ones_like,
    pi,
    random,
    sqrt,
    zeros,
)
from pyrecest.distributions.hypersphere_subset.abstract_spherical_distribution import (
    AbstractSphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_real import (
    SphericalHarmonicsDistributionReal,
)


class SphericalHarmonicsDistributionRealTest(unittest.TestCase):
    def testNormalizationError(self):
        self.assertRaises(ValueError, SphericalHarmonicsDistributionReal, array(0.0))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def testNormalizationWarning(self):
        with warnings.catch_warnings(record=True) as w:
            SphericalHarmonicsDistributionReal(random.uniform(size=(3, 5)))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def testNormalization(self):
        unnormalized_coeffs = random.uniform(size=(3, 5))
        shd = SphericalHarmonicsDistributionReal(unnormalized_coeffs)
        self.assertAlmostEqual(shd.integrate(), 1.0, delta=1e-6)
        x, y, z = SphericalHarmonicsDistributionRealTest._gen_naive_grid(10)

        vals_normalized = shd.pdf(column_stack((x, y, z)))
        shd.coeff_mat = unnormalized_coeffs
        vals_unnormalized = shd.pdf(column_stack((x, y, z)))
        self.assertTrue(
            allclose(
                diff(vals_normalized / vals_unnormalized),
                zeros(x.shape[0] - 1),
                atol=1e-6,
            )
        )

    @parameterized.expand(
        [  # jscpd:ignore-start-python
            (
                "l0m0",
                [
                    [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 0, float("NaN"), float("NaN")],
                    [0, 0, 0, 0, 0],
                ],
                lambda x, _, __: ones_like(x) * sqrt(1 / (4 * pi)),
            ),
            (
                "l1mneg1",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [1, 0, 0, float("NaN"), float("NaN")],
                    [0, 0, 0, 0, 0],
                ],
                lambda _, y, __: sqrt(3 / (4 * pi)) * y,
            ),
            (
                "l1_m0",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 1, 0, float("NaN"), float("NaN")],
                    [0, 0, 0, 0, 0],
                ],
                lambda _, __, z: sqrt(3 / (4 * pi)) * z,
            ),
            (
                "l1_m1",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 1, float("NaN"), float("NaN")],
                    [0, 0, 0, 0, 0],
                ],
                lambda x, _, __: sqrt(3 / (4 * pi)) * x,
            ),
            (
                "l2_mneg2",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 0, float("NaN"), float("NaN")],
                    [1, 0, 0, 0, 0],
                ],
                lambda x, y, __: 1 / 2 * sqrt(15 / pi) * x * y,
            ),
            (
                "l2_mneg1",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 0, float("NaN"), float("NaN")],
                    [0, 1, 0, 0, 0],
                ],
                lambda _, y, z: 1 / 2 * sqrt(15 / pi) * y * z,
            ),
            (
                "l2_m0",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 0, float("NaN"), float("NaN")],
                    [0, 0, 1, 0, 0],
                ],
                lambda x, y, z: 1 / 4 * sqrt(5 / pi) * (2 * z**2 - x**2 - y**2),
            ),
            (
                "l2_m1",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 0, float("NaN"), float("NaN")],
                    [0, 0, 0, 1, 0],
                ],
                lambda x, _, z: 1 / 2 * sqrt(15 / pi) * x * z,
            ),
            (
                "l2_m2",
                [
                    [0, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                    [0, 0, 0, float("NaN"), float("NaN")],
                    [0, 0, 0, 0, 1],
                ],
                lambda x, y, _: 1 / 4 * sqrt(15 / pi) * (x**2 - y**2),
            ),
        ]  # jscpd:ignore-end
    )
    def test_basis_function(self, name, coeff_mat, result_func):
        random.seed(10)
        shd = SphericalHarmonicsDistributionReal(1 / sqrt(4 * pi))
        shd.coeff_mat = array(coeff_mat)
        x, y, z = SphericalHarmonicsDistributionRealTest._gen_naive_grid(10)
        npt.assert_allclose(
            shd.pdf(column_stack((x, y, z))),
            result_func(x, y, z),
            rtol=0.002,
            atol=1e-5,
            err_msg=name,
        )

    @staticmethod
    def _gen_naive_grid(n_per_dim):
        phi = random.uniform(size=n_per_dim) * 2.0 * pi
        theta = random.uniform(size=n_per_dim) * pi - pi / 2.0
        return AbstractSphericalDistribution.sph_to_cart(phi, theta)

    @parameterized.expand(
        [  # jscpd:ignore-start-python
            (
                "l0_m0",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l1_mneg1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [1, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l1_m0",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 1, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l1_m1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 1, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l2_mneg2",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 0, float("NaN"), float("NaN")],
                        [1, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l2_mneg1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 0, float("NaN"), float("NaN")],
                        [0, 1, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l2_m0",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 1, 0, 0],
                    ]
                ),
            ),
            (
                "l2_m1",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 1, 0],
                    ]
                ),
            ),
            (
                "l2_m2",
                array(
                    [
                        [1, float("NaN"), float("NaN"), float("NaN"), float("NaN")],
                        [0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 1],
                    ]
                ),
            ),
            (
                "l3_mneg3",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [1, 0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l3_mneg2",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [0, 1, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l3_mneg1",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 1, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l3_m0",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 1, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l3_m1",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 1, 0, 0],
                    ]
                ),
            ),
            (
                "l3_m2",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 0, 1, 0],
                    ]
                ),
            ),
            (
                "l3_m3",
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
                            0,
                            0,
                            0,
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                            float("NaN"),
                        ],
                        [0, 0, 0, 0, 0, float("NaN"), float("NaN")],
                        [0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            ),
            ("random", random.uniform(size=(4, 7))),
        ]  # jscpd:ignore-end
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_conversion(self, _, coeff_mat):
        rshd = SphericalHarmonicsDistributionReal(coeff_mat)
        cshd = rshd.to_spherical_harmonics_distribution_complex()
        phi_to_test, theta_to_test = (
            random.uniform(size=10) * 2 * pi,
            random.uniform(size=10) * pi - pi / 2,
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi_to_test, theta_to_test)
        npt.assert_allclose(
            cshd.pdf(column_stack((x, y, z))),
            rshd.pdf(column_stack((x, y, z))),
            atol=1e-6,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_conversion_to_complex_and_back(self):
        # Suppress warnings related to normalization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rshd = SphericalHarmonicsDistributionReal(random.uniform(size=(4, 7)))

        cshd = rshd.to_spherical_harmonics_distribution_complex()
        rshd2 = cshd.to_spherical_harmonics_distribution_real()
        npt.assert_allclose(rshd2.coeff_mat, rshd.coeff_mat, atol=1e-6, equal_nan=True)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Test not supported for this backend",
    )
    def test_integral_analytical(self):
        # Suppress warnings related to normalization
        unnormalized_coeffs = random.uniform(size=(3, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shd = SphericalHarmonicsDistributionReal(unnormalized_coeffs)

        npt.assert_allclose(shd.integrate_numerically(), shd.integrate(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
