"""Regression tests for discrete arguments in directional distributions."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution, VonMisesFisherDistribution


class TestDirectionalDiscreteArgumentValidation(unittest.TestCase):
    def test_von_mises_moment_order_rejects_non_integer_aliases(self):
        dist = VonMisesDistribution(0.3, 2.0)
        invalid_orders = (
            True,
            np.bool_(True),
            1.0,
            np.float64(1.0),
            array(1.0),
            [1],
        )

        for method in (
            dist.trigonometric_moment,
            dist.trigonometric_moment_analytic,
        ):
            for order in invalid_orders:
                with self.subTest(method=method.__name__, order=repr(order)):
                    with self.assertRaisesRegex(ValueError, "must be an integer"):
                        method(order)

    def test_von_mises_moment_order_accepts_integer_scalars(self):
        dist = VonMisesDistribution(0.3, 2.0)
        expected = dist.trigonometric_moment(1)

        for order in (np.int64(1), array(1)):
            with self.subTest(order=repr(order)):
                npt.assert_allclose(dist.trigonometric_moment(order), expected)
                npt.assert_allclose(dist.trigonometric_moment_analytic(order), expected)

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "numpy",
        reason="vMF random sampling is only supported on NumPy",
    )
    def test_vmf_sample_count_validation_is_independent_of_kappa(self):
        distributions = (
            VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 0.0),
            VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 2.0),
        )
        invalid_counts = (0, -1, True, np.bool_(True), 1.5)

        for dist in distributions:
            for count in invalid_counts:
                with self.subTest(kappa=dist.kappa, count=repr(count)):
                    with self.assertRaisesRegex(ValueError, "n must"):
                        dist.sample(count)

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "numpy",
        reason="vMF random sampling is only supported on NumPy",
    )
    def test_vmf_sample_count_acceptance_is_independent_of_kappa(self):
        distributions = (
            VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 0.0),
            VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 2.0),
        )
        valid_counts = (np.int64(2), 2.0, np.array(2))

        for dist in distributions:
            for count in valid_counts:
                with self.subTest(kappa=dist.kappa, count=repr(count)):
                    self.assertEqual(dist.sample(count).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
