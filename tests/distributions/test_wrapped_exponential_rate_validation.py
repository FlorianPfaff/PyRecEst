import unittest

import numpy as np
import pyrecest.backend

from pyrecest.distributions.circle.wrapped_exponential_distribution import (
    WrappedExponentialDistribution,
)


class WrappedExponentialRateValidationTest(unittest.TestCase):
    def test_rejects_non_real_rate_scalars_before_backend_conversion(self):
        invalid_rates = (
            True,
            np.bool_(True),
            np.array(True, dtype=object),
            "1.0",
            np.str_("1.0"),
            np.array("1.0", dtype=object),
            1.0 + 0.0j,
            np.complex128(1.0 + 0.0j),
            np.array(1.0 + 0.0j),
            np.datetime64("2026-01-01"),
            np.timedelta64(1, "s"),
        )

        for rate in invalid_rates:
            with self.subTest(rate=rate):
                with self.assertRaisesRegex(ValueError, "positive real scalar"):
                    WrappedExponentialDistribution(rate)

    def test_accepts_real_numpy_scalar(self):
        distribution = WrappedExponentialDistribution(np.float64(2.0))

        self.assertAlmostEqual(float(distribution.lambda_), 2.0)

    def test_normalizes_one_element_rate_array_to_scalar(self):
        distribution = WrappedExponentialDistribution(np.array([2.0]))

        for value in (
            distribution.lambda_,
            distribution.pdf(1.0),
            distribution.trigonometric_moment(1),
            distribution.entropy(),
        ):
            with self.subTest(value=value):
                converted = np.asarray(pyrecest.backend.to_numpy(value))
                self.assertEqual(converted.shape, ())


if __name__ == "__main__":
    unittest.main()
