import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.distributions import WatsonDistribution


class WatsonScalarValidationTest(unittest.TestCase):
    def test_constructor_rejects_non_real_kappa_values(self):
        invalid_values = (
            True,
            np.bool_(False),
            "2.0",
            b"2.0",
            2.0 + 0.0j,
            np.timedelta64(2, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
            np.array(
                np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
                dtype=object,
            ),
        )

        for kappa in invalid_values:
            with self.subTest(kappa=kappa):
                with self.assertRaisesRegex(ValueError, "finite real scalar"):
                    WatsonDistribution(array([1.0, 0.0, 0.0]), kappa)

    def test_constructor_normalizes_numeric_scalar_kappa(self):
        for kappa in (2, np.int64(2), np.float64(2.5), np.array(3.5)):
            with self.subTest(kappa=kappa):
                distribution = WatsonDistribution(
                    array([1.0, 0.0, 0.0]),
                    kappa,
                )

                self.assertIsInstance(distribution.kappa, float)
                self.assertEqual(distribution.kappa, float(kappa))


if __name__ == "__main__":
    unittest.main()
