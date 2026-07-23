import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.distributions.cart_prod.mardia_sutton_distribution import (
    MardiaSuttonDistribution,
)


class TestMardiaSuttonParameterValidation(unittest.TestCase):
    @staticmethod
    def _create(**overrides):
        parameters = {
            "mu": 2.0,
            "mu0": 1.0,
            "kappa": 0.7,
            "rho1": 0.5,
            "rho2": 0.3,
            "sigma": 1.5,
        }
        parameters.update(overrides)
        return MardiaSuttonDistribution(**parameters)

    def test_rejects_malformed_location_parameters(self):
        invalid_values = (True, np.nan, np.inf, -np.inf, [0.0], 1.0 + 0.0j)
        for name in ("mu", "mu0"):
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self._create(**{name: value})

    def test_rejects_malformed_correlation_parameters(self):
        invalid_values = (True, np.nan, np.inf, -np.inf, [0.2], 0.2 + 0.0j)
        for name in ("rho1", "rho2"):
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self._create(**{name: value})

    def test_accepts_zero_dimensional_numpy_scalars(self):
        dist = self._create(
            mu=np.array(2.0),
            mu0=np.array(1.0),
            rho1=np.array(0.5),
            rho2=np.array(0.3),
        )

        npt.assert_allclose(dist.mode(), np.array([1.0, 2.0]))
        self.assertIsInstance(dist.mu, float)
        self.assertIsInstance(dist.rho1, float)
        self.assertIsInstance(dist.rho2, float)


if __name__ == "__main__":
    unittest.main()
