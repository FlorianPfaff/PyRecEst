import unittest
from math import sqrt

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions import SO3ProductDiracDistribution


class SO3ProductDiracComponentStatisticsIndexValidationTest(unittest.TestCase):
    def setUp(self):
        identity = [0.0, 0.0, 0.0, 1.0]
        x_ninety = [sqrt(0.5), 0.0, 0.0, sqrt(0.5)]
        z_ninety = [0.0, 0.0, sqrt(0.5), sqrt(0.5)]
        self.dist = SO3ProductDiracDistribution(
            array(
                [
                    [identity, x_ninety],
                    [z_ninety, identity],
                ]
            ),
            array([0.25, 0.75]),
        )

    def test_component_statistics_accept_numpy_integer_scalar(self):
        component_index = np.int64(1)

        npt.assert_allclose(
            self.dist.component_particles(component_index), self.dist.d[:, 1, :]
        )
        npt.assert_allclose(self.dist.moment(component_index), self.dist.moment()[1])
        npt.assert_allclose(
            self.dist.mean_quaternion(component_index), self.dist.mean_quaternion()[1]
        )

    def test_component_statistics_reject_invalid_indices(self):
        invalid_indices = (True, np.bool_(True), 1.0, -1, 2, np.array([0]))
        methods = (
            self.dist.component_particles,
            self.dist.moment,
            self.dist.mean_quaternion,
        )

        for method in methods:
            for invalid_index in invalid_indices:
                with self.subTest(method=method.__name__, invalid_index=invalid_index):
                    with self.assertRaisesRegex(ValueError, "component_index"):
                        method(invalid_index)


if __name__ == "__main__":
    unittest.main()
