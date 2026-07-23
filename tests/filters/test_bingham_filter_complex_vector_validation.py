import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, complex128, cos, sin
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.filters.bingham_filter import BinghamFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="BinghamFilter is not supported on JAX",
)
class TestBinghamFilterComplexVectorValidation(unittest.TestCase):
    def setUp(self):
        phi = 0.4
        self.state = BinghamDistribution(
            array([-5.0, 0.0]),
            array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]),
        )
        self.noise = BinghamDistribution(
            array([-3.0, 0.0]),
            array([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.filter = BinghamFilter()
        self.filter.filter_state = self.state

    def test_update_rejects_complex_measurement(self):
        measurement = array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex128)

        with self.assertRaisesRegex(ValueError, "measurement z must be real-valued"):
            self.filter.update_identity(self.noise, measurement)

    def test_prediction_rejects_complex_system_output(self):
        def system_function(sample):
            return array(
                [sample[0] + 0.0j, sample[1] + 0.0j],
                dtype=complex128,
            )

        with self.assertRaisesRegex(
            ValueError, "system function output must be real-valued"
        ):
            self.filter.predict_nonlinear(system_function, self.noise)


if __name__ == "__main__":
    unittest.main()
