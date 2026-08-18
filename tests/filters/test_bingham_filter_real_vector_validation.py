import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.filters.bingham_filter import BinghamFilter


class TestBinghamFilterRealVectorValidation(unittest.TestCase):
    def setUp(self):
        if pyrecest.backend.__backend_name__ == "jax":
            self.skipTest("BinghamFilter is not supported on the JAX backend")

        self.filter = BinghamFilter()
        self.filter.filter_state = BinghamDistribution(
            array([-5.0, 0.0]), eye(2)
        )
        self.noise = BinghamDistribution(array([-2.0, 0.0]), eye(2))

    def test_update_identity_rejects_boolean_measurement(self):
        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.filter.update_identity(self.noise, [True, False])

        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.filter.update_identity(self.noise, array([True, False]))

    def test_predict_nonlinear_rejects_boolean_system_output(self):
        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.filter.predict_nonlinear(
                lambda _: [True, False],
                self.noise,
            )

    def test_rejects_textual_vectors_before_backend_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.filter.update_identity(self.noise, ["1", "0"])

        with self.assertRaisesRegex(ValueError, "real numeric"):
            self.filter.predict_nonlinear(
                lambda _: ["1", "0"],
                self.noise,
            )


if __name__ == "__main__":
    unittest.main()
