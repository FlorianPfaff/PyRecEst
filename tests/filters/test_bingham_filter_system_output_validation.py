import unittest

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, cos, sin
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.filters.bingham_filter import BinghamFilter


class TestBinghamFilterSystemOutputValidation(unittest.TestCase):
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

    def _predict_with_first_output(self, first_output):
        calls = 0

        def system_function(sample):
            nonlocal calls
            calls += 1
            if calls == 1:
                return first_output
            return sample

        self.filter.predict_nonlinear(system_function, self.noise)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_scalar_output_before_column_broadcast(self):
        with self.assertRaisesRegex(
            ValueError, "system function output must have shape"
        ):
            self._predict_with_first_output(2**-0.5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_nonfinite_output(self):
        with self.assertRaisesRegex(
            ValueError, "system function output must be finite"
        ):
            self._predict_with_first_output(array([float("nan"), 0.0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_nonunit_output(self):
        with self.assertRaisesRegex(
            ValueError, "system function output must be a unit vector"
        ):
            self._predict_with_first_output(array([2.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
