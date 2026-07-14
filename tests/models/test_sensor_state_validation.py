"""Regression tests for sensor-model state rank validation."""

import unittest

import numpy as np
from pyrecest.models import range_bearing_measurement
from pyrecest.models.sensor_models import (
    range_bearing_measurement as module_range_bearing_measurement,
)


class TestSensorStateValidation(unittest.TestCase):
    def test_rejects_nonvector_states_with_clear_value_error(self):
        measurement_functions = (
            range_bearing_measurement,
            module_range_bearing_measurement,
        )
        invalid_states = (
            np.asarray(1.0),
            np.asarray([[3.0, 4.0, 0.0, 0.0]]),
            np.asarray([[3.0], [4.0], [0.0], [0.0]]),
        )

        for measurement_function in measurement_functions:
            for state in invalid_states:
                with self.subTest(
                    function=measurement_function.__module__,
                    shape=state.shape,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "state must be a one-dimensional array",
                    ):
                        measurement_function(state)


if __name__ == "__main__":
    unittest.main()
