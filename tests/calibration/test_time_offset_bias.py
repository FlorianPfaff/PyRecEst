import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.bias import (
    fit_sensor_bias_correction,
    make_bias_training_examples,
)
from pyrecest.calibration.time_offset import (
    apply_time_offset,
    fit_time_offset,
    make_offset_grid,
    time_offset_error_summary,
)


class TimeOffsetCalibrationTest(unittest.TestCase):
    def test_offset_grid_is_inclusive(self):
        grid = make_offset_grid(-0.5, 0.5, 0.25)

        npt.assert_allclose(grid, np.array([-0.5, -0.25, 0.0, 0.25, 0.5]))

    def test_apply_time_offset_shifts_times(self):
        npt.assert_allclose(
            apply_time_offset(np.array([1.0, 2.0]), 0.5), np.array([1.5, 2.5])
        )

    def test_fit_time_offset_recovers_known_shift(self):
        reference_times = np.linspace(0.0, 10.0, 101)
        reference_values = np.column_stack([reference_times, reference_times**2])
        measurement_times = np.linspace(1.0, 8.0, 15)
        true_offset = 0.4
        measurement_values = np.column_stack(
            [
                measurement_times + true_offset,
                (measurement_times + true_offset) ** 2,
            ]
        )
        offsets = make_offset_grid(-1.0, 1.0, 0.1)

        result = fit_time_offset(
            measurement_times,
            measurement_values,
            reference_times,
            reference_values,
            offsets,
            max_time_delta_s=0.25,
        )

        self.assertAlmostEqual(result.best_offset_s, true_offset, places=9)
        self.assertEqual(result.summary()["best_count"], len(measurement_times))

    def test_time_offset_summary_reports_empty_when_no_overlap(self):
        summary = time_offset_error_summary(
            np.array([100.0]),
            np.array([[1.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.0], [1.0]]),
            0.0,
        )

        self.assertEqual(summary["count"], 0.0)


class BiasCalibrationTest(unittest.TestCase):
    def test_make_bias_training_examples_uses_nearest_reference(self):
        examples = make_bias_training_examples(
            np.array([0.0, 1.0]),
            np.array([[1.0], [3.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.0], [2.0]]),
        )

        npt.assert_allclose(examples.residual, np.array([[1.0], [1.0]]))

    def test_fit_sensor_bias_correction_subtracts_predicted_bias(self):
        times = np.arange(8.0)
        reference = np.column_stack([times, -times])
        feature = times.reshape(-1, 1)
        bias = np.column_stack([2.0 + 0.5 * times, -1.0 + 0.25 * times])
        measurements = reference + bias

        model = fit_sensor_bias_correction(
            times,
            measurements,
            times,
            reference,
            feature_values=feature,
            ridge_alpha=0.0,
            min_samples=2,
        )
        corrected = model.apply(measurements, feature)

        npt.assert_allclose(corrected, reference, atol=1e-10)
        self.assertEqual(model.training_count, len(times))

    def test_constant_bias_model_when_features_are_unavailable(self):
        times = np.arange(5.0)
        reference = times.reshape(-1, 1)
        measurements = reference + 3.0

        model = fit_sensor_bias_correction(
            times, measurements, times, reference, min_samples=2
        )
        corrected = model.apply(measurements)

        self.assertEqual(model.feature_dim, 0)
        npt.assert_allclose(corrected, reference, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
