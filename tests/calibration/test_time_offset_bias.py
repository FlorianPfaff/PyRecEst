import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.bias import (
    BiasTrainingExamples,
    fit_sensor_bias_correction,
    fit_sensor_bias_correction_from_examples,
    make_bias_training_examples,
)
from pyrecest.calibration.time_offset import (
    TimeOffsetFitResult,
    aggregate_time_offset_sweeps,
    apply_time_offset,
    fit_time_offset,
    interpolate_reference_values,
    make_offset_grid,
    nearest_time_indices,
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

    def test_nearest_time_indices_accepts_unsorted_reference_times(self):
        indices = nearest_time_indices(
            np.array([10.0, 0.0, 5.0]), np.array([0.2, 7.0, 9.0])
        )

        npt.assert_array_equal(indices, np.array([1, 2, 0]))

    def test_aggregate_time_offset_sweeps_preserves_rmse_and_max_semantics(self):
        aggregated = aggregate_time_offset_sweeps(
            [
                [
                    {
                        "time_offset_s": 0.0,
                        "count": 1.0,
                        "mean": 1.0,
                        "rmse": 3.0,
                        "p95": 3.0,
                        "max": 3.0,
                    }
                ],
                [
                    {
                        "time_offset_s": 0.0,
                        "count": 3.0,
                        "mean": 2.0,
                        "rmse": 4.0,
                        "p95": 4.0,
                        "max": 7.0,
                    }
                ],
            ]
        )

        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0]["count"], 4.0)
        self.assertAlmostEqual(aggregated[0]["mean"], 1.75)
        self.assertAlmostEqual(aggregated[0]["rmse"], np.sqrt(57.0 / 4.0))
        self.assertEqual(aggregated[0]["max"], 7.0)

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

    def test_fit_result_summary_uses_best_nonempty_offset_row(self):
        result = TimeOffsetFitResult(
            best_offset_s=0.5,
            metric="rmse",
            offsets_s=np.array([0.0, 0.5]),
            metric_values=np.array([0.0, 1.25]),
            counts=np.array([0, 7]),
        )

        summary = result.summary()

        self.assertEqual(summary["best_offset_s"], 0.5)
        self.assertEqual(summary["best_metric_value"], 1.25)
        self.assertEqual(summary["best_count"], 7)

    def test_time_offset_summary_reports_empty_when_no_overlap(self):
        summary = time_offset_error_summary(
            np.array([100.0]),
            np.array([[1.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.0], [1.0]]),
            0.0,
        )

        self.assertEqual(summary["count"], 0.0)

    def test_interpolation_rejects_negative_max_time_delta(self):
        with self.assertRaisesRegex(
            ValueError,
            "max_time_delta_s must be nonnegative",
        ):
            interpolate_reference_values(
                np.array([0.0, 1.0]),
                np.array([[0.0], [1.0]]),
                np.array([0.5]),
                max_time_delta_s=-1.0,
            )

    def test_interpolation_rejects_scalar_reference_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "reference_values must be one- or two-dimensional",
        ):
            interpolate_reference_values(
                np.array([0.0, 1.0]),
                np.array(1.0),
                np.array([0.5]),
            )

    def test_interpolation_rejects_higher_rank_reference_values(self):
        with self.assertRaisesRegex(
            ValueError,
            "reference_values must be one- or two-dimensional",
        ):
            interpolate_reference_values(
                np.array([0.0, 1.0]),
                np.zeros((2, 1, 1)),
                np.array([0.5]),
            )

    def test_time_offset_summary_rejects_mismatched_measurement_lengths(self):
        with self.assertRaisesRegex(
            ValueError,
            "measurement_times_s length must match measurement_values rows",
        ):
            time_offset_error_summary(
                np.array([0.0, 1.0]),
                np.array([[0.0]]),
                np.array([0.0, 1.0]),
                np.array([[0.0], [1.0]]),
                0.0,
            )

    def test_time_offset_summary_rejects_higher_rank_measurements(self):
        with self.assertRaisesRegex(
            ValueError,
            "measurement_values must be one- or two-dimensional",
        ):
            time_offset_error_summary(
                np.array([0.0]),
                np.zeros((1, 1, 1)),
                np.array([0.0, 1.0]),
                np.array([[0.0], [1.0]]),
                0.0,
            )


class BiasCalibrationTest(unittest.TestCase):
    def test_make_bias_training_examples_uses_nearest_reference(self):
        examples = make_bias_training_examples(
            np.array([0.0, 1.0]),
            np.array([[1.0], [3.0]]),
            np.array([0.0, 1.0]),
            np.array([[0.0], [2.0]]),
        )

        npt.assert_allclose(examples.residual, np.array([[1.0], [1.0]]))

    def test_make_bias_training_examples_validates_feature_rows_without_references(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "feature_values rows must match measurement_values rows",
        ):
            make_bias_training_examples(
                np.array([0.0, 1.0]),
                np.array([[1.0], [2.0]]),
                np.array([]),
                np.empty((0, 1)),
                feature_values=np.array([[0.0]]),
            )

    def test_fit_sensor_bias_correction_from_examples_rejects_mismatched_rows(self):
        examples = BiasTrainingExamples(
            measured=np.zeros((2, 1)),
            reference=np.zeros((2, 1)),
            residual=np.zeros((2, 1)),
            features=np.zeros((1, 1)),
            time_delta_s=np.zeros(2),
        )

        with self.assertRaisesRegex(
            ValueError,
            "examples.features rows must match examples.residual rows",
        ):
            fit_sensor_bias_correction_from_examples(examples, min_samples=1)

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
