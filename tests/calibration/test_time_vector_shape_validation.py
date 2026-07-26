import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.bias import make_bias_training_examples
from pyrecest.calibration.time_offset import (
    apply_time_offset,
    interpolate_reference_values,
    nearest_time_indices,
)


class CalibrationTimeVectorShapeValidationTest(unittest.TestCase):
    def test_time_offset_helpers_reject_matrix_shaped_time_vectors(self):
        row_matrix = np.array([[0.0, 1.0]])
        column_matrix = np.array([[0.0], [1.0]])

        for invalid_times in (row_matrix, column_matrix):
            with self.subTest(helper="apply_time_offset", shape=invalid_times.shape):
                with self.assertRaisesRegex(
                    ValueError, "times_s must be one-dimensional"
                ):
                    apply_time_offset(invalid_times, 0.25)

            with self.subTest(helper="nearest_reference", shape=invalid_times.shape):
                with self.assertRaisesRegex(
                    ValueError, "reference_times_s must be one-dimensional"
                ):
                    nearest_time_indices(invalid_times, np.array([0.25]))

            with self.subTest(helper="nearest_query", shape=invalid_times.shape):
                with self.assertRaisesRegex(
                    ValueError, "query_times_s must be one-dimensional"
                ):
                    nearest_time_indices(np.array([0.0, 1.0]), invalid_times)

            with self.subTest(helper="interpolate_reference", shape=invalid_times.shape):
                with self.assertRaisesRegex(
                    ValueError, "reference_times_s must be one-dimensional"
                ):
                    interpolate_reference_values(
                        invalid_times,
                        np.array([[0.0], [1.0]]),
                        np.array([0.25]),
                    )

            with self.subTest(helper="interpolate_query", shape=invalid_times.shape):
                with self.assertRaisesRegex(
                    ValueError, "query_times_s must be one-dimensional"
                ):
                    interpolate_reference_values(
                        np.array([0.0, 1.0]),
                        np.array([[0.0], [1.0]]),
                        invalid_times,
                    )

    def test_bias_examples_reject_matrix_shaped_time_vectors(self):
        valid_kwargs = {
            "measurement_times_s": np.array([0.0, 1.0]),
            "measurement_values": np.array([[1.0], [2.0]]),
            "reference_times_s": np.array([0.0, 1.0]),
            "reference_values": np.array([[0.0], [1.0]]),
            "max_time_delta_s": 0.0,
        }

        for field in ("measurement_times_s", "reference_times_s"):
            for invalid_times in (
                np.array([[0.0, 1.0]]),
                np.array([[0.0], [1.0]]),
            ):
                kwargs = dict(valid_kwargs)
                kwargs[field] = invalid_times
                with self.subTest(field=field, shape=invalid_times.shape):
                    with self.assertRaisesRegex(
                        ValueError, f"{field} must be one-dimensional"
                    ):
                        make_bias_training_examples(**kwargs)

    def test_scalar_and_vector_time_inputs_remain_supported(self):
        npt.assert_allclose(apply_time_offset(1.0, 0.25), np.array([1.25]))

        examples = make_bias_training_examples(
            measurement_times_s=0.0,
            measurement_values=np.array([[2.0]]),
            reference_times_s=0.0,
            reference_values=np.array([[1.0]]),
            max_time_delta_s=0.0,
        )

        npt.assert_allclose(examples.residual, np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main()
