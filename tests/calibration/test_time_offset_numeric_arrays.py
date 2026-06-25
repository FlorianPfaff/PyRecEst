import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.calibration.time_offset import (
    apply_time_offset,
    interpolate_reference_values,
    nearest_time_indices,
    time_offset_error_summary,
)


class TimeOffsetNumericArrayValidationTest(unittest.TestCase):
    def assert_rejects_numeric_array(self, func, expected_name, *args):
        with self.assertRaisesRegex(
            ValueError,
            f"{expected_name} must contain real numeric values",
        ):
            func(*args)

    def test_apply_time_offset_rejects_bool_and_text_time_arrays(self):
        invalid_times = (
            np.array([False, True]),
            np.array(["0.0", "1.0"]),
            np.array([0.0, "1.0"], dtype=object),
            np.array([0.0, b"1.0"], dtype=object),
        )
        for times_s in invalid_times:
            with self.subTest(dtype=times_s.dtype, values=times_s):
                self.assert_rejects_numeric_array(
                    apply_time_offset,
                    "times_s",
                    times_s,
                    0.0,
                )

    def test_nearest_time_indices_rejects_bool_and_text_time_arrays(self):
        cases = (
            (
                np.array([False, True]),
                np.array([0.25]),
                "reference_times_s",
            ),
            (
                np.array(["0.0", "1.0"]),
                np.array([0.25]),
                "reference_times_s",
            ),
            (
                np.array([0.0, 1.0]),
                np.array([False]),
                "query_times_s",
            ),
            (
                np.array([0.0, 1.0]),
                np.array(["0.25"]),
                "query_times_s",
            ),
        )
        for reference_times_s, query_times_s, expected_name in cases:
            with self.subTest(expected_name=expected_name):
                self.assert_rejects_numeric_array(
                    nearest_time_indices,
                    expected_name,
                    reference_times_s,
                    query_times_s,
                )

    def test_interpolation_rejects_bool_and_text_data_arrays(self):
        numeric_times = np.array([0.0, 1.0])
        numeric_values = np.array([[0.0], [1.0]])
        numeric_query = np.array([0.25])
        cases = (
            (
                np.array([False, True]),
                numeric_values,
                numeric_query,
                "reference_times_s",
            ),
            (
                np.array(["0.0", "1.0"]),
                numeric_values,
                numeric_query,
                "reference_times_s",
            ),
            (
                numeric_times,
                np.array([[False], [True]]),
                numeric_query,
                "reference_values",
            ),
            (
                numeric_times,
                np.array([["0.0"], ["1.0"]]),
                numeric_query,
                "reference_values",
            ),
            (
                numeric_times,
                numeric_values,
                np.array([False]),
                "query_times_s",
            ),
            (
                numeric_times,
                numeric_values,
                np.array(["0.25"]),
                "query_times_s",
            ),
        )
        for reference_times_s, reference_values, query_times_s, expected_name in cases:
            with self.subTest(expected_name=expected_name):
                self.assert_rejects_numeric_array(
                    interpolate_reference_values,
                    expected_name,
                    reference_times_s,
                    reference_values,
                    query_times_s,
                )

    def test_time_offset_summary_rejects_bool_and_text_measurement_values(self):
        reference_times = np.array([0.0, 1.0])
        reference_values = np.array([[0.0], [1.0]])
        for measurement_values in (np.array([True]), np.array(["0.0"])):
            with self.subTest(dtype=measurement_values.dtype):
                self.assert_rejects_numeric_array(
                    time_offset_error_summary,
                    "measurement_values",
                    np.array([0.0]),
                    measurement_values,
                    reference_times,
                    reference_values,
                    0.0,
                )

    def test_interpolation_accepts_numeric_object_arrays(self):
        interpolated, valid = interpolate_reference_values(
            np.array([0, 1], dtype=object),
            np.array([[0.0], [1.0]], dtype=object),
            np.array([0.25], dtype=object),
        )

        npt.assert_allclose(interpolated, np.array([[0.25]]))
        npt.assert_array_equal(valid, np.array([True]))


if __name__ == "__main__":
    unittest.main()
