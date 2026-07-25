import unittest

import numpy as np
from pyrecest.experimental.dvs.event_likelihood import (
    ContourSample,
    EventLikelihoodConfig,
    PointProcessUpdateConfig,
    expected_event_count,
    normal_flow_activities,
)


def _temporal_scalars():
    return (
        np.timedelta64(1, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000001"),
        np.array(np.timedelta64(1, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
    )


def _contour():
    return ContourSample(
        points=np.array([[0.0, 0.0], [1.0, 0.0]]),
        normals=np.array([[1.0, 0.0], [0.0, 1.0]]),
        weights=np.array([0.5, 0.5]),
    )


class TestDVSEventLikelihoodTemporalValidation(unittest.TestCase):
    def test_event_likelihood_config_rejects_temporal_numeric_scalars(self):
        for field in (
            "spatial_sigma_px",
            "foreground_rate",
            "background_rate",
            "activity_floor",
            "min_intensity",
            "batch_duration",
        ):
            for value in _temporal_scalars():
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        EventLikelihoodConfig(**{field: value})

    def test_point_process_update_config_rejects_temporal_count_scalars(self):
        cases = (
            ("contour_samples", np.timedelta64(3, "ns")),
            (
                "contour_samples",
                np.datetime64("1970-01-01T00:00:00.000000003"),
            ),
            ("max_map_iterations", np.timedelta64(0, "ns")),
            (
                "shape_update_modes",
                np.datetime64("1970-01-01T00:00:00.000000000"),
            ),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    PointProcessUpdateConfig(**{field: value})

    def test_point_process_update_config_rejects_temporal_float_scalars(self):
        cases = (
            ("finite_difference_eps", np.timedelta64(1, "ns")),
            (
                "map_step_size",
                np.datetime64("1970-01-01T00:00:00.000000000"),
            ),
            ("covariance_damping", np.timedelta64(1, "ns")),
            ("max_state_update_norm", np.timedelta64(1, "ns")),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    PointProcessUpdateConfig(**{field: value})

    def test_likelihood_helpers_reject_temporal_runtime_scalars(self):
        contour = _contour()
        velocity = np.array([1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "activity_floor"):
            normal_flow_activities(
                contour.normals,
                velocity,
                activity_floor=np.timedelta64(1, "ns"),
            )

        with self.assertRaisesRegex(ValueError, "batch_duration"):
            expected_event_count(
                contour,
                velocity,
                batch_duration=np.timedelta64(1, "ns"),
            )

        with self.assertRaisesRegex(ValueError, "image_area"):
            expected_event_count(
                contour,
                velocity,
                image_area=np.datetime64("1970-01-01T00:00:00.000000001"),
            )


if __name__ == "__main__":
    unittest.main()
