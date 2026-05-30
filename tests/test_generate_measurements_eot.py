# pylint: disable=duplicate-code
import unittest
from unittest.mock import patch

import numpy as np
from pyrecest.backend import all as backend_all
from pyrecest.backend import array
from pyrecest.evaluation import generate_measurements
from pyrecest.evaluation.eot_shape_database import PolygonWithSampling
from shapely.geometry import Polygon


class TestGenerateMeasurementsEot(unittest.TestCase):
    """Regression coverage for EOT measurement placement over time."""

    def assert_all_between(self, values, lower, upper):
        self.assertTrue(bool(backend_all(lower < values)))
        self.assertTrue(bool(backend_all(values < upper)))

    def test_eot_measurements_use_current_timestep_pose(self):
        simulation_param = {
            "eot": True,
            "target_shape": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "eot_sampling_style": "within",
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": [5, 5],
        }
        groundtruth = array(
            [
                [0.0, 0.0],
                [10.0, 20.0],
            ]
        )

        measurements = generate_measurements(groundtruth, simulation_param)

        first_timestep = measurements[0]
        second_timestep = measurements[1]
        self.assert_all_between(first_timestep[:, 0], -1e-12, 1.0 + 1e-12)
        self.assert_all_between(first_timestep[:, 1], -1e-12, 1.0 + 1e-12)
        self.assert_all_between(second_timestep[:, 0], 10.0 - 1e-12, 11.0 + 1e-12)
        self.assert_all_between(second_timestep[:, 1], 20.0 - 1e-12, 21.0 + 1e-12)

    def test_eot_se2_rotation_interprets_heading_as_radians(self):
        captured_exterior = {}

        def capture_shape_exterior(shape, _num_points):
            captured_exterior["coords"] = np.asarray(shape.exterior.coords[:-1])
            return array([[0.0, 0.0]])

        simulation_param = {
            "eot": True,
            "target_shape": Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
            "eot_sampling_style": "boundary",
            "n_timesteps": 1,
            "n_meas_at_individual_time_step": [1],
        }
        groundtruth = array([[np.pi / 2.0, 0.0, 0.0]])

        with patch.object(
            PolygonWithSampling, "sample_on_boundary", capture_shape_exterior
        ):
            generate_measurements(groundtruth, simulation_param)

        np.testing.assert_allclose(
            captured_exterior["coords"],
            np.asarray(
                [
                    [1.5, -0.5],
                    [1.5, 1.5],
                    [0.5, 1.5],
                    [0.5, -0.5],
                ]
            ),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
