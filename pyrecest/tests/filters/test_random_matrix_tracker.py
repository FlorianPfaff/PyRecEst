import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, concatenate, diag, eye, linalg, mean, zeros
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.kalman_filter import KalmanFilter
from pyrecest.filters.random_matrix_tracker import RandomMatrixTracker


class TestRandomMatrixTracker(unittest.TestCase):
    def setUp(self):
        self.initial_state = array([1.0, 2.0])
        self.initial_covariance = array([[0.1, 0.0], [0.0, 0.1]])
        self.initial_extent = array([[1.0, 0.1], [0.1, 1.0]])
        self.measurement_noise = array([[0.2, 0.0], [0.0, 0.2]])

        self.tracker = RandomMatrixTracker(
            self.initial_state, self.initial_covariance, self.initial_extent
        )

    def test_initialization(self):
        npt.assert_array_equal(self.tracker.kinematic_state, self.initial_state)
        npt.assert_array_equal(self.tracker.covariance, self.initial_covariance)
        npt.assert_array_equal(self.tracker.extent, self.initial_extent)

    def test_get_point_estimate(self):
        expected = concatenate(
            [self.initial_state, array(self.initial_extent).flatten()]
        )
        npt.assert_array_equal(self.tracker.get_point_estimate(), expected)

    def test_get_point_estimate_kinematics(self):
        npt.assert_array_equal(
            self.tracker.get_point_estimate_kinematics(), self.initial_state
        )

    def test_get_point_estimate_extent(self):
        npt.assert_array_equal(
            self.tracker.get_point_estimate_extent(), self.initial_extent
        )

    def test_predict(self):
        dt = 0.1
        Cw = array([[0.05, 0.0], [0.0, 0.05]])
        tau = 1.0

        system_matrix = eye(2)  # 2-D random walk

        # Call the predict method
        self.tracker.predict(dt, Cw, system_matrix, tau)

        # Check if state and state covariance are updated correctly
        expected_state = array([1.0, 2.0])
        expected_covariance = self.initial_covariance + Cw
        expected_extent = self.initial_extent

        npt.assert_array_almost_equal(
            self.tracker.kinematic_state, expected_state, decimal=5
        )
        npt.assert_array_almost_equal(
            self.tracker.covariance, expected_covariance, decimal=5
        )
        npt.assert_array_almost_equal(self.tracker.extent, expected_extent, decimal=5)

    @parameterized.expand(
        [
            (
                "smaller",
                array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]]),
                "The extent should now be smaller since the measurements are closely spaced",
            ),
            (
                "larger",
                array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
                "The extent should now be larger since the measurements are spaced more widely",
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",
    )
    def test_update(self, name, offset, _):
        ys = array([self.initial_state + offset_row for offset_row in offset]).T
        Cv = array([[0.1, 0.0], [0.0, 0.1]])
        H = eye(self.initial_state.shape[0])

        # Call the update method
        self.tracker.update(ys, H, Cv)

        # Check if state, state covariance, and extent are updated correctly. Use KF for comparison
        kf = KalmanFilter(
            GaussianDistribution(self.initial_state, self.initial_covariance)
        )
        kf.update_linear(mean(ys, axis=1), H, (self.initial_extent + Cv) / ys.shape[1])

        npt.assert_array_almost_equal(
            self.tracker.kinematic_state, kf.get_point_estimate(), decimal=5
        )
        npt.assert_array_almost_equal(
            self.tracker.covariance, kf.filter_state.C, decimal=5
        )

        # Check if extent has changed as expected
        if name == "smaller":
            npt.assert_array_less(
                zeros(2), linalg.eig(self.initial_extent - self.tracker.extent)[0]
            )
        elif name == "larger":
            npt.assert_array_less(
                zeros(2), linalg.eig(self.tracker.extent - self.initial_extent)[0]
            )
        else:
            raise ValueError(f"Invalid test name: {name}")

    @patch("matplotlib.pyplot.show")
    def test_draw_extent_3d(self, mock_show):
        self.tracker = RandomMatrixTracker(
            zeros(3),
            eye(3),
            diag(array([1.0, 2.0, 3.0])),
            kinematic_state_to_pos_matrix=eye(3),
        )
        self.tracker.plot_point_estimate()

        # Check that the plot was created
        self.assertIsInstance(plt.gcf(), plt.Figure)

        # Check that the plot is shown
        mock_show.assert_called_once()

    def test_get_contour_points(self):
        # Create a kinematic state (e.g., position and velocity in 2D)
        kinematic_state = array([10.0, 5.0])
        # Create a covariance matrix (not directly used in this test)
        covariance = eye(2)
        # Create an extent matrix that defines the size and orientation of the ellipse
        extent = array([[3.0, 0.0], [0.0, 1.0]])

        # Initialize the RandomMatrixTracker
        tracker = RandomMatrixTracker(kinematic_state, covariance, extent)

        # Number of points to generate
        n_points = 100

        # Get the contour points
        contour_points = tracker.get_contour_points(n_points)

        # Check the number of points returned
        self.assertEqual(contour_points.shape[0], n_points)  # Expecting n_points
        self.assertEqual(contour_points.shape[1], 2)  # Expecting x and y coordinates


if __name__ == "__main__":
    unittest.main()
