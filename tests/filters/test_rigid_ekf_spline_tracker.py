import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, diag, eye, linalg
from pyrecest.filters import RigidEKFSplineTracker, RigidEkfSplineTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Finite-difference spline tracker tests are numpy-backend checks.",
)
class TestRigidEKFSplineTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = RigidEKFSplineTracker(
            kinematic_state=array([0.0, 0.0, 0.0, 0.0, 0.0]),
            covariance=diag(array([0.1, 0.1, 0.01, 0.01, 0.01])),
            measurement_noise=0.01 * eye(2),
        )

    def test_initialization_and_alias(self):
        self.assertIs(RigidEkfSplineTracker, RigidEKFSplineTracker)
        self.assertEqual(self.tracker.state.shape, (5,))
        self.assertEqual(self.tracker.get_scaled_control_points().shape, (8, 2))
        self.assertEqual(
            self.tracker.get_point_estimate_extent(flatten_matrix=True).shape,
            (16,),
        )
        npt.assert_allclose(self.tracker.scale_state, array([1.0, 1.0]), atol=1e-12)

    def test_contour_and_bounding_box_shapes(self):
        contour_points = self.tracker.get_contour_points(16)
        box = self.tracker.get_bounding_box(32)
        dimensions = np.asarray(box["dimension"])

        self.assertTupleEqual(tuple(contour_points.shape), (16, 2))
        self.assertTupleEqual(tuple(box["center_xy"].shape), (2,))
        self.assertTupleEqual(tuple(dimensions.shape), (2,))
        self.assertTrue(np.all(dimensions > 0.0))

    def test_predict_uses_constant_turn_kinematics(self):
        starting_state = array([0.0, 0.0, 0.0, 2.0, 0.2])
        covariance = diag(array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4]))
        tracker = RigidEKFSplineTracker(
            kinematic_state=starting_state,
            covariance=covariance,
            measurement_noise=0.01 * eye(2),
        )

        tracker.predict(dt=0.5)

        predicted_pose = tracker.get_point_estimate_kinematics()[:3]
        npt.assert_allclose(
            predicted_pose,
            array([1.0, 0.0, 0.1]),
            rtol=0.0,
            atol=1e-4,
        )
        covariance_eigenvalues = linalg.eigvalsh(tracker.covariance)
        self.assertGreater(float(np.min(covariance_eigenvalues)), -1e-10)

    def test_update_moves_position_without_changing_extent(self):
        tracker = RigidEKFSplineTracker(
            covariance=diag(array([0.05, 0.05, 0.01, 0.01, 0.01])),
            measurement_noise=0.01 * eye(2),
        )
        prior_extent = tracker.get_point_estimate_extent()

        tracker.update(array([3.0, 0.0]))

        self.assertGreater(float(tracker.kinematic_state[0]), 0.0)
        npt.assert_allclose(tracker.get_point_estimate_extent(), prior_extent, atol=1e-12)
        self.assertIsNotNone(tracker.last_quadratic_form)
        covariance_eigenvalues = linalg.eigvalsh(tracker.covariance)
        self.assertGreater(float(np.min(covariance_eigenvalues)), -1e-10)

    def test_orientation_correction_can_be_disabled(self):
        tracker = RigidEKFSplineTracker(
            covariance=diag(array([0.05, 0.05, 0.2, 0.01, 0.01])),
            measurement_noise=0.01 * eye(2),
            orientation_correction=False,
        )

        tracker.update(array([2.5, 0.7]))

        self.assertEqual(float(tracker.kinematic_state[2]), 0.0)

    def test_update_accepts_multiple_measurement_layouts(self):
        measurement_rows = array([[3.0, 0.2], [0.0, 1.5], [-2.5, -0.1]])
        tracker = RigidEKFSplineTracker(measurement_noise=0.01 * eye(2))

        for measurement_batch in (measurement_rows, measurement_rows.T):
            tracker.update(measurement_batch)
            self.assertIsNotNone(tracker.last_quadratic_form)
