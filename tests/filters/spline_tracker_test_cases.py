import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import all as backend_all
from pyrecest.backend import array, diag, eye, linalg
from pyrecest.filters import (
    EKFSplineTracker,
    UKFSplineTracker,
    EkfSplineTracker,
    UkfSplineTracker,
)


_TRACKER_CASES = {
    "ekf": {
        "tracker_class": EKFSplineTracker,
        "tracker_alias": EkfSplineTracker,
        "skip_reason": "Finite-difference spline tracker tests are numpy-backend checks.",
        "initial_covariance_diag": (0.1, 0.1, 0.01, 0.01, 0.01, 0.2, 0.2),
        "outer_measurement_covariance_diag": (
            0.05,
            0.05,
            0.01,
            0.01,
            0.01,
            0.2,
            0.01,
        ),
        "scale_correction_covariance_diag": (
            0.05,
            0.05,
            0.01,
            0.01,
            0.01,
            0.2,
            0.2,
        ),
        "scale_correction_atol": 1e-10,
    },
    "ukf": {
        "tracker_class": UKFSplineTracker,
        "tracker_alias": UkfSplineTracker,
        "skip_reason": "Unscented spline tracker tests are numpy-backend checks.",
        "initial_covariance_diag": (0.1, 0.1, 0.01, 0.01, 0.01, 0.05, 0.05),
        "outer_measurement_covariance_diag": (
            0.05,
            0.05,
            0.01,
            0.01,
            0.01,
            0.05,
            0.01,
        ),
        "scale_correction_covariance_diag": (
            0.05,
            0.05,
            0.01,
            0.01,
            0.01,
            0.05,
            0.05,
        ),
        "scale_correction_atol": 1e-8,
    },
}


class SplineTrackerCommonTests:
    tracker_key: str = ""

    @property
    def tracker_case(self):
        return _TRACKER_CASES[self.tracker_key]

    def make_tracker(self, **kwargs):
        kwargs.setdefault("measurement_noise", 0.01 * eye(2))
        return self.tracker_case["tracker_class"](**kwargs)

    def setUp(self):
        if pyrecest.backend.__backend_name__ != "numpy":
            self.skipTest(self.tracker_case["skip_reason"])

        self.tracker = self.make_tracker(
            kinematic_state=array([0.0, 0.0, 0.0, 0.0, 0.0]),
            scale_state=array([1.0, 1.0]),
            covariance=diag(array(self.tracker_case["initial_covariance_diag"])),
        )

    def test_initialization_and_alias(self):
        self.assertIs(
            self.tracker_case["tracker_alias"],
            self.tracker_case["tracker_class"],
        )
        self.assertEqual(self.tracker.state.shape, (7,))
        self.assertEqual(self.tracker.get_scaled_control_points().shape, (8, 2))
        self.assertEqual(
            self.tracker.get_point_estimate_extent(flatten_matrix=True).shape,
            (16,),
        )

    def test_contour_and_bounding_box_shapes(self):
        contour = self.tracker.get_contour_points(16)
        bounding_box = self.tracker.get_bounding_box(32)

        self.assertEqual(contour.shape, (16, 2))
        self.assertEqual(bounding_box["center_xy"].shape, (2,))
        self.assertEqual(bounding_box["dimension"].shape, (2,))
        self.assertGreater(float(bounding_box["dimension"][0]), 0.0)
        self.assertGreater(float(bounding_box["dimension"][1]), 0.0)

    def test_predict_uses_constant_turn_kinematics(self):
        tracker = self.make_tracker(
            kinematic_state=array([0.0, 0.0, 0.0, 2.0, 0.2]),
            covariance=diag(array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])),
        )

        tracker.predict(dt=0.5)

        npt.assert_allclose(
            tracker.kinematic_state[:3],
            array([1.0, 0.0, 0.1]),
            atol=1e-4,
        )
        self.assertTrue(backend_all(linalg.eigvalsh(tracker.covariance) > -1e-10))

    def test_update_moves_position_and_scale_for_outer_measurement(self):
        tracker = self.make_tracker(
            covariance=diag(
                array(self.tracker_case["outer_measurement_covariance_diag"])
            ),
        )

        tracker.update(array([3.0, 0.0]))

        self.assertGreater(float(tracker.kinematic_state[0]), 0.0)
        self.assertGreater(float(tracker.scale_state[0]), 1.0)
        self.assertIsNotNone(tracker.last_quadratic_form)
        self.assertTrue(backend_all(linalg.eigvalsh(tracker.covariance) > -1e-10))

    def test_scale_correction_can_be_disabled(self):
        tracker = self.make_tracker(
            covariance=diag(
                array(self.tracker_case["scale_correction_covariance_diag"])
            ),
            scale_correction=False,
        )

        tracker.update(array([3.0, 0.0]))

        npt.assert_allclose(
            tracker.scale_state,
            array([1.0, 1.0]),
            atol=self.tracker_case["scale_correction_atol"],
        )

    def test_update_accepts_multiple_measurement_layouts(self):
        measurements = array([[3.0, 0.2], [0.0, 1.5], [-2.5, -0.1]])
        tracker = self.make_tracker()

        tracker.update(measurements)
        tracker.update(measurements.T)

        self.assertIsNotNone(tracker.last_quadratic_form)
