import unittest

import numpy.testing as npt
import pyrecest.backend
import scipy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.multi_bernoulli_tracker import (
    BernoulliComponent,
    MultiBernoulliTracker,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
class MultiBernoulliTrackerTest(unittest.TestCase):
    """Test case for the MultiBernoulliTracker class."""

    def setUp(self):
        self.measurement_matrix = array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.system_matrix = array(
            scipy.linalg.block_diag(
                array([[1.0, 1.0], [0.0, 1.0]]),
                array([[1.0, 1.0], [0.0, 1.0]]),
            )
        )
        self.birth_covariance = diag(array([4.0, 25.0, 4.0, 25.0]))

        self.initial_components = [
            BernoulliComponent(
                0.8,
                KalmanFilter(
                    GaussianDistribution(zeros(4), diag(array([1.0, 1.0, 1.0, 1.0])))
                ),
            ),
            BernoulliComponent(
                0.3,
                KalmanFilter(
                    GaussianDistribution(
                        array([10.0, 0.0, 20.0, 0.0]),
                        diag(array([2.0, 2.0, 2.0, 2.0])),
                    )
                ),
            ),
        ]

    def test_cardinality_and_state_extraction(self):
        tracker = MultiBernoulliTracker(initial_prior=self.initial_components)

        cardinality_distribution = tracker.get_cardinality_distribution()
        npt.assert_allclose(cardinality_distribution, array([0.14, 0.62, 0.24]))
        self.assertEqual(tracker.get_number_of_targets(), 1)
        self.assertAlmostEqual(tracker.get_expected_number_of_targets(), 1.1)
        self.assertEqual(tracker.get_point_estimate().shape, (4, 1))
        npt.assert_array_equal(tracker.get_point_estimate().flatten(), zeros(4))

    def test_predict_linear_updates_existence_and_state(self):
        tracker = MultiBernoulliTracker(
            initial_prior=[self.initial_components[0]],
            tracker_param={
                "survival_probability": 0.9,
                "detection_probability": 0.95,
                "clutter_intensity": 1e-9,
                "gating_probability": 0.999,
                "gating_distance_threshold": None,
                "pruning_threshold": 1e-4,
                "maximum_number_of_components": None,
                "birth_existence_probability": 0.8,
                "birth_covariance": self.birth_covariance,
                "measurement_to_state_matrix": None,
            },
        )
        tracker.predict_linear(self.system_matrix, eye(4))

        npt.assert_array_equal(
            tracker.bernoulli_components[0].get_point_estimate(),
            zeros(4),
        )
        npt.assert_array_equal(
            tracker.bernoulli_components[0].filter_state.C,
            scipy.linalg.block_diag([[3.0, 1.0], [1.0, 2.0]], [[3.0, 1.0], [1.0, 2.0]]),
        )
        self.assertAlmostEqual(
            tracker.bernoulli_components[0].existence_probability,
            0.72,
        )

    def test_update_linear_with_measurement_reinforces_track(self):
        tracker = MultiBernoulliTracker(
            initial_prior=[self.initial_components[0]],
            tracker_param={
                "survival_probability": 0.99,
                "detection_probability": 0.9,
                "clutter_intensity": 1e-12,
                "gating_probability": 0.999,
                "gating_distance_threshold": None,
                "pruning_threshold": 1e-4,
                "maximum_number_of_components": None,
                "birth_existence_probability": 0.8,
                "birth_covariance": self.birth_covariance,
                "measurement_to_state_matrix": None,
            },
        )
        measurement = array([[0.0], [0.0]])
        tracker.update_linear(measurement, self.measurement_matrix, eye(2))

        self.assertGreater(
            tracker.bernoulli_components[0].existence_probability,
            0.99,
        )
        npt.assert_array_equal(
            tracker.bernoulli_components[0].get_point_estimate(),
            zeros(4),
        )
        updated_covariance = tracker.bernoulli_components[0].filter_state.C
        self.assertTrue(updated_covariance[0, 0] < 1.0)
        self.assertTrue(updated_covariance[2, 2] < 1.0)
        self.assertAlmostEqual(updated_covariance[1, 1], 1.0)
        self.assertAlmostEqual(updated_covariance[3, 3], 1.0)

    def test_update_linear_without_measurements_reduces_existence(self):
        tracker = MultiBernoulliTracker(
            initial_prior=[self.initial_components[0]],
            tracker_param={
                "survival_probability": 0.99,
                "detection_probability": 0.8,
                "clutter_intensity": 1e-9,
                "gating_probability": 0.999,
                "gating_distance_threshold": None,
                "pruning_threshold": 1e-4,
                "maximum_number_of_components": None,
                "birth_existence_probability": 0.8,
                "birth_covariance": self.birth_covariance,
                "measurement_to_state_matrix": None,
            },
        )
        tracker.update_linear(zeros((2, 0)), self.measurement_matrix, eye(2))

        self.assertAlmostEqual(
            tracker.bernoulli_components[0].existence_probability,
            0.4444444444444445,
            places=12,
        )

    def test_unassigned_measurement_creates_birth_component(self):
        tracker = MultiBernoulliTracker(
            tracker_param={
                "survival_probability": 0.99,
                "detection_probability": 0.9,
                "clutter_intensity": 1e-9,
                "gating_probability": 0.999,
                "gating_distance_threshold": None,
                "pruning_threshold": 1e-4,
                "maximum_number_of_components": None,
                "birth_existence_probability": 0.7,
                "birth_covariance": self.birth_covariance,
                "measurement_to_state_matrix": None,
            }
        )
        measurement = array([[10.0], [20.0]])
        tracker.update_linear(measurement, self.measurement_matrix, eye(2))

        self.assertEqual(tracker.get_number_of_components(), 1)
        self.assertAlmostEqual(
            tracker.bernoulli_components[0].existence_probability,
            0.7,
        )
        npt.assert_array_equal(
            tracker.bernoulli_components[0].get_point_estimate(),
            array([10.0, 0.0, 20.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
