import unittest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter, MultiHypothesisTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Currently supported for numpy backend only",
)
class MultiHypothesisTrackerSelectorValidationTest(unittest.TestCase):
    def setUp(self):
        association_param = {
            "gating_distance_threshold": 1.0e6,
            "max_global_hypotheses": 5,
            "max_hypotheses_per_global_hypothesis": 5,
            "max_measurements_per_track": 5,
            "detection_probability": 0.9,
            "clutter_intensity": 1.0e-6,
        }
        self.tracker = MultiHypothesisTracker(
            association_param=association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        self.tracker.filter_state = [
            KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2))),
            KalmanFilter(GaussianDistribution(array([10.0, 0.0]), eye(2))),
        ]
        self.tracker.update_linear(
            array([[9.9, 0.1], [0.2, -0.1]]),
            eye(2),
            eye(2),
        )

    def test_rejects_lossy_or_boolean_selectors(self):
        invalid_calls = {
            "fractional k": lambda: self.tracker.get_top_hypotheses(k=1.5),
            "boolean k": lambda: self.tracker.get_top_hypotheses(k=True),
            "fractional lag": lambda: self.tracker.get_assignment_distribution(
                lag=0.5
            ),
            "boolean lag": lambda: self.tracker.get_assignment_distribution(lag=True),
            "fractional time index": lambda: self.tracker.get_assignment_distribution(
                time_index=0.5
            ),
            "fractional hypothesis index": lambda: self.tracker.get_lagged_point_estimate(
                lag=0, hypothesis_index=0.5
            ),
        }
        for label, call in invalid_calls.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    call()

        with self.assertRaisesRegex(ValueError, "k must be non-negative"):
            self.tracker.get_top_hypotheses(k=-1)

    def test_accepts_exact_backend_integer_scalars(self):
        self.assertEqual(len(self.tracker.get_top_hypotheses(k=array(1))), 1)
        self.assertEqual(self.tracker.get_top_hypotheses(k=0), [])

        distribution = self.tracker.get_assignment_distribution(lag=array(0))
        self.assertAlmostEqual(sum(distribution.values()), 1.0)

        indexed_distribution = self.tracker.get_assignment_distribution(
            time_index=array(-1)
        )
        self.assertEqual(distribution, indexed_distribution)


if __name__ == "__main__":
    unittest.main()
