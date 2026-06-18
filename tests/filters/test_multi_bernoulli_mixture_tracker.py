import unittest
from unittest.mock import patch

import numpy.testing as npt
import pyrecest.backend
import pyrecest.filters as filters_namespace

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.multi_bernoulli_mixture_tracker import (
    MultiBernoulliMixtureHypothesis,
    MultiBernoulliMixtureTracker,
    NearestNeighborMultiBernoulliTracker,
)
from pyrecest.filters.multi_bernoulli_tracker import (
    BernoulliComponent,
    MultiBernoulliTracker,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
class MultiBernoulliMixtureTrackerTest(unittest.TestCase):
    def setUp(self):
        self.measurement_matrix = array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.initial_component = BernoulliComponent(
            0.8,
            GaussianDistribution(zeros(4), diag(array([1.0, 1.0, 1.0, 1.0]))),
        )
        self.tracker_param = {
            "survival_probability": 0.99,
            "detection_probability": 0.8,
            "clutter_intensity": 0.5,
            "gating_probability": 0.999,
            "gating_distance_threshold": None,
            "pruning_threshold": 1e-12,
            "hypothesis_pruning_threshold": 1e-12,
            "maximum_number_of_components": None,
            "maximum_number_of_hypotheses": None,
            "birth_existence_probability": 0.8,
            "birth_covariance": None,
            "measurement_to_state_matrix": None,
            "measurement_driven_births": False,
        }

    def test_nearest_neighbor_alias_preserves_existing_tracker(self):
        self.assertIs(NearestNeighborMultiBernoulliTracker, MultiBernoulliTracker)

    def test_public_filter_namespace_exports_mbm_trackers(self):
        self.assertIs(
            filters_namespace.MultiBernoulliMixtureHypothesis,
            MultiBernoulliMixtureHypothesis,
        )
        self.assertIs(
            filters_namespace.MultiBernoulliMixtureTracker,
            MultiBernoulliMixtureTracker,
        )
        self.assertIs(
            filters_namespace.NearestNeighborMultiBernoulliTracker,
            MultiBernoulliTracker,
        )

    def test_predict_linear_rejects_nonzero_mean_gaussian_system_noise(self):
        tracker = MultiBernoulliMixtureTracker(
            initial_prior=[self.initial_component],
            tracker_param=self.tracker_param,
        )

        with self.assertRaisesRegex(ValueError, "zero mean"):
            tracker.predict_linear(
                eye(4),
                GaussianDistribution(array([1.0, 0.0, 0.0, 0.0]), eye(4)),
            )

    def test_update_linear_rejects_unsupported_backend(self):
        tracker = MultiBernoulliMixtureTracker(
            initial_prior=[self.initial_component],
            tracker_param=self.tracker_param,
        )

        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                tracker.update_linear(
                    array([[0.0], [0.0]]),
                    self.measurement_matrix,
                    eye(2),
                )

    def test_exact_mbm_update_keeps_missed_and_detected_hypotheses(self):
        tracker = MultiBernoulliMixtureTracker(
            initial_prior=[self.initial_component],
            tracker_param=self.tracker_param,
        )

        measurement = array([[0.0], [0.0]])
        prior_component = tracker.hypotheses[0].bernoulli_components[0]
        likelihood, _ = (
            tracker._measurement_likelihood_and_distance(  # pylint: disable=protected-access
                prior_component,
                measurement[:, 0],
                self.measurement_matrix,
                eye(2),
            )
        )
        r = prior_component.existence_probability
        p_d = self.tracker_param["detection_probability"]
        clutter = self.tracker_param["clutter_intensity"]
        missed_weight = 1.0 - r * p_d
        detected_weight = r * p_d * likelihood / clutter
        missed_existence = r * (1.0 - p_d) / missed_weight
        expected_existence = (missed_weight * missed_existence + detected_weight) / (
            missed_weight + detected_weight
        )

        tracker.update_linear(measurement, self.measurement_matrix, eye(2))

        self.assertEqual(len(tracker.hypotheses), 2)
        npt.assert_allclose(sum(tracker.get_mixture_weights()), 1.0)
        npt.assert_allclose(
            tracker.get_existence_probabilities()[0],
            expected_existence,
        )

    def test_exact_mbm_hypothesis_cap(self):
        tracker_param = dict(self.tracker_param)
        tracker_param["maximum_number_of_hypotheses"] = 1
        tracker = MultiBernoulliMixtureTracker(
            initial_prior=[self.initial_component],
            tracker_param=tracker_param,
        )

        tracker.update_linear(array([[0.0], [0.0]]), self.measurement_matrix, eye(2))

        self.assertEqual(len(tracker.hypotheses), 1)
        npt.assert_allclose(tracker.get_mixture_weights()[0], 1.0)

    def test_measurement_driven_birth_labels_are_stable_across_hypotheses(self):
        tracker_param = dict(self.tracker_param)
        tracker_param.update(
            {
                "birth_covariance": diag(array([4.0, 25.0, 4.0, 25.0])),
                "measurement_driven_births": True,
                "maximum_number_of_hypotheses": None,
                "hypothesis_pruning_threshold": 1e-12,
            }
        )
        tracker = MultiBernoulliMixtureTracker(
            initial_prior=[self.initial_component],
            tracker_param=tracker_param,
        )
        existing_label = tracker.hypotheses[0].bernoulli_components[0].label

        tracker.update_linear(
            array([[0.0, 1.0], [0.0, 0.0]]),
            self.measurement_matrix,
            eye(2),
        )

        labels_by_birth_position = {}
        for hypothesis in tracker.hypotheses:
            for component in hypothesis.bernoulli_components:
                if component.label == existing_label:
                    continue
                estimate = component.get_point_estimate()
                key = (
                    round(float(estimate[0]), 12),
                    round(float(estimate[2]), 12),
                )
                labels_by_birth_position.setdefault(key, set()).add(component.label)

        self.assertEqual(
            set(labels_by_birth_position),
            {(0.0, 0.0), (1.0, 0.0)},
        )
        self.assertTrue(
            all(len(labels) == 1 for labels in labels_by_birth_position.values())
        )


if __name__ == "__main__":
    unittest.main()
