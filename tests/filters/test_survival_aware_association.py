import math
import unittest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter, NISGate
from pyrecest.filters.survival_aware_association import (
    SurvivalAwareAssociationConfig,
    build_survival_aware_linear_gaussian_hypothesis_associator,
    survival_aware_linear_gaussian_association_hypotheses,
    survival_aware_missed_detection_costs,
    survival_aware_track_log_prior,
)
from pyrecest.filters.survival_aware_crp import (
    SurvivalAwareCRPAssociationPrior,
    SurvivalAwareTrackEvidence,
)
from pyrecest.filters.track_manager import Track, TrackStatus


def _track(mean, *, track_id=0, hits=3, misses=0, existence=1.0):
    state = GaussianDistribution(array([float(mean)]), array([[0.2]]))
    return Track(
        track_id=track_id,
        single_target_filter=KalmanFilter(state),
        status=TrackStatus.CONFIRMED,
        hits=hits,
        misses=misses,
        age=max(1, hits + misses),
        metadata={"existence_probability": existence},
    )


class SurvivalAwareAssociationTest(unittest.TestCase):
    def test_prior_delegates_to_survival_aware_crp_core(self):
        config = SurvivalAwareAssociationConfig(
            survival_probability=0.8,
            detection_probability=0.9,
            visibility_probability=0.7,
            mass_decay=0.5,
        )
        track = _track(0.0, hits=4, misses=2, existence=0.6)
        expected_prior = SurvivalAwareCRPAssociationPrior(
            temporal_decay=0.5,
            minimum_total_weight=1.0e-12,
        )
        expected_evidence = SurvivalAwareTrackEvidence(
            mass=4.0,
            existence_probability=0.6,
            survival_probability=0.8,
            detection_probability=0.9,
            visibility_probability=0.7,
            last_seen_steps=2,
        )

        expected_weight = expected_prior.existing_track_weight(expected_evidence)

        self.assertAlmostEqual(
            math.exp(survival_aware_track_log_prior(track, config=config)),
            expected_weight,
        )

    def test_prior_discounts_stale_tracks(self):
        config = SurvivalAwareAssociationConfig(
            survival_probability=0.8,
            detection_probability=0.9,
            visibility_probability=1.0,
            mass_decay=0.5,
        )
        fresh = _track(0.0, hits=4, misses=0)
        stale = _track(0.0, hits=4, misses=3)

        self.assertLess(
            survival_aware_track_log_prior(stale, config=config),
            survival_aware_track_log_prior(fresh, config=config),
        )

    def test_missed_detection_is_cheaper_when_visibility_is_low(self):
        track = _track(0.0, existence=0.9)
        high_visibility = SurvivalAwareAssociationConfig(
            detection_probability=0.9,
            visibility_probability=1.0,
        )
        low_visibility = SurvivalAwareAssociationConfig(
            detection_probability=0.9,
            visibility_probability=0.1,
        )

        high_visibility_cost = survival_aware_missed_detection_costs(
            [track],
            config=high_visibility,
        )[0]
        low_visibility_cost = survival_aware_missed_detection_costs(
            [track],
            config=low_visibility,
        )[0]

        self.assertLess(low_visibility_cost, high_visibility_cost)

    def test_missed_detection_is_cheaper_when_survival_is_low(self):
        track = _track(0.0, existence=0.9)
        high_survival = SurvivalAwareAssociationConfig(
            survival_probability=1.0,
            detection_probability=0.9,
            visibility_probability=1.0,
        )
        low_survival = SurvivalAwareAssociationConfig(
            survival_probability=0.1,
            detection_probability=0.9,
            visibility_probability=1.0,
        )

        high_survival_cost = survival_aware_missed_detection_costs(
            [track],
            config=high_survival,
        )[0]
        low_survival_cost = survival_aware_missed_detection_costs(
            [track],
            config=low_survival,
        )[0]

        self.assertLess(low_survival_cost, high_survival_cost)

    def test_linear_gaussian_hypotheses_expose_crp_prior_metadata(self):
        track = _track(0.0, hits=5, misses=0, existence=0.95)
        config = SurvivalAwareAssociationConfig(
            detection_probability=0.95,
            visibility_probability=1.0,
        )

        hypotheses = survival_aware_linear_gaussian_association_hypotheses(
            [track],
            array([[0.05]]),
            array([[1.0]]),
            array([[0.1]]),
            measurement_axis="columns",
            config=config,
        )

        self.assertEqual(len(hypotheses), 1)
        metadata = hypotheses[0].metadata
        self.assertIn("survival_aware_crp_weight", metadata)
        self.assertIn("survival_aware_measurement_log_score", metadata)
        self.assertIsNone(hypotheses[0].probability)
        self.assertAlmostEqual(
            hypotheses[0].cost, -metadata["survival_aware_log_score"]
        )

    def test_track_manager_associator_keeps_far_measurement_unassigned(self):
        tracks = [_track(0.0, hits=5, misses=0, existence=0.95)]
        measurements = array([[0.05, 6.0]])
        config = SurvivalAwareAssociationConfig(
            detection_probability=0.95,
            visibility_probability=1.0,
            birth_weight=0.05,
            clutter_weight=0.05,
        )
        associator = build_survival_aware_linear_gaussian_hypothesis_associator(
            array([[1.0]]),
            array([[0.1]]),
            gates=NISGate(threshold=9.0),
            config=config,
        )

        result = associator(tracks, measurements, measurement_axis="columns")

        self.assertEqual(result.matches, [(0, 0)])
        self.assertEqual(result.unmatched_measurement_indices, [1])


if __name__ == "__main__":
    unittest.main()
