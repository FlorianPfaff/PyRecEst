import unittest

import numpy.testing as npt

from pyrecest.filters.survival_aware_crp import (
    SurvivalAwareCRPAssociationPrior,
    SurvivalAwareTrackEvidence,
)


class SurvivalAwareCRPAssociationPriorTest(unittest.TestCase):
    def test_predictive_probabilities_match_hand_computed_weights(self):
        prior = SurvivalAwareCRPAssociationPrior(
            concentration=2.0,
            discount=0.25,
            temporal_decay=0.5,
        )
        track_evidence = SurvivalAwareTrackEvidence(
            mass=4.0,
            existence_probability=0.8,
            survival_probability=0.9,
            detection_probability=0.7,
            visibility_probability=0.5,
            kinematic_likelihood=0.25,
            appearance_likelihood=0.5,
            last_seen_steps=1,
        )

        probabilities = prior.predictive_assignment_probabilities(
            [track_evidence],
            base_birth_weight=0.3,
            clutter_weight=0.25,
        )

        existing_weight = (
            (4.0 * 0.5 - 0.25)
            * 0.8
            * 0.9
            * 0.7
            * 0.5
            * 0.25
            * 0.5
        )
        birth_weight = 0.3 * (2.0 + 0.25 * 1.0)
        clutter_weight = 0.25
        total_weight = existing_weight + birth_weight + clutter_weight

        npt.assert_allclose(
            probabilities.existing_track_probabilities,
            (existing_weight / total_weight,),
        )
        self.assertAlmostEqual(
            probabilities.birth_probability,
            birth_weight / total_weight,
        )
        self.assertAlmostEqual(
            probabilities.clutter_probability,
            clutter_weight / total_weight,
        )
        self.assertAlmostEqual(probabilities.total_probability, 1.0)

    def test_survival_and_compatibility_can_override_raw_track_mass(self):
        prior = SurvivalAwareCRPAssociationPrior(
            concentration=0.5,
            temporal_decay=0.5,
        )
        stale_popular_track = SurvivalAwareTrackEvidence(
            mass=100.0,
            existence_probability=0.2,
            survival_probability=0.1,
            detection_probability=0.9,
            visibility_probability=0.9,
            kinematic_likelihood=0.01,
            last_seen_steps=5,
        )
        fresh_compatible_track = SurvivalAwareTrackEvidence(
            mass=2.0,
            existence_probability=0.9,
            survival_probability=0.99,
            detection_probability=0.9,
            visibility_probability=1.0,
            kinematic_likelihood=1.0,
            last_seen_steps=0,
        )

        probabilities = prior.predictive_assignment_probabilities(
            [stale_popular_track, fresh_compatible_track],
            base_birth_weight=0.1,
            clutter_weight=0.1,
        )

        self.assertLess(
            probabilities.existing_track_probabilities[0],
            probabilities.existing_track_probabilities[1],
        )

    def test_missed_detection_update_is_visibility_aware(self):
        visible_posterior = (
            SurvivalAwareCRPAssociationPrior.missed_detection_existence_probability(
                predicted_existence_probability=0.8,
                detection_probability=0.9,
                visibility_probability=1.0,
            )
        )
        occluded_posterior = (
            SurvivalAwareCRPAssociationPrior.missed_detection_existence_probability(
                predicted_existence_probability=0.8,
                detection_probability=0.9,
                visibility_probability=0.1,
            )
        )

        self.assertLess(visible_posterior, occluded_posterior)
        self.assertAlmostEqual(visible_posterior, 0.08 / 0.28)
        self.assertAlmostEqual(occluded_posterior, 0.728 / 0.928)

    def test_dict_track_evidence_is_accepted(self):
        prior = SurvivalAwareCRPAssociationPrior(concentration=1.0)

        probabilities = prior.predictive_assignment_probabilities(
            [{"mass": 2.0, "existence_probability": 0.5}],
            base_birth_weight=1.0,
        )

        npt.assert_allclose(probabilities.as_tuple, (0.5, 0.5, 0.0))

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(TypeError):
            SurvivalAwareCRPAssociationPrior(concentration=True)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(discount=1.0)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior(temporal_decay=1.1)
        with self.assertRaises(TypeError):
            SurvivalAwareTrackEvidence(last_seen_steps=True)
        with self.assertRaises(ValueError):
            SurvivalAwareTrackEvidence(kinematic_likelihood=-1.0)
        with self.assertRaises(ValueError):
            SurvivalAwareCRPAssociationPrior().predictive_assignment_probabilities(
                [],
                base_birth_weight=0.0,
                clutter_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
