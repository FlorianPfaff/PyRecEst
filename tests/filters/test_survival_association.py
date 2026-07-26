import math
import unittest
from dataclasses import dataclass, field

import numpy as np
from pyrecest.filters.association_hypotheses import AssociationHypothesis
from pyrecest.filters.survival_association import (
    SurvivalAwareAssociationConfig,
    apply_survival_association_prior,
    survival_aware_birth_costs,
    survival_aware_missed_detection_costs,
    track_survival_prior_components,
)


@dataclass
class DummyTrack:
    hits: int = 1
    misses: int = 0
    metadata: dict = field(default_factory=dict)


class SurvivalAssociationTest(unittest.TestCase):
    def test_resolves_survival_weight_from_track_lifecycle(self):
        tracks = [
            DummyTrack(hits=4, misses=2, metadata={"visibility_probability": 0.25})
        ]
        config = SurvivalAwareAssociationConfig(
            existence_probability=0.8,
            survival_probability=0.5,
            detection_probability=0.5,
            track_mass_decay=0.5,
            minimum_track_mass=0.0,
        )

        components = track_survival_prior_components(tracks, config=config)

        self.assertEqual(components[0].steps_since_seen, 3)
        self.assertAlmostEqual(components[0].track_mass, 1.0)
        self.assertAlmostEqual(components[0].survival_probability, 0.125)
        self.assertAlmostEqual(components[0].assignment_weight, 0.0125)
        self.assertAlmostEqual(
            components[0].missed_detection_cost,
            -math.log(1.0 - 0.8 * 0.125 * 0.5 * 0.25),
        )

    def test_prior_can_prefer_recent_track_over_popular_stale_track(self):
        tracks = [DummyTrack(hits=1, misses=0), DummyTrack(hits=10, misses=4)]
        hypotheses = [
            AssociationHypothesis(
                track_index=0, measurement_index=0, cost=0.0, log_likelihood=0.0
            ),
            AssociationHypothesis(
                track_index=1, measurement_index=0, cost=0.0, log_likelihood=0.0
            ),
        ]
        config = SurvivalAwareAssociationConfig(
            existence_probability=1.0,
            survival_probability=0.5,
            detection_probability=1.0,
            visibility_probability=1.0,
            minimum_track_mass=0.0,
        )

        adjusted = apply_survival_association_prior(hypotheses, tracks, config=config)

        self.assertLess(adjusted[0].cost, adjusted[1].cost)
        self.assertAlmostEqual(adjusted[0].cost, -math.log(0.5))
        self.assertAlmostEqual(adjusted[1].cost, -math.log(10.0 * 0.5**5))
        self.assertIn("survival_prior", adjusted[0].metadata)

    def test_missed_detection_cost_decreases_for_low_visibility_track(self):
        tracks = [
            DummyTrack(metadata={"visibility_probability": 1.0}),
            DummyTrack(metadata={"visibility_probability": 0.1}),
        ]
        config = SurvivalAwareAssociationConfig(
            existence_probability=0.9,
            survival_probability=1.0,
            detection_probability=0.8,
        )

        costs = survival_aware_missed_detection_costs(tracks, config=config)

        self.assertGreater(costs[0], costs[1])

    def test_birth_costs_accept_per_measurement_probabilities(self):
        config = SurvivalAwareAssociationConfig(birth_probability=[0.25, 1.0])

        costs = survival_aware_birth_costs(
            [np.array([0.0]), np.array([1.0])],
            config=config,
            measurement_axis="sequence",
        )

        np.testing.assert_allclose(costs, np.array([-math.log(0.25), 0.0]))

    def test_probability_scalars_reject_text_bool_and_temporal_payloads(self):
        invalid_values = [
            "0.5",
            b"0.5",
            np.array(True, dtype=object),
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001", "ns"),
            np.array(np.timedelta64(1, "ns"), dtype=object),
        ]
        for value in invalid_values:
            with self.subTest(value=repr(value)):
                config = SurvivalAwareAssociationConfig(survival_probability=value)
                with self.assertRaisesRegex(ValueError, "survival_probability"):
                    track_survival_prior_components([DummyTrack()], config=config)

    def test_numeric_object_probability_scalar_is_preserved(self):
        config = SurvivalAwareAssociationConfig(
            survival_probability=np.array(0.5, dtype=object)
        )

        components = track_survival_prior_components([DummyTrack()], config=config)

        self.assertAlmostEqual(components[0].survival_probability, 0.5)


if __name__ == "__main__":
    unittest.main()
