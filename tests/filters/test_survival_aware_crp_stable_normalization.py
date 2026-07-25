import unittest

import numpy.testing as npt
from pyrecest.filters.survival_aware_crp import (
    SurvivalAwareCRPAssociationPrior,
    SurvivalAwareTrackEvidence,
)


class SurvivalAwareCRPStableNormalizationTest(unittest.TestCase):
    def test_predictive_probabilities_normalize_large_weights_without_overflow(self):
        prior = SurvivalAwareCRPAssociationPrior(concentration=1.0)

        probabilities = prior.predictive_assignment_probabilities(
            [
                SurvivalAwareTrackEvidence(mass=1.0e308),
                SurvivalAwareTrackEvidence(mass=1.0e308),
            ],
            base_birth_weight=0.0,
            clutter_weight=0.0,
        )

        npt.assert_allclose(probabilities.existing_track_probabilities, (0.5, 0.5))
        self.assertEqual(probabilities.birth_probability, 0.0)
        self.assertEqual(probabilities.clutter_probability, 0.0)
        self.assertAlmostEqual(probabilities.total_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
