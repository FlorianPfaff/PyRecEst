import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.utils import CalibratedPairwiseAssociationModel


class ConstantProbabilityModel:
    def predict_match_probability(self, features):
        del features
        return array([0.5])


class TestAssociationProbabilityClipValidation(unittest.TestCase):
    def test_rejects_malformed_probability_clip_values(self):
        invalid_values = (
            [1.0e-12],
            np.array([1.0e-12]),
            True,
            np.nan,
            np.inf,
        )

        for probability_clip in invalid_values:
            with self.subTest(probability_clip=probability_clip):
                with self.assertRaisesRegex(ValueError, "probability_clip"):
                    CalibratedPairwiseAssociationModel(
                        ConstantProbabilityModel(),
                        feature_names=("distance",),
                        probability_clip=probability_clip,
                    )

    def test_accepts_numpy_scalar_probability_clip(self):
        model = CalibratedPairwiseAssociationModel(
            ConstantProbabilityModel(),
            feature_names=("distance",),
            probability_clip=np.float64(1.0e-3),
        )

        self.assertEqual(model.probability_clip, 1.0e-3)


if __name__ == "__main__":
    unittest.main()
