import unittest

from pyrecest.backend import array
from pyrecest.utils import CalibratedPairwiseAssociationModel


class NonfiniteMatchProbabilityModel:
    def predict_match_probability(self, features):
        del features
        return array([0.2, float("nan")])


class NonfinitePredictProbaModel:
    classes_ = array([0, 1])

    def predict_proba(self, features):
        return array([[0.8, 0.2], [0.3, float("nan")]])[: features.shape[0]]


class NonfiniteCostModel:
    def pairwise_cost_matrix(self, features):
        del features
        return array([0.0, float("-inf")])


class TestCalibratedAssociationNonfiniteProbabilities(unittest.TestCase):
    def test_rejects_nonfinite_match_probabilities(self):
        model = CalibratedPairwiseAssociationModel(
            NonfiniteMatchProbabilityModel(), feature_names=("distance",)
        )

        with self.assertRaisesRegex(
            ValueError, "predicted probabilities must be finite"
        ):
            model.predict_match_probability(array([[0.1], [0.2]]))

    def test_rejects_nonfinite_predict_proba_values(self):
        model = CalibratedPairwiseAssociationModel(
            NonfinitePredictProbaModel(), feature_names=("distance",)
        )

        with self.assertRaisesRegex(
            ValueError, "predicted probabilities must be finite"
        ):
            model.predict_match_probability(array([[0.1], [0.2]]))

    def test_rejects_nonfinite_cost_derived_probabilities(self):
        model = CalibratedPairwiseAssociationModel(
            NonfiniteCostModel(), feature_names=("distance",)
        )

        with self.assertRaisesRegex(
            ValueError, "predicted probabilities must be finite"
        ):
            model.predict_match_probability(array([[0.1], [0.2]]))


if __name__ == "__main__":
    unittest.main()
