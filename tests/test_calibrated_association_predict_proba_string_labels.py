import unittest

import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.utils import CalibratedPairwiseAssociationModel


class TestCalibratedAssociationPredictProbaStringLabels(unittest.TestCase):
    def test_predict_proba_respects_stringified_binary_class_order(self):
        class StringBinaryPredictProbaModel:
            classes_ = ["1", "0"]

            def predict_proba(self, features):
                del features
                return array([[0.8, 0.2], [0.3, 0.7]])

        calibrated_model = CalibratedPairwiseAssociationModel(
            StringBinaryPredictProbaModel(), feature_names=("distance", "similarity")
        )
        features = array([[0.1, 0.9], [2.0, 0.1]])

        probabilities = calibrated_model.predict_match_probability(features)

        npt.assert_allclose(probabilities, array([0.8, 0.3]))


if __name__ == "__main__":
    unittest.main()
