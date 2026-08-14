"""Regression tests for calibrated association feature validation."""

import pytest
from pyrecest.backend import array
from pyrecest.utils import CalibratedPairwiseAssociationModel


class _RecordingPredictProbaModel:
    classes_ = array([0, 1])

    def __init__(self):
        self.called = False

    def predict_proba(self, features):
        self.called = True
        return array([[0.25, 0.75]])


def test_predict_proba_rejects_complex_direct_features_before_model_call():
    model = _RecordingPredictProbaModel()
    calibrated_model = CalibratedPairwiseAssociationModel(
        model,
        feature_names=("distance", "similarity"),
    )
    complex_features = array([[1.0 + 2.0j, 0.5]])

    with pytest.raises(ValueError, match="real numeric"):
        calibrated_model.predict_match_probability(complex_features)

    assert not model.called
