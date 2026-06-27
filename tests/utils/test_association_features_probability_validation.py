from __future__ import annotations

import numpy as np
import pytest
from pyrecest.utils.association_features import CalibratedPairwiseAssociationModel


class _MatchProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_match_probability(self, _features):
        return self.probabilities


class _PredictProbaModel:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, _features):
        return self.probabilities


class _PairwiseCostModel:
    def __init__(self, costs):
        self.costs = costs

    def pairwise_cost_matrix(self, _features):
        return self.costs


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([True, False], dtype=bool),
        np.array(["0.25", "0.75"]),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
    ],
)
def test_predict_match_probability_rejects_non_real_probability_values(probabilities):
    model = CalibratedPairwiseAssociationModel(
        _MatchProbabilityModel(probabilities),
        feature_names=["distance"],
    )

    with pytest.raises(
        ValueError, match="predicted probabilities must be real numeric"
    ):
        model.predict_match_probability(np.array([[0.0], [1.0]]))


def test_predict_proba_rejects_non_real_probabilities_before_class_selection():
    model = CalibratedPairwiseAssociationModel(
        _PredictProbaModel(np.array([[False, True]], dtype=bool)),
        feature_names=["distance"],
    )

    with pytest.raises(
        ValueError, match="predicted probabilities must be real numeric"
    ):
        model.predict_match_probability(np.array([[0.0]]))


def test_pairwise_cost_model_rejects_temporal_cost_values():
    model = CalibratedPairwiseAssociationModel(
        _PairwiseCostModel(np.array(["2026-01-01"], dtype="datetime64[D]")),
        feature_names=["distance"],
    )

    with pytest.raises(
        ValueError, match="predicted pairwise costs must be real numeric"
    ):
        model.predict_match_probability(np.array([[0.0]]))
