import numpy as np
import pytest
from pyrecest.utils import CandidatePruningConfig, candidate_mask_from_costs


def _nested_object_matrix(value):
    matrix = np.empty((1, 1), dtype=object)
    matrix[0, 0] = np.asarray(value)
    return matrix


def test_candidate_pruning_rejects_nested_complex_cost_values():
    costs = _nested_object_matrix(1.0 + 2.0j)

    with pytest.raises(ValueError, match="cost_matrix must be real-valued numeric"):
        candidate_mask_from_costs(costs)


def test_candidate_pruning_rejects_nested_complex_probability_values():
    probabilities = _nested_object_matrix(0.75 + 0.25j)

    with pytest.raises(
        ValueError,
        match="probability_matrix must be real-valued numeric",
    ):
        candidate_mask_from_costs(
            np.array([[1.0]]),
            probability_matrix=probabilities,
            config=CandidatePruningConfig(probability_threshold=0.5),
        )


def test_candidate_pruning_accepts_nested_real_scalar_arrays():
    mask = candidate_mask_from_costs(
        _nested_object_matrix(1.0),
        probability_matrix=_nested_object_matrix(0.75),
        config=CandidatePruningConfig(probability_threshold=0.5),
    )

    np.testing.assert_array_equal(mask, np.array([[True]]))
