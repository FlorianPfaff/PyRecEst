import numpy as np
import pytest
from pyrecest.utils import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    prune_pairwise_cost_matrix,
)


def test_configured_candidate_pruning_rejects_nan_costs() -> None:
    costs = np.array([[1.0, np.nan], [2.0, 3.0]])
    config = CandidatePruningConfig(row_top_k=1)
    message = "cost_matrix may only contain finite values or positive infinity"

    with pytest.raises(ValueError, match=message):
        candidate_mask_from_costs(costs, config=config)

    with pytest.raises(ValueError, match=message):
        prune_pairwise_cost_matrix(costs, config=config)
