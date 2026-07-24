import numpy as np
import numpy.testing as npt
from pyrecest.utils import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    prune_pairwise_cost_matrix,
)
from scipy.optimize import linear_sum_assignment


def test_pruned_cost_cannot_undercut_complete_retained_assignment():
    costs = np.array(
        [
            [600_000.0, 600_001.0],
            [0.0, 600_000.0],
        ]
    )
    config = CandidatePruningConfig(max_cost=600_000.0)

    retained = candidate_mask_from_costs(costs, config=config)
    pruned_costs = prune_pairwise_cost_matrix(costs, config=config)
    row_indices, column_indices = linear_sum_assignment(pruned_costs)

    npt.assert_array_equal(column_indices, np.array([0, 1]))
    assert np.all(retained[row_indices, column_indices])
