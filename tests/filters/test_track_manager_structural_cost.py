import numpy as np
from pyrecest.filters.track_manager import solve_global_nearest_neighbor


def test_invalid_pair_replacement_is_not_used_for_structural_dummy_edges():
    # Matching at cost 150 is cheaper than paying both missed costs (100 + 100).
    association = solve_global_nearest_neighbor(
        np.array([[150.0], [160.0]]),
        unassigned_track_cost=100.0,
        unassigned_measurement_cost=100.0,
        invalid_cost=1.0,
    )

    assert association.matches == [(0, 0)]
    assert association.unmatched_track_indices == [1]
    assert association.unmatched_measurement_indices == []
