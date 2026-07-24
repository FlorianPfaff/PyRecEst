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


def test_default_invalid_pair_cost_cannot_undercut_large_valid_matches():
    association = solve_global_nearest_neighbor(
        np.array([[5.0e12, np.inf], [6.0e12, 7.0e12]]),
        unassigned_track_cost=1.0e13,
        unassigned_measurement_cost=1.0e13,
    )

    assert association.matches == [(0, 0), (1, 1)]
    assert association.unmatched_track_indices == []
    assert association.unmatched_measurement_indices == []
