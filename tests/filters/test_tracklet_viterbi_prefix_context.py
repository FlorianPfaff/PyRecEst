from pyrecest.filters.tracklet_viterbi import (
    TrackletAssociationCandidate,
    TrackletViterbiConfig,
    solve_fixed_lag_tracklet_viterbi,
    solve_tracklet_viterbi,
)


def test_fixed_lag_prefix_context_affects_selection_after_initial_gap():
    frames = [
        [],
        [TrackletAssociationCandidate("candidate", unary_cost=4.0)],
    ]
    config = TrackletViterbiConfig(
        missed_detection_cost=2.0,
        consecutive_miss_cost=5.0,
    )

    full = solve_tracklet_viterbi(frames, config=config)
    fixed_lag = solve_fixed_lag_tracklet_viterbi(
        frames,
        lag_s=0.1,
        config=config,
    )

    assert full.path == [None, frames[1][0]]
    assert full.total_cost == 6.0
    assert fixed_lag.path == full.path
    assert fixed_lag.total_cost == full.total_cost
