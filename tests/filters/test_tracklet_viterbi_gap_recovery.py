"""Regression coverage for fixed-lag transition context after a miss."""

from pyrecest.filters.tracklet_viterbi import (
    TrackletAssociationCandidate,
    TrackletViterbiConfig,
    solve_fixed_lag_tracklet_viterbi,
    solve_tracklet_viterbi,
)


def test_fixed_lag_recovery_after_committed_gap_uses_missing_previous_context():
    first = TrackletAssociationCandidate("d0", track_id="A", time_s=0.0)
    recovered = TrackletAssociationCandidate("d2", track_id="B", time_s=2.0)
    frames = [[first], [], [recovered]]
    config = TrackletViterbiConfig(
        missed_detection_cost=5.0,
        consecutive_miss_cost=0.0,
    )

    def transition(previous, current, miss_streak):
        if current is None:
            return 1.0
        if previous is None and miss_streak > 0:
            return 0.0
        if previous is not None and miss_streak > 0:
            return 100.0
        return 0.0

    full = solve_tracklet_viterbi(
        frames,
        config=config,
        transition_cost=transition,
    )
    fixed_lag = solve_fixed_lag_tracklet_viterbi(
        frames,
        lag_s=0.1,
        config=config,
        transition_cost=transition,
    )

    assert full.path == [first, None, recovered]
    assert fixed_lag.path == full.path
    assert fixed_lag.missed_detection_count == 1
    assert fixed_lag.total_cost == full.total_cost == 1.0
