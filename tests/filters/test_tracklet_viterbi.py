from pyrecest.filters.tracklet_viterbi import (
    TrackletAssociationCandidate,
    TrackletViterbiConfig,
    prefix_track_support,
    retain_top_and_track_representatives,
    solve_fixed_lag_tracklet_viterbi,
    solve_tracklet_viterbi,
    track_support_cost,
)


def test_switch_cost_prefers_coherent_tracklet():
    frames = [
        [TrackletAssociationCandidate("a0", unary_cost=0.0, track_id="A")],
        [
            TrackletAssociationCandidate("b1", unary_cost=0.0, track_id="B"),
            TrackletAssociationCandidate("a1", unary_cost=2.0, track_id="A"),
        ],
    ]
    result = solve_tracklet_viterbi(
        frames,
        config=TrackletViterbiConfig(switch_cost=10.0, missed_detection_cost=100.0),
    )
    assert [candidate.candidate_id for candidate in result.path] == ["a0", "a1"]


def test_missed_detection_branch_is_selected_when_candidates_are_expensive():
    result = solve_tracklet_viterbi(
        [[TrackletAssociationCandidate("bad", unary_cost=100.0)]],
        config=TrackletViterbiConfig(missed_detection_cost=1.0),
    )
    assert result.path == [None]
    assert result.missed_detection_count == 1


def test_fixed_lag_solver_uses_prefix_memory():
    frames = [
        [TrackletAssociationCandidate("a0", unary_cost=0.0, track_id="A", time_s=0.0)],
        [
            TrackletAssociationCandidate(
                "b1", unary_cost=0.0, track_id="B", time_s=1.0
            ),
            TrackletAssociationCandidate(
                "a1", unary_cost=2.0, track_id="A", time_s=1.0
            ),
        ],
    ]
    result = solve_fixed_lag_tracklet_viterbi(
        frames,
        lag_s=0.1,
        config=TrackletViterbiConfig(switch_cost=10.0, missed_detection_cost=100.0),
    )
    assert [candidate.candidate_id for candidate in result.path] == ["a0", "a1"]


def test_retention_keeps_track_representative_outside_top_k():
    candidates = [
        TrackletAssociationCandidate("best", unary_cost=0.0, track_id="A"),
        TrackletAssociationCandidate("second", unary_cost=1.0, track_id="A"),
        TrackletAssociationCandidate("track-b", unary_cost=5.0, track_id="B"),
    ]
    kept = retain_top_and_track_representatives(
        candidates,
        config=TrackletViterbiConfig(
            max_candidates_per_frame=1, max_candidate_pool_per_frame=3
        ),
    )
    assert {candidate.candidate_id for candidate in kept} == {"best", "track-b"}


def test_prefix_track_support_yields_bounded_reward():
    frames = [
        [TrackletAssociationCandidate("a0", track_id="A", time_s=0.0)],
        [TrackletAssociationCandidate("a1", track_id="A", time_s=1.0)],
    ]
    support = prefix_track_support(frames)
    candidate = TrackletAssociationCandidate("a1", track_id="A", time_s=1.0)
    assert track_support_cost(candidate, support[1]) < 0.0
