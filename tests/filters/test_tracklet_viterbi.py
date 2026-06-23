import numpy as np
import pytest
from pyrecest.filters.tracklet_viterbi import (
    TrackletAssociationCandidate,
    TrackletViterbiConfig,
    prefix_track_support,
    retain_top_and_track_representatives,
    solve_fixed_lag_tracklet_viterbi,
    solve_tracklet_viterbi,
    track_support_cost,
)


def test_config_validates_candidate_limits():
    for name in ("max_candidates_per_frame", "max_candidate_pool_per_frame"):
        for value in (0, 1.5, np.nan, True, np.array([1])):
            with pytest.raises(ValueError, match=name):
                TrackletViterbiConfig(**{name: value})

    for value in (-1, 1.5, np.nan, True, np.array([1])):
        with pytest.raises(ValueError, match="max_candidates_per_track_id"):
            TrackletViterbiConfig(max_candidates_per_track_id=value)


def test_config_normalizes_integer_like_candidate_limits():
    config = TrackletViterbiConfig(
        max_candidates_per_frame=np.array(2.0),
        max_candidate_pool_per_frame=3.0,
        max_candidates_per_track_id=np.int64(1),
    )

    assert config.max_candidates_per_frame == 2
    assert config.max_candidate_pool_per_frame == 3
    assert config.max_candidates_per_track_id == 1


def test_config_rejects_nonfinite_float_parameters():
    invalid_cases = {
        "missed_detection_cost": np.nan,
        "max_speed_penalty": np.inf,
        "transition_position_std": np.nan,
        "transition_velocity_std": np.inf,
        "max_speed": np.nan,
    }
    for name, value in invalid_cases.items():
        with pytest.raises(ValueError, match=name):
            TrackletViterbiConfig(**{name: value})


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


def test_fixed_lag_solver_rejects_invalid_lag():
    frames = [[TrackletAssociationCandidate("a0")]]

    for lag_s in (0.0, -1.0, np.nan, np.inf, True, np.array([1.0])):
        with pytest.raises(ValueError, match="lag_s"):
            solve_fixed_lag_tracklet_viterbi(frames, lag_s=lag_s)


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


def test_fixed_lag_total_cost_scores_committed_path_once():
    frames = [
        [TrackletAssociationCandidate("a0", unary_cost=1.0, time_s=0.0)],
        [TrackletAssociationCandidate("a1", unary_cost=2.0, time_s=1.0)],
    ]
    result = solve_fixed_lag_tracklet_viterbi(
        frames,
        lag_s=10.0,
        config=TrackletViterbiConfig(missed_detection_cost=100.0),
    )

    assert [candidate.candidate_id for candidate in result.path] == ["a0", "a1"]
    assert result.total_cost == 3.0


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


def _miss_penalty_config() -> TrackletViterbiConfig:
    return TrackletViterbiConfig(
        missed_detection_cost=2.0,
        consecutive_miss_cost=5.0,
    )


def test_fixed_lag_leading_consecutive_misses_match_full_viterbi_cost():
    frames = [[], []]
    config = _miss_penalty_config()

    full = solve_tracklet_viterbi(frames, config=config)
    fixed_lag = solve_fixed_lag_tracklet_viterbi(
        frames,
        lag_s=0.1,
        config=config,
    )

    assert full.path == [None, None]
    assert fixed_lag.path == full.path
    assert full.total_cost == 9.0
    assert fixed_lag.total_cost == full.total_cost


def test_fixed_lag_consecutive_miss_after_detection_uses_committed_streak():
    frames = [
        [TrackletAssociationCandidate("d0", unary_cost=0.0, track_id="track")],
        [],
        [],
    ]
    config = _miss_penalty_config()

    fixed_lag = solve_fixed_lag_tracklet_viterbi(
        frames,
        lag_s=0.1,
        config=config,
    )

    assert fixed_lag.path == [frames[0][0], None, None]
    assert fixed_lag.total_cost == 9.0
