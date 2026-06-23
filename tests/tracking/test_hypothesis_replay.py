from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    HypothesisReplay,
    InnovationConsistencyScoreConfig,
    TrackingRecord,
    rank_hypothesis_replays,
    rank_replayed_hypotheses,
    scores_to_dicts,
)


def test_rank_hypothesis_replays_prefers_innovation_consistency_over_graph_cost() -> (
    None
):
    bad_graph_winner = HypothesisReplay(
        hypothesis_id="graph-rank-1",
        records=[{"nis": 200.0, "residual_norm_m": 1000.0, "action": "updated"}],
        graph_cost=0.0,
        coverage_count=100,
    )
    good_replay = HypothesisReplay(
        hypothesis_id="graph-rank-2",
        records=[{"nis": 1.0, "residual_norm_m": 10.0, "action": "updated"}],
        graph_cost=10.0,
        coverage_count=100,
    )

    scores = rank_hypothesis_replays(
        [bad_graph_winner, good_replay],
        config=InnovationConsistencyScoreConfig(
            graph_cost_weight=0.1,
            nis_weight=1.0,
            residual_weight=0.1,
            nis_clip=50.0,
            residual_clip=500.0,
        ),
    )

    assert [score.hypothesis_id for score in scores] == ["graph-rank-2", "graph-rank-1"]
    assert scores[0].rank == 1
    assert scores_to_dicts(scores)[0]["hypothesis_id"] == "graph-rank-2"


def test_hypothesis_replay_accepts_tracking_record_objects() -> None:
    record = TrackingRecord(
        time=1.0,
        source="radar",
        action="update",
        prior_mean=np.zeros(2),
        prior_cov=np.eye(2),
        posterior_mean=np.ones(2),
        posterior_cov=np.eye(2),
        innovation=np.array([3.0, 4.0]),
        innovation_cov=np.eye(2),
        nis=25.0,
        accepted=True,
    )
    replay = HypothesisReplay(hypothesis_id=1, records=[record], graph_cost=0.0)

    score = rank_hypothesis_replays([replay])[0]

    assert score.finite_nis_count == 1
    assert score.finite_residual_count == 1
    assert score.robust_sum_residual == 5.0 / 100.0


def test_score_config_rejects_invalid_scalar_controls() -> None:
    invalid_cases = (
        ("residual_weight", np.inf, "residual_weight must be finite"),
        ("coverage_reward", True, "coverage_reward must be finite"),
        ("nis_clip", -1.0, "nis_clip must be nonnegative"),
        ("residual_clip", np.array([1.0]), "residual_clip must be finite"),
        ("residual_normalizer", 0.0, "residual_normalizer must be positive"),
    )

    for field_name, value, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            InnovationConsistencyScoreConfig(**{field_name: value})


def test_hypothesis_replay_rejects_fractional_count_fields() -> None:
    with pytest.raises(
        ValueError,
        match="track_switches must be a nonnegative integer",
    ):
        HypothesisReplay(hypothesis_id="bad-count", records=[], track_switches=1.5)


def test_rank_replayed_hypotheses_calls_replay_function() -> None:
    def replay_fn(value: int) -> HypothesisReplay:
        return HypothesisReplay(
            hypothesis_id=value,
            records=[{"nis": float(value), "action": "updated"}],
            graph_cost=0.0,
        )

    scores = rank_replayed_hypotheses([3, 1, 2], replay_fn)

    assert [score.hypothesis_id for score in scores] == [1, 2, 3]
