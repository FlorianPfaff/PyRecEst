from pyrecest.tracking import HypothesisReplay, rank_hypothesis_replays


def test_negative_replay_diagnostics_cannot_reduce_hypothesis_score() -> None:
    replay = HypothesisReplay(
        hypothesis_id="malformed-diagnostics",
        records=[
            {
                "nis": -100.0,
                "residual_norm_m": -1000.0,
                "action": "updated",
            }
        ],
    )

    score = rank_hypothesis_replays([replay])[0]

    assert score.robust_sum_nis == 0.0
    assert score.robust_sum_residual == 0.0
    assert score.finite_nis_count == 0
    assert score.finite_residual_count == 0
    assert score.total_score == 0.0
