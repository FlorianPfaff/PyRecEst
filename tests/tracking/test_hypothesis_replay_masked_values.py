from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    HypothesisReplay,
    InnovationConsistencyScoreConfig,
    rank_hypothesis_replays,
)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        (
            "nis_weight",
            np.ma.array(7.0, mask=True),
            "nis_weight must be finite",
        ),
        (
            "residual_normalizer",
            np.ma.masked,
            "residual_normalizer must be finite",
        ),
    ),
)
def test_score_config_rejects_masked_scalar_controls(
    field_name: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        InnovationConsistencyScoreConfig(**{field_name: value})


@pytest.mark.parametrize(
    "value",
    (np.ma.array(3, mask=True), np.ma.masked),
)
def test_hypothesis_replay_rejects_masked_count_controls(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="track_switches must be a nonnegative integer",
    ):
        HypothesisReplay(
            hypothesis_id="masked-count",
            records=[],
            track_switches=value,
        )


def test_masked_record_diagnostics_do_not_affect_scores() -> None:
    replay = HypothesisReplay(
        hypothesis_id="masked-diagnostics",
        records=[
            {
                "nis": np.ma.array(999.0, mask=True),
                "innovation": np.ma.array(
                    [3.0, 400.0],
                    mask=[False, True],
                ),
            }
        ],
    )

    score = rank_hypothesis_replays([replay])[0]

    assert score.finite_nis_count == 0
    assert score.finite_residual_count == 0
    assert score.robust_sum_nis == 0.0
    assert score.robust_sum_residual == 0.0


def test_nested_masked_fallback_statistic_is_ignored() -> None:
    replay = HypothesisReplay(
        hypothesis_id="nested-masked-diagnostic",
        records=[{"innovation": [3.0, np.ma.masked]}],
    )

    score = rank_hypothesis_replays([replay])[0]

    assert score.finite_residual_count == 0
    assert score.robust_sum_residual == 0.0


def test_clear_mask_wrappers_remain_supported() -> None:
    config = InnovationConsistencyScoreConfig(
        nis_weight=np.ma.array(2.0, mask=False),
    )
    replay = HypothesisReplay(
        hypothesis_id="clear-mask",
        records=[
            {
                "nis": np.ma.array(4.0, mask=False),
                "innovation": np.ma.array([3.0, 4.0], mask=False),
            }
        ],
        track_switches=np.ma.array(1, mask=False),
    )

    score = rank_hypothesis_replays([replay], config=config)[0]

    assert config.nis_weight == pytest.approx(2.0)
    assert replay.track_switches == 1
    assert score.finite_nis_count == 1
    assert score.finite_residual_count == 1
    assert score.robust_sum_nis == pytest.approx(4.0)
    assert score.robust_sum_residual == pytest.approx(5.0 / 100.0)
