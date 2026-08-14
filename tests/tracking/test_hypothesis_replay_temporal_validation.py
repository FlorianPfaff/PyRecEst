from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    HypothesisReplay,
    InnovationConsistencyScoreConfig,
    rank_hypothesis_replays,
)

_TEMPORAL_VALUES = (
    np.timedelta64(2, "ns"),
    np.datetime64("1970-01-01T00:00:00.000000002"),
    np.asarray(np.timedelta64(3, "ns"), dtype=object),
)


@pytest.mark.parametrize("value", _TEMPORAL_VALUES)
def test_score_config_rejects_temporal_scalar_controls(value: object) -> None:
    for field_name in ("nis_weight", "nis_clip", "residual_normalizer"):
        with pytest.raises(ValueError):
            InnovationConsistencyScoreConfig(**{field_name: value})


@pytest.mark.parametrize("value", _TEMPORAL_VALUES)
def test_hypothesis_replay_rejects_temporal_scalar_fields(value: object) -> None:
    for field_name in (
        "graph_cost",
        "tail_duration_s",
        "track_switches",
        "coverage_count",
    ):
        with pytest.raises(ValueError):
            HypothesisReplay(
                hypothesis_id="invalid-temporal-field",
                records=[],
                **{field_name: value},
            )


def test_temporal_record_statistics_are_ignored() -> None:
    replay = HypothesisReplay(
        hypothesis_id="temporal-records",
        records=[
            {
                "nis": np.timedelta64(4, "ns"),
                "residual_norm_m": np.datetime64("1970-01-01T00:00:00.000000005"),
            },
            {
                "nis": np.asarray(np.timedelta64(6, "ns"), dtype=object),
                "innovation": np.array([np.timedelta64(3, "ns")]),
            },
        ],
    )

    score = rank_hypothesis_replays([replay])[0]

    assert score.finite_nis_count == 0
    assert score.finite_residual_count == 0
    assert score.robust_sum_nis == 0.0
    assert score.robust_sum_residual == 0.0


def test_numeric_numpy_scalars_remain_supported() -> None:
    config = InnovationConsistencyScoreConfig(
        nis_weight=np.float64(2.5),
        nis_clip=np.int64(7),
    )
    replay = HypothesisReplay(
        hypothesis_id="valid-numeric-scalars",
        records=[],
        track_switches=np.int64(2),
    )

    assert config.nis_weight == 2.5
    assert config.nis_clip == 7.0
    assert replay.track_switches == 2
