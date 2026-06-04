"""Innovation-consistency ranking for replayed association hypotheses.

The functions in this module are filter-independent.  They score candidate
association hypotheses after a caller has replayed each hypothesis through a
tracker and produced ordinary record dictionaries or :class:`TrackingRecord`
instances.  This lets multi-sensor trackers rank top-k association/path
hypotheses by innovation consistency without using ground truth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np

try:  # Optional only so this module can be used without event_records imports at runtime.
    from .event_records import TrackingRecord
except Exception:  # pragma: no cover - defensive for partial downstream copies
    TrackingRecord = object  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class InnovationConsistencyScoreConfig:
    """Weights and clipping thresholds for replay-hypothesis ranking."""

    graph_cost_weight: float = 1.0
    nis_weight: float = 1.0
    residual_weight: float = 0.01
    switch_weight: float = 2.0
    missed_detection_weight: float = 1.0
    rejected_weight: float = 0.25
    coast_weight: float = 0.25
    unsupported_measurement_weight: float = 5.0
    hard_quarantine_weight: float = 1000.0
    tail_duration_weight: float = 0.05
    coverage_reward: float = 0.001
    nis_clip: float = 50.0
    residual_clip: float = 500.0
    residual_normalizer: float = 100.0


@dataclass(frozen=True)
class HypothesisReplay:
    """One candidate hypothesis and its already-computed replay records."""

    hypothesis_id: Hashable
    records: Sequence[Any]
    graph_cost: float = 0.0
    track_switches: int = 0
    missed_detection_count: int = 0
    rejected_count: int = 0
    coast_count: int = 0
    unsupported_measurement_count: int = 0
    hard_quarantine_count: int = 0
    tail_duration_s: float = 0.0
    coverage_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self, "graph_cost", _finite_float(self.graph_cost, "graph_cost")
        )
        object.__setattr__(
            self,
            "track_switches",
            _nonnegative_int(self.track_switches, "track_switches"),
        )
        object.__setattr__(
            self,
            "missed_detection_count",
            _nonnegative_int(self.missed_detection_count, "missed_detection_count"),
        )
        object.__setattr__(
            self,
            "rejected_count",
            _nonnegative_int(self.rejected_count, "rejected_count"),
        )
        object.__setattr__(
            self, "coast_count", _nonnegative_int(self.coast_count, "coast_count")
        )
        object.__setattr__(
            self,
            "unsupported_measurement_count",
            _nonnegative_int(
                self.unsupported_measurement_count, "unsupported_measurement_count"
            ),
        )
        object.__setattr__(
            self,
            "hard_quarantine_count",
            _nonnegative_int(self.hard_quarantine_count, "hard_quarantine_count"),
        )
        object.__setattr__(
            self,
            "tail_duration_s",
            _nonnegative_float(self.tail_duration_s, "tail_duration_s"),
        )
        object.__setattr__(
            self,
            "coverage_count",
            _nonnegative_int(self.coverage_count, "coverage_count"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class HypothesisReplayScore:
    """Score and diagnostics for one replayed hypothesis."""

    hypothesis_id: Hashable
    total_score: float
    graph_cost: float
    robust_sum_nis: float
    robust_sum_residual: float
    finite_nis_count: int
    finite_residual_count: int
    track_switches: int
    missed_detection_count: int
    rejected_count: int
    coast_count: int
    unsupported_measurement_count: int
    hard_quarantine_count: int
    tail_duration_s: float
    coverage_count: int
    rank: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_rank(self, rank: int) -> "HypothesisReplayScore":
        """Return a copy with ``rank`` set."""

        return HypothesisReplayScore(
            hypothesis_id=self.hypothesis_id,
            total_score=self.total_score,
            graph_cost=self.graph_cost,
            robust_sum_nis=self.robust_sum_nis,
            robust_sum_residual=self.robust_sum_residual,
            finite_nis_count=self.finite_nis_count,
            finite_residual_count=self.finite_residual_count,
            track_switches=self.track_switches,
            missed_detection_count=self.missed_detection_count,
            rejected_count=self.rejected_count,
            coast_count=self.coast_count,
            unsupported_measurement_count=self.unsupported_measurement_count,
            hard_quarantine_count=self.hard_quarantine_count,
            tail_duration_s=self.tail_duration_s,
            coverage_count=self.coverage_count,
            rank=int(rank),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly dictionary."""

        return {
            "rank": self.rank,
            "hypothesis_id": self.hypothesis_id,
            "total_score": self.total_score,
            "graph_cost": self.graph_cost,
            "robust_sum_nis": self.robust_sum_nis,
            "robust_sum_residual": self.robust_sum_residual,
            "finite_nis_count": self.finite_nis_count,
            "finite_residual_count": self.finite_residual_count,
            "track_switches": self.track_switches,
            "missed_detection_count": self.missed_detection_count,
            "rejected_count": self.rejected_count,
            "coast_count": self.coast_count,
            "unsupported_measurement_count": self.unsupported_measurement_count,
            "hard_quarantine_count": self.hard_quarantine_count,
            "tail_duration_s": self.tail_duration_s,
            "coverage_count": self.coverage_count,
            **{
                f"metadata_{key}": _jsonable(value)
                for key, value in dict(self.metadata).items()
            },
        }


def score_hypothesis_replay(
    replay: HypothesisReplay,
    config: InnovationConsistencyScoreConfig | None = None,
) -> HypothesisReplayScore:
    """Score one replayed hypothesis without using truth information."""

    cfg = InnovationConsistencyScoreConfig() if config is None else config
    nis_values = _finite_record_values(replay.records, ("nis",))
    residual_values = _finite_record_values(
        replay.records,
        ("residual_norm_m", "residual_norm", "innovation_norm", "innovation_norm_m"),
        fallback_norm_keys=("innovation", "residual"),
    )
    robust_sum_nis = (
        float(np.sum(np.minimum(nis_values, float(cfg.nis_clip))))
        if nis_values.size
        else 0.0
    )
    robust_sum_residual = (
        float(np.sum(np.minimum(residual_values, float(cfg.residual_clip))))
        / float(cfg.residual_normalizer)
        if residual_values.size
        else 0.0
    )
    missed = int(
        replay.missed_detection_count
        + _count_record_actions(replay.records, {"missed_detection"})
    )
    rejected = int(
        replay.rejected_count
        + _count_record_actions(
            replay.records, {"rejected", "residual_rejected", "safety_rejected"}
        )
    )
    coast = int(
        replay.coast_count + _count_record_actions(replay.records, {"coast", "predict"})
    )
    total = (
        cfg.graph_cost_weight * replay.graph_cost
        + cfg.nis_weight * robust_sum_nis
        + cfg.residual_weight * robust_sum_residual
        + cfg.switch_weight * replay.track_switches
        + cfg.missed_detection_weight * missed
        + cfg.rejected_weight * rejected
        + cfg.coast_weight * coast
        + cfg.unsupported_measurement_weight * replay.unsupported_measurement_count
        + cfg.hard_quarantine_weight * replay.hard_quarantine_count
        + cfg.tail_duration_weight * replay.tail_duration_s
        - cfg.coverage_reward * replay.coverage_count
    )
    return HypothesisReplayScore(
        hypothesis_id=replay.hypothesis_id,
        total_score=float(total),
        graph_cost=float(replay.graph_cost),
        robust_sum_nis=float(robust_sum_nis),
        robust_sum_residual=float(robust_sum_residual),
        finite_nis_count=int(nis_values.size),
        finite_residual_count=int(residual_values.size),
        track_switches=int(replay.track_switches),
        missed_detection_count=missed,
        rejected_count=rejected,
        coast_count=coast,
        unsupported_measurement_count=int(replay.unsupported_measurement_count),
        hard_quarantine_count=int(replay.hard_quarantine_count),
        tail_duration_s=float(replay.tail_duration_s),
        coverage_count=int(replay.coverage_count),
        metadata=dict(replay.metadata),
    )


def rank_hypothesis_replays(
    replays: Iterable[HypothesisReplay],
    config: InnovationConsistencyScoreConfig | None = None,
) -> list[HypothesisReplayScore]:
    """Return replay scores sorted from best to worst."""

    scores = [score_hypothesis_replay(replay, config=config) for replay in replays]
    scores.sort(key=lambda item: (item.total_score, str(item.hypothesis_id)))
    return [score.with_rank(rank) for rank, score in enumerate(scores, start=1)]


def rank_replayed_hypotheses(
    hypotheses: Iterable[Any],
    replay_fn: Callable[[Any], HypothesisReplay],
    config: InnovationConsistencyScoreConfig | None = None,
) -> list[HypothesisReplayScore]:
    """Replay hypotheses with ``replay_fn`` and rank the resulting replays."""

    return rank_hypothesis_replays(
        (replay_fn(hypothesis) for hypothesis in hypotheses), config=config
    )


def scores_to_dicts(scores: Iterable[HypothesisReplayScore]) -> list[dict[str, Any]]:
    """Return ranked score dictionaries."""

    return [score.to_dict() for score in scores]


def _finite_record_values(
    records: Sequence[Any],
    keys: tuple[str, ...],
    *,
    fallback_norm_keys: tuple[str, ...] = (),
) -> np.ndarray:
    values: list[float] = []
    for record in records:
        value = _record_value(record, keys)
        if value is None and fallback_norm_keys:
            vector = _record_value(record, fallback_norm_keys)
            if vector is not None:
                try:
                    value = float(
                        np.linalg.norm(np.asarray(vector, dtype=float).reshape(-1))
                    )
                except (TypeError, ValueError):
                    value = None
        if value is None:
            continue
        try:
            parsed = float(np.asarray(value, dtype=float))
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(parsed):
            values.append(parsed)
    return np.asarray(values, dtype=float)


def _count_record_actions(records: Sequence[Any], actions: set[str]) -> int:
    count = 0
    for record in records:
        action = _record_value(record, ("action", "update_action"))
        if action is not None and str(action) in actions:
            count += 1
    return count


def _record_value(record: Any, keys: tuple[str, ...]) -> Any | None:
    if isinstance(record, Mapping):
        for key in keys:
            if key in record:
                return record[key]
        return None
    for key in keys:
        if hasattr(record, key):
            return getattr(record, key)
    return None


def _finite_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _finite_float(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "HypothesisReplay",
    "HypothesisReplayScore",
    "InnovationConsistencyScoreConfig",
    "rank_hypothesis_replays",
    "rank_replayed_hypotheses",
    "score_hypothesis_replay",
    "scores_to_dicts",
]
