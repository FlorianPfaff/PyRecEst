"""Bounded residual multi-hypothesis selection for discrete edit candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class ResidualEditCandidate:
    """One discrete residual edit candidate."""

    candidate_id: str
    score: float
    conflict_keys: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualHypothesis:
    """A bounded set of residual edit candidates."""

    candidate_ids: tuple[str, ...]
    score: float
    candidate_scores: tuple[float, ...]

    @property
    def n_edits(self) -> int:
        """Return the number of edits in this hypothesis."""

        return len(self.candidate_ids)


@dataclass(frozen=True)
class ResidualMHTConfig:
    """Configuration for bounded residual-MHT enumeration."""

    max_edits: int = 2
    max_hypotheses: int = 16
    edit_penalty: float = 0.25
    score_threshold: float = 1.0
    include_empty: bool = True


def enumerate_residual_hypotheses(
    candidates: Iterable[ResidualEditCandidate],
    *,
    config: ResidualMHTConfig | None = None,
) -> tuple[ResidualHypothesis, ...]:
    """Enumerate compatible bounded residual hypotheses."""

    cfg = config or ResidualMHTConfig()
    ordered = sorted(
        tuple(candidates),
        key=lambda candidate: (-float(candidate.score), str(candidate.candidate_id)),
    )
    max_edits = max(0, int(cfg.max_edits))
    max_hypotheses = max(1, int(cfg.max_hypotheses))
    hypotheses: list[ResidualHypothesis] = []
    if cfg.include_empty:
        hypotheses.append(ResidualHypothesis((), 0.0, ()))

    for size in range(1, min(max_edits, len(ordered)) + 1):
        for group in combinations(ordered, size):
            if not _compatible(group):
                continue
            candidate_scores = tuple(float(candidate.score) for candidate in group)
            score = sum(candidate_scores) - float(cfg.edit_penalty) * float(size)
            hypotheses.append(
                ResidualHypothesis(
                    candidate_ids=tuple(
                        str(candidate.candidate_id) for candidate in group
                    ),
                    score=float(score),
                    candidate_scores=candidate_scores,
                )
            )

    hypotheses.sort(
        key=lambda hypothesis: (
            -float(hypothesis.score),
            int(hypothesis.n_edits),
            tuple(hypothesis.candidate_ids),
        )
    )
    return tuple(hypotheses[:max_hypotheses])


def select_residual_hypothesis(
    candidates: Iterable[ResidualEditCandidate],
    *,
    config: ResidualMHTConfig | None = None,
) -> ResidualHypothesis:
    """Return the best residual hypothesis above threshold, else no-edit."""

    cfg = config or ResidualMHTConfig()
    hypotheses = enumerate_residual_hypotheses(candidates, config=cfg)
    empty = ResidualHypothesis((), 0.0, ())
    if not hypotheses:
        return empty
    best = hypotheses[0]
    if best.n_edits <= 0:
        return best
    if float(best.score) < float(cfg.score_threshold):
        return empty
    return best


def hypothesis_to_dict(hypothesis: ResidualHypothesis) -> dict[str, Any]:
    """Return a JSON/CSV-friendly hypothesis record."""

    return {
        "candidate_ids": ";".join(hypothesis.candidate_ids),
        "n_edits": int(hypothesis.n_edits),
        "score": float(hypothesis.score),
        "candidate_scores": ";".join(
            f"{score:.12g}" for score in hypothesis.candidate_scores
        ),
    }


def hypotheses_to_dicts(
    hypotheses: Sequence[ResidualHypothesis],
) -> tuple[dict[str, Any], ...]:
    """Serialize hypotheses for diagnostics."""

    return tuple(hypothesis_to_dict(hypothesis) for hypothesis in hypotheses)


def _compatible(candidates: Sequence[ResidualEditCandidate]) -> bool:
    seen: set[str] = set()
    for candidate in candidates:
        overlap = seen.intersection(set(candidate.conflict_keys))
        if overlap:
            return False
        seen.update(candidate.conflict_keys)
    return True
