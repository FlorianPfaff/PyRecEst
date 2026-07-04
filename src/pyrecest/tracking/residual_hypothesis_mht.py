"""Bounded residual multi-hypothesis selection for discrete edit candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any, Literal

ResidualMHTPreset = Literal["conservative", "frontier", "multi_family"]


@dataclass(frozen=True)
class ResidualEditCandidate:
    """One discrete residual edit candidate.

    ``family`` is an optional generic grouping label, such as ``"split"``,
    ``"merge"``, or ``"terminal_veto"``.  It has no intrinsic meaning to
    PyRecEst, but :class:`ResidualMHTConfig` can cap the number of edits selected
    from each family.

    ``group_keys`` are softer grouping labels than ``conflict_keys``.  Conflicts
    are hard mutual exclusions.  Group keys can be capped with
    ``max_edits_per_group_key`` while still allowing more than one candidate from
    the same group when the cap permits it.
    """

    candidate_id: str
    score: float
    conflict_keys: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    family: str | None = None
    group_keys: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ResidualHypothesis:
    """A bounded set of residual edit candidates."""

    candidate_ids: tuple[str, ...]
    score: float
    candidate_scores: tuple[float, ...]
    candidate_families: tuple[str, ...] = ()
    candidate_group_keys: tuple[str, ...] = ()

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
    max_edits_per_family: Mapping[str, int] = field(default_factory=dict)
    max_edits_per_group_key: int | None = None
    candidate_top_k: int | None = None
    min_candidate_score: float | None = None


def residual_mht_config_for_preset(
    preset: ResidualMHTPreset,
    *,
    max_edits: int | None = None,
    max_hypotheses: int | None = None,
    score_threshold: float | None = None,
) -> ResidualMHTConfig:
    """Return a small generic preset for residual-MHT selection.

    The presets are deliberately domain-agnostic.  They encode selector breadth,
    not the meaning of candidates.
    """

    if preset == "conservative":
        config = ResidualMHTConfig(
            max_edits=1,
            max_hypotheses=8,
            edit_penalty=0.25,
            score_threshold=1.0,
            candidate_top_k=4,
            max_edits_per_group_key=1,
        )
    elif preset == "frontier":
        config = ResidualMHTConfig(
            max_edits=2,
            max_hypotheses=32,
            edit_penalty=0.40,
            score_threshold=1.4,
            candidate_top_k=8,
            max_edits_per_group_key=1,
        )
    elif preset == "multi_family":
        config = ResidualMHTConfig(
            max_edits=3,
            max_hypotheses=64,
            edit_penalty=0.25,
            score_threshold=1.0,
            candidate_top_k=12,
            max_edits_per_group_key=1,
        )
    else:
        raise ValueError(f"Unknown residual-MHT preset: {preset!r}")
    if max_edits is not None:
        config = replace(config, max_edits=int(max_edits))
    if max_hypotheses is not None:
        config = replace(config, max_hypotheses=int(max_hypotheses))
    if score_threshold is not None:
        config = replace(config, score_threshold=float(score_threshold))
    return config


def enumerate_residual_hypotheses(
    candidates: Iterable[ResidualEditCandidate],
    *,
    config: ResidualMHTConfig | None = None,
) -> tuple[ResidualHypothesis, ...]:
    """Enumerate compatible bounded residual hypotheses."""

    cfg = config or ResidualMHTConfig()
    ordered = _candidate_frontier(candidates, cfg)
    max_edits = max(0, int(cfg.max_edits))
    max_hypotheses = max(1, int(cfg.max_hypotheses))
    hypotheses: list[ResidualHypothesis] = []
    if cfg.include_empty:
        hypotheses.append(ResidualHypothesis((), 0.0, (), (), ()))

    for size in range(1, min(max_edits, len(ordered)) + 1):
        for group in combinations(ordered, size):
            if not _compatible(
                group,
                max_edits_per_family=cfg.max_edits_per_family,
                max_edits_per_group_key=cfg.max_edits_per_group_key,
            ):
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
                    candidate_families=tuple(
                        "" if candidate.family is None else str(candidate.family)
                        for candidate in group
                    ),
                    candidate_group_keys=tuple(
                        ";".join(sorted(str(key) for key in candidate.group_keys))
                        for candidate in group
                    ),
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
    empty = ResidualHypothesis((), 0.0, (), (), ())
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
        "candidate_families": ";".join(hypothesis.candidate_families),
        "candidate_group_keys": "|".join(hypothesis.candidate_group_keys),
    }


def hypotheses_to_dicts(
    hypotheses: Sequence[ResidualHypothesis],
) -> tuple[dict[str, Any], ...]:
    """Serialize hypotheses for diagnostics."""

    return tuple(hypothesis_to_dict(hypothesis) for hypothesis in hypotheses)


def _candidate_frontier(
    candidates: Iterable[ResidualEditCandidate],
    config: ResidualMHTConfig,
) -> tuple[ResidualEditCandidate, ...]:
    filtered = [
        candidate
        for candidate in candidates
        if config.min_candidate_score is None
        or float(candidate.score) >= float(config.min_candidate_score)
    ]
    filtered.sort(
        key=lambda candidate: (-float(candidate.score), str(candidate.candidate_id))
    )
    if config.candidate_top_k is not None:
        return tuple(filtered[: max(0, int(config.candidate_top_k))])
    return tuple(filtered)


def _compatible(
    candidates: Sequence[ResidualEditCandidate],
    *,
    max_edits_per_family: Mapping[str, int],
    max_edits_per_group_key: int | None,
) -> bool:
    seen: set[str] = set()
    family_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for candidate in candidates:
        overlap = seen.intersection(set(candidate.conflict_keys))
        if overlap:
            return False
        seen.update(candidate.conflict_keys)
        if candidate.family is not None:
            family = str(candidate.family)
            family_counts[family] = family_counts.get(family, 0) + 1
            family_cap = max_edits_per_family.get(family)
            if family_cap is not None and family_counts[family] > int(family_cap):
                return False
        if max_edits_per_group_key is not None:
            for group_key in candidate.group_keys:
                group = str(group_key)
                group_counts[group] = group_counts.get(group, 0) + 1
                if group_counts[group] > int(max_edits_per_group_key):
                    return False
    return True
