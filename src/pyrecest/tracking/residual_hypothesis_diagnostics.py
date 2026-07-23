"""Diagnostic serialization helpers for residual hypothesis selection.

The helpers in this module intentionally know nothing about any application
specific tracking domain.  Domain-specific information can be stored in
``ResidualEditCandidate.metadata`` and is flattened with a configurable prefix
for CSV/JSON ledgers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .residual_hypothesis_mht import ResidualEditCandidate, ResidualHypothesis


def candidate_to_dict(
    candidate: ResidualEditCandidate,
    *,
    rank: int | None = None,
    selected: bool = False,
    applied: bool = False,
    include_metadata: bool = True,
    metadata_prefix: str = "metadata_",
) -> dict[str, Any]:
    """Return a CSV/JSON-friendly residual-edit candidate record."""

    record: dict[str, Any] = {
        "candidate_id": str(candidate.candidate_id),
        "score": float(candidate.score),
        "family": "" if candidate.family is None else str(candidate.family),
        "conflict_keys": ";".join(sorted(str(key) for key in candidate.conflict_keys)),
        "selected": int(bool(selected)),
        "applied": int(bool(applied)),
    }
    if rank is not None:
        record["rank"] = int(rank)
    if include_metadata:
        for key, value in candidate.metadata.items():
            record[f"{metadata_prefix}{key}"] = value
    return record


def candidates_to_dicts(
    candidates: Iterable[ResidualEditCandidate],
    *,
    selected_ids: Iterable[str] | None = None,
    applied_ids: Iterable[str] | None = None,
    include_metadata: bool = True,
    metadata_prefix: str = "metadata_",
) -> tuple[dict[str, Any], ...]:
    """Serialize residual edit candidates with selected/applied flags."""

    selected = {str(candidate_id) for candidate_id in selected_ids or ()}
    applied = {str(candidate_id) for candidate_id in applied_ids or ()}
    ordered = sorted(
        tuple(candidates),
        key=lambda candidate: (-float(candidate.score), str(candidate.candidate_id)),
    )
    return tuple(
        candidate_to_dict(
            candidate,
            rank=index,
            selected=str(candidate.candidate_id) in selected,
            applied=str(candidate.candidate_id) in applied,
            include_metadata=include_metadata,
            metadata_prefix=metadata_prefix,
        )
        for index, candidate in enumerate(ordered, start=1)
    )


def hypothesis_diagnostic_to_dict(
    hypothesis: ResidualHypothesis,
    *,
    rank: int | None = None,
    selected: bool = False,
) -> dict[str, Any]:
    """Return a CSV/JSON-friendly residual hypothesis record."""

    record: dict[str, Any] = {
        "candidate_ids": ";".join(hypothesis.candidate_ids),
        "n_edits": int(hypothesis.n_edits),
        "score": float(hypothesis.score),
        "candidate_scores": ";".join(
            f"{score:.12g}" for score in hypothesis.candidate_scores
        ),
        "candidate_families": ";".join(hypothesis.candidate_families),
        "selected": int(bool(selected)),
    }
    if rank is not None:
        record["rank"] = int(rank)
    return record


def hypotheses_to_diagnostic_dicts(
    hypotheses: Sequence[ResidualHypothesis],
    *,
    selected_hypothesis: ResidualHypothesis | None = None,
) -> tuple[dict[str, Any], ...]:
    """Serialize residual hypotheses and mark the selected hypothesis."""

    selected_ids = None
    if selected_hypothesis is not None:
        selected_ids = tuple(selected_hypothesis.candidate_ids)
    return tuple(
        hypothesis_diagnostic_to_dict(
            hypothesis,
            rank=index,
            selected=(
                selected_ids is not None and hypothesis.candidate_ids == selected_ids
            ),
        )
        for index, hypothesis in enumerate(hypotheses, start=1)
    )


def selection_ledger_to_dicts(
    candidates: Iterable[ResidualEditCandidate],
    *,
    hypotheses: Sequence[ResidualHypothesis] = (),
    selected_hypothesis: ResidualHypothesis | None = None,
    applied_ids: Iterable[str] | None = None,
    include_metadata: bool = True,
    metadata_prefix: str = "metadata_",
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Serialize candidates and hypotheses for a residual-MHT selection ledger."""

    selected_ids: tuple[str, ...] = ()
    if selected_hypothesis is not None:
        selected_ids = tuple(
            str(candidate_id) for candidate_id in selected_hypothesis.candidate_ids
        )
    return {
        "candidates": candidates_to_dicts(
            candidates,
            selected_ids=selected_ids,
            applied_ids=applied_ids,
            include_metadata=include_metadata,
            metadata_prefix=metadata_prefix,
        ),
        "hypotheses": hypotheses_to_diagnostic_dicts(
            hypotheses,
            selected_hypothesis=selected_hypothesis,
        ),
    }
