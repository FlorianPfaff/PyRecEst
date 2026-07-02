"""Local identity-set F1 fixes for track-evaluation exports."""

from __future__ import annotations

from typing import Any


def patch_identity_f1(namespace: dict[str, Any]) -> None:
    """Install an identity-set scorer with a zero-valued disjoint F1."""

    safe_ratio = namespace["_safe_ratio"]
    zero_ratio = namespace["_zero_ratio"]

    def score_identity_sets(
        predicted: set[Any],
        reference: set[Any],
        *,
        prefix: str,
        predicted_total_name: str,
        reference_total_name: str,
    ) -> dict[str, float | int]:
        true_positives = len(predicted & reference)
        false_positives = len(predicted - reference)
        false_negatives = len(reference - predicted)
        precision = safe_ratio(true_positives, true_positives + false_positives)
        recall = safe_ratio(true_positives, true_positives + false_negatives)
        f1 = zero_ratio(2.0 * precision * recall, precision + recall)
        return {
            f"{prefix}_true_positives": true_positives,
            f"{prefix}_false_positives": false_positives,
            f"{prefix}_false_negatives": false_negatives,
            f"{prefix}_precision": precision,
            f"{prefix}_recall": recall,
            f"{prefix}_f1": f1,
            predicted_total_name: len(predicted),
            reference_total_name: len(reference),
        }

    namespace["_score_identity_sets"] = score_identity_sets
