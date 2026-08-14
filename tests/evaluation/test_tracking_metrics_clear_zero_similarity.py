"""Regression coverage for zero-similarity CLEAR continuations."""

from __future__ import annotations

import numpy as np
from pyrecest.evaluation.tracking_metrics import TrackingSequence, evaluate_clear


def _single_track_sequence(second_similarity: float) -> TrackingSequence:
    return TrackingSequence(
        gt_ids=(np.array([0]), np.array([0])),
        tracker_ids=(np.array([0]), np.array([0])),
        similarity_scores=(
            np.array([[1.0]]),
            np.array([[second_similarity]]),
        ),
        num_gt_ids=1,
        num_tracker_ids=1,
    )


def test_zero_threshold_does_not_resurrect_zero_similarity_by_continuity() -> None:
    zero_counts = evaluate_clear(_single_track_sequence(0.0), threshold=0.0)
    positive_counts = evaluate_clear(_single_track_sequence(0.01), threshold=0.0)

    assert (
        zero_counts.tp,
        zero_counts.fp,
        zero_counts.fn,
        zero_counts.id_switches,
        zero_counts.motp_sum,
    ) == (1, 1, 1, 0, 1.0)
    assert (
        positive_counts.tp,
        positive_counts.fp,
        positive_counts.fn,
        positive_counts.id_switches,
    ) == (2, 0, 0, 0)
    assert positive_counts.motp_sum == 1.01
