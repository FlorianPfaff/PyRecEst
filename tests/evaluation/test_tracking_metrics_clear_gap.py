from __future__ import annotations

import numpy as np
import pytest
from pyrecest.evaluation.tracking_metrics import TrackingSequence, evaluate_clear


@pytest.mark.parametrize(
    ("gap_gt_ids", "gap_tracker_ids", "gap_similarity", "expected_fp", "expected_fn"),
    [
        (
            np.empty(0, dtype=int),
            np.array([0], dtype=int),
            np.empty((0, 1), dtype=float),
            1,
            0,
        ),
        (
            np.array([0, 1], dtype=int),
            np.empty(0, dtype=int),
            np.empty((2, 0), dtype=float),
            0,
            2,
        ),
    ],
)
def test_clear_continuity_does_not_cross_empty_frame(
    gap_gt_ids: np.ndarray,
    gap_tracker_ids: np.ndarray,
    gap_similarity: np.ndarray,
    expected_fp: int,
    expected_fn: int,
) -> None:
    data = TrackingSequence(
        gt_ids=(
            np.array([0, 1], dtype=int),
            gap_gt_ids,
            np.array([0, 1], dtype=int),
        ),
        tracker_ids=(
            np.array([0, 1], dtype=int),
            gap_tracker_ids,
            np.array([0, 1], dtype=int),
        ),
        similarity_scores=(
            np.eye(2, dtype=float),
            gap_similarity,
            np.array([[0.6, 0.9], [0.9, 0.6]], dtype=float),
        ),
        num_gt_ids=2,
        num_tracker_ids=2,
    )

    counts = evaluate_clear(data, threshold=0.5)

    assert counts.tp == 4
    assert counts.fp == expected_fp
    assert counts.fn == expected_fn
    assert counts.id_switches == 2
    assert counts.motp_sum == pytest.approx(3.8)
