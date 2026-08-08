from __future__ import annotations

import numpy as np
import pytest

from pyrecest.evaluation.tracking_metrics import TrackingSequence


@pytest.mark.parametrize(
    ("gt_ids", "tracker_ids", "similarity", "num_gt_ids", "num_tracker_ids"),
    [
        ([], [0, 1], np.empty((2, 0)), 0, 2),
        ([0, 1], [], np.empty((0, 2)), 2, 0),
        ([], [0], np.empty((0, 0, 1)), 0, 1),
    ],
)
def test_rejects_misshaped_empty_similarity_matrices(
    gt_ids,
    tracker_ids,
    similarity,
    num_gt_ids,
    num_tracker_ids,
) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        TrackingSequence(
            gt_ids=(gt_ids,),
            tracker_ids=(tracker_ids,),
            similarity_scores=(similarity,),
            num_gt_ids=num_gt_ids,
            num_tracker_ids=num_tracker_ids,
        )


def test_preserves_flat_empty_similarity_shorthand() -> None:
    data = TrackingSequence(
        gt_ids=([0], []),
        tracker_ids=([], [0, 1]),
        similarity_scores=([], []),
        num_gt_ids=1,
        num_tracker_ids=2,
    )

    assert data.similarity_scores[0].shape == (1, 0)
    assert data.similarity_scores[1].shape == (0, 2)
    assert not data.similarity_scores[0].flags.writeable
    assert not data.similarity_scores[1].flags.writeable
