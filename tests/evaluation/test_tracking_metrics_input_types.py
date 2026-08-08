from __future__ import annotations

import numpy as np
import pytest

from pyrecest.evaluation.tracking_metrics import TrackingSequence


@pytest.mark.parametrize(
    "similarity",
    [
        np.array([[True]], dtype=bool),
        np.array([["0.5"]]),
        np.array([[b"0.5"]]),
        np.array([[np.datetime64("1970-01-02", "D")]]),
        np.array([[np.timedelta64(1, "ns")]]),
        np.array([[True]], dtype=object),
        np.array([["0.5"]], dtype=object),
        np.array([[np.datetime64("1970-01-02", "D")]], dtype=object),
        np.array([[np.timedelta64(1, "ns")]], dtype=object),
    ],
)
def test_rejects_non_numeric_similarity_values(similarity: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite numeric matrix"):
        TrackingSequence(
            gt_ids=([0],),
            tracker_ids=([0],),
            similarity_scores=(similarity,),
            num_gt_ids=1,
            num_tracker_ids=1,
        )


def test_accepts_numeric_object_similarity_values() -> None:
    data = TrackingSequence(
        gt_ids=([0],),
        tracker_ids=([0],),
        similarity_scores=(np.array([[np.float32(0.5)]], dtype=object),),
        num_gt_ids=1,
        num_tracker_ids=1,
    )

    assert data.similarity_scores[0].dtype == float
    assert data.similarity_scores[0][0, 0] == pytest.approx(0.5)
