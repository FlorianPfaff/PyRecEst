from __future__ import annotations

import math

import numpy as np
import pytest

from pyrecest.evaluation.tracking_metrics import (
    HOTA_ALPHAS,
    TrackingSequence,
    combine_clear,
    combine_hota,
    combine_identity,
    evaluate_clear,
    evaluate_hota,
    evaluate_identity,
    finalize_clear,
    finalize_hota,
    finalize_identity,
)


def _sequence(
    gt_ids: list[list[int]],
    tracker_ids: list[list[int]],
    similarities: list[list[list[float]]],
    *,
    num_gt_ids: int,
    num_tracker_ids: int,
) -> TrackingSequence:
    return TrackingSequence(
        gt_ids=tuple(np.asarray(values, dtype=int) for values in gt_ids),
        tracker_ids=tuple(np.asarray(values, dtype=int) for values in tracker_ids),
        similarity_scores=tuple(np.asarray(values, dtype=float) for values in similarities),
        num_gt_ids=num_gt_ids,
        num_tracker_ids=num_tracker_ids,
    )


def test_perfect_sequence_scores_one() -> None:
    data = _sequence(
        [[0], [0]],
        [[0], [0]],
        [[[1.0]], [[1.0]]],
        num_gt_ids=1,
        num_tracker_ids=1,
    )

    hota = finalize_hota(evaluate_hota(data))
    clear = finalize_clear(evaluate_clear(data, threshold=0.5))
    identity = finalize_identity(evaluate_identity(data, threshold=0.5))

    assert np.all(hota["hota"] == pytest.approx(1.0))
    assert np.all(hota["deta"] == pytest.approx(1.0))
    assert np.all(hota["assa"] == pytest.approx(1.0))
    assert np.all(hota["loca"] == pytest.approx(1.0))
    assert clear == pytest.approx({"mota": 1.0, "motp": 1.0})
    assert identity == pytest.approx(
        {"idf1": 1.0, "id_precision": 1.0, "id_recall": 1.0}
    )


def test_identity_switch_reduces_association_metrics() -> None:
    data = _sequence(
        [[0], [0]],
        [[0], [1]],
        [[[1.0]], [[1.0]]],
        num_gt_ids=1,
        num_tracker_ids=2,
    )

    hota_counts = evaluate_hota(data)
    clear_counts = evaluate_clear(data, threshold=0.5)
    identity_counts = evaluate_identity(data, threshold=0.5)
    hota = finalize_hota(hota_counts)

    assert np.all(hota["deta"] == pytest.approx(1.0))
    assert np.all(hota["assa"] == pytest.approx(0.5))
    assert np.all(hota["hota"] == pytest.approx(math.sqrt(0.5)))
    assert clear_counts.id_switches == 1
    assert finalize_clear(clear_counts)["mota"] == pytest.approx(0.5)
    assert finalize_identity(identity_counts)["idf1"] == pytest.approx(0.5)


def test_hota_uses_multiple_localization_thresholds() -> None:
    data = _sequence(
        [[0]],
        [[0]],
        [[[1.0 / 3.0]]],
        num_gt_ids=1,
        num_tracker_ids=1,
    )

    counts = evaluate_hota(data)
    metrics = finalize_hota(counts)
    matched = np.asarray(HOTA_ALPHAS) <= 1.0 / 3.0 + 1e-12

    assert counts.tp.tolist() == matched.astype(int).tolist()
    assert float(np.mean(metrics["hota"])) == pytest.approx(float(np.mean(matched)))


def test_combination_is_detection_weighted() -> None:
    perfect = _sequence(
        [[0]],
        [[0]],
        [[[1.0]]],
        num_gt_ids=1,
        num_tracker_ids=1,
    )
    missed = _sequence(
        [[0], [0], [0]],
        [[], [], []],
        [[], [], []],
        num_gt_ids=1,
        num_tracker_ids=0,
    )

    hota = combine_hota([evaluate_hota(perfect), evaluate_hota(missed)])
    clear = combine_clear(
        [evaluate_clear(perfect, threshold=0.5), evaluate_clear(missed, threshold=0.5)]
    )
    identity = combine_identity(
        [
            evaluate_identity(perfect, threshold=0.5),
            evaluate_identity(missed, threshold=0.5),
        ]
    )

    assert np.all(finalize_hota(hota)["deta"] == pytest.approx(0.25))
    assert finalize_clear(clear)["mota"] == pytest.approx(0.25)
    assert finalize_identity(identity)["id_recall"] == pytest.approx(0.25)


def test_zero_threshold_rejects_zero_similarity_identity_pairs() -> None:
    zero = _sequence(
        [[0]],
        [[0]],
        [[[0.0]]],
        num_gt_ids=1,
        num_tracker_ids=1,
    )
    positive = _sequence(
        [[0]],
        [[0]],
        [[[0.01]]],
        num_gt_ids=1,
        num_tracker_ids=1,
    )

    zero_counts = evaluate_identity(zero, threshold=0.0)
    positive_counts = evaluate_identity(positive, threshold=0.0)

    assert (zero_counts.tp, zero_counts.fp, zero_counts.fn) == (0, 1, 1)
    assert (positive_counts.tp, positive_counts.fp, positive_counts.fn) == (1, 0, 0)


def test_tracking_sequence_validates_shapes_and_identity_ranges() -> None:
    with pytest.raises(ValueError, match="shape"):
        TrackingSequence(
            gt_ids=([0],),
            tracker_ids=([0],),
            similarity_scores=(np.zeros((2, 1)),),
            num_gt_ids=1,
            num_tracker_ids=1,
        )
    with pytest.raises(ValueError, match="declared range"):
        TrackingSequence(
            gt_ids=([1],),
            tracker_ids=([0],),
            similarity_scores=([[1.0]],),
            num_gt_ids=1,
            num_tracker_ids=1,
        )
    with pytest.raises(ValueError, match="duplicate identities"):
        TrackingSequence(
            gt_ids=([0, 0],),
            tracker_ids=([0],),
            similarity_scores=(np.ones((2, 1)),),
            num_gt_ids=1,
            num_tracker_ids=1,
        )


def test_tracking_sequence_rejects_masked_values_before_coercion() -> None:
    invalid_inputs = (
        (
            {
                "gt_ids": (np.ma.array([0], mask=[True]),),
                "tracker_ids": ([0],),
                "similarity_scores": ([[1.0]],),
                "num_gt_ids": 1,
                "num_tracker_ids": 1,
            },
            "gt_ids",
        ),
        (
            {
                "gt_ids": ([0],),
                "tracker_ids": (np.ma.array([0], mask=[True]),),
                "similarity_scores": ([[1.0]],),
                "num_gt_ids": 1,
                "num_tracker_ids": 1,
            },
            "tracker_ids",
        ),
        (
            {
                "gt_ids": ([0],),
                "tracker_ids": ([0],),
                "similarity_scores": (np.ma.array([[1.0]], mask=[[True]]),),
                "num_gt_ids": 1,
                "num_tracker_ids": 1,
            },
            "similarity_scores",
        ),
        (
            {
                "gt_ids": (),
                "tracker_ids": (),
                "similarity_scores": (),
                "num_gt_ids": np.ma.array(1, mask=True),
                "num_tracker_ids": 0,
            },
            "num_gt_ids",
        ),
    )

    for kwargs, field_name in invalid_inputs:
        with pytest.raises(ValueError, match=field_name):
            TrackingSequence(**kwargs)  # type: ignore[arg-type]


def test_tracking_sequence_accepts_clear_mask_wrappers() -> None:
    data = TrackingSequence(
        gt_ids=(np.ma.array([0], mask=[False]),),
        tracker_ids=(np.ma.array([0], mask=[False]),),
        similarity_scores=(np.ma.array([[1.0]], mask=[[False]]),),
        num_gt_ids=np.ma.array(1, mask=False),
        num_tracker_ids=np.ma.array(1, mask=False),
    )

    assert data.gt_ids[0].tolist() == [0]
    assert data.tracker_ids[0].tolist() == [0]
    assert data.similarity_scores[0].tolist() == [[1.0]]
    assert data.num_gt_ids == 1
    assert data.num_tracker_ids == 1


def test_metric_thresholds_are_validated() -> None:
    empty = TrackingSequence((), (), (), 0, 0)

    for threshold in (
        -0.1,
        1.1,
        float("nan"),
        True,
        "0.5",
        np.array([0.5]),
        np.ma.array(0.5, mask=True),
    ):
        with pytest.raises(ValueError, match="threshold"):
            evaluate_clear(empty, threshold=threshold)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="threshold"):
            evaluate_identity(empty, threshold=threshold)  # type: ignore[arg-type]
