"""HOTA evaluation on prepared similarity matrices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._data import TrackingSequence

HOTA_ALPHAS = tuple(float(value) for value in np.arange(0.05, 0.99, 0.05))
_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class HotaCounts:
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    assa: np.ndarray
    loca: np.ndarray


def evaluate_hota(data: TrackingSequence) -> HotaCounts:
    """Evaluate HOTA counts for one sequence at the standard thresholds."""

    alphas = np.asarray(HOTA_ALPHAS, dtype=float)
    tp = np.zeros(len(alphas), dtype=int)
    fp = np.zeros(len(alphas), dtype=int)
    fn = np.zeros(len(alphas), dtype=int)
    loca_sum = np.zeros(len(alphas), dtype=float)
    if data.num_tracker_detections == 0:
        fn[:] = data.num_gt_detections
        return HotaCounts(tp, fp, fn, np.zeros_like(alphas), np.ones_like(alphas))
    if data.num_gt_detections == 0:
        fp[:] = data.num_tracker_detections
        return HotaCounts(tp, fp, fn, np.zeros_like(alphas), np.ones_like(alphas))

    potential = np.zeros((data.num_gt_ids, data.num_tracker_ids), dtype=float)
    gt_count = np.zeros((data.num_gt_ids, 1), dtype=float)
    tracker_count = np.zeros((1, data.num_tracker_ids), dtype=float)
    for gt_ids, tracker_ids, similarity in zip(
        data.gt_ids, data.tracker_ids, data.similarity_scores, strict=True
    ):
        if similarity.size:
            denominator = (
                similarity.sum(axis=0, keepdims=True)
                + similarity.sum(axis=1, keepdims=True)
                - similarity
            )
            normalized = np.zeros_like(similarity)
            valid = denominator > _EPS
            normalized[valid] = similarity[valid] / denominator[valid]
            potential[np.ix_(gt_ids, tracker_ids)] += normalized
        if len(gt_ids):
            gt_count[gt_ids] += 1
        if len(tracker_ids):
            tracker_count[0, tracker_ids] += 1
    alignment = potential / np.maximum(_EPS, gt_count + tracker_count - potential)
    match_counts = [np.zeros_like(potential) for _ in alphas]

    for gt_ids, tracker_ids, similarity in zip(
        data.gt_ids, data.tracker_ids, data.similarity_scores, strict=True
    ):
        if len(gt_ids) == 0:
            fp += len(tracker_ids)
            continue
        if len(tracker_ids) == 0:
            fn += len(gt_ids)
            continue
        score = alignment[np.ix_(gt_ids, tracker_ids)] * similarity
        match_rows, match_cols = linear_sum_assignment(-score)
        matched_similarity = similarity[match_rows, match_cols]
        for index, alpha in enumerate(alphas):
            valid = matched_similarity >= alpha - _EPS
            rows = match_rows[valid]
            cols = match_cols[valid]
            count = len(rows)
            tp[index] += count
            fn[index] += len(gt_ids) - count
            fp[index] += len(tracker_ids) - count
            if count:
                loca_sum[index] += float(similarity[rows, cols].sum())
                np.add.at(match_counts[index], (gt_ids[rows], tracker_ids[cols]), 1)

    assa = np.zeros(len(alphas), dtype=float)
    for index, counts in enumerate(match_counts):
        association = counts / np.maximum(1.0, gt_count + tracker_count - counts)
        assa[index] = float((counts * association).sum() / max(1, tp[index]))
    loca = np.maximum(1e-10, loca_sum) / np.maximum(1e-10, tp)
    return HotaCounts(tp, fp, fn, assa, loca)


def combine_hota(rows: Iterable[HotaCounts]) -> HotaCounts:
    """Combine sequence counts using TrackEval's detection-weighted rule."""

    rows = list(rows)
    length = len(HOTA_ALPHAS)
    if not rows:
        return HotaCounts(
            np.zeros(length, dtype=int),
            np.zeros(length, dtype=int),
            np.zeros(length, dtype=int),
            np.zeros(length, dtype=float),
            np.ones(length, dtype=float),
        )
    tp = sum((row.tp for row in rows), np.zeros(length, dtype=int))
    fp = sum((row.fp for row in rows), np.zeros(length, dtype=int))
    fn = sum((row.fn for row in rows), np.zeros(length, dtype=int))
    assa = sum(
        (row.assa * row.tp for row in rows), np.zeros(length, dtype=float)
    ) / np.maximum(1, tp)
    loca_weighted = sum(
        (row.loca * row.tp for row in rows), np.zeros(length, dtype=float)
    )
    loca = np.maximum(1e-10, loca_weighted) / np.maximum(1e-10, tp)
    return HotaCounts(tp, fp, fn, assa, loca)


def finalize_hota(counts: HotaCounts) -> dict[str, np.ndarray]:
    """Derive HOTA, DetA, AssA, and LocA arrays from additive counts."""

    deta = counts.tp / np.maximum(1, counts.tp + counts.fn + counts.fp)
    return {
        "hota": np.sqrt(deta * counts.assa),
        "deta": deta,
        "assa": counts.assa,
        "loca": counts.loca,
    }
