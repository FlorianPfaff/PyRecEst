"""Validated sequence representation shared by tracking metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingSequence:
    """Per-frame dense identities and caller-provided pairwise similarities.

    Identity arrays contain unique, zero-based indices. Similarity matrices must
    have shape ``(len(gt_ids[t]), len(tracker_ids[t]))`` and values in ``[0, 1]``.
    """

    gt_ids: Sequence[np.ndarray]
    tracker_ids: Sequence[np.ndarray]
    similarity_scores: Sequence[np.ndarray]
    num_gt_ids: int
    num_tracker_ids: int

    def __post_init__(self) -> None:
        num_gt_ids = _nonnegative_int(self.num_gt_ids, name="num_gt_ids")
        num_tracker_ids = _nonnegative_int(
            self.num_tracker_ids, name="num_tracker_ids"
        )
        gt_frames = tuple(
            _identity_array(values, num_gt_ids, f"gt_ids[{index}]")
            for index, values in enumerate(self.gt_ids)
        )
        tracker_frames = tuple(
            _identity_array(values, num_tracker_ids, f"tracker_ids[{index}]")
            for index, values in enumerate(self.tracker_ids)
        )
        if len(gt_frames) != len(tracker_frames):
            raise ValueError("gt_ids and tracker_ids must contain the same number of frames")
        if len(self.similarity_scores) != len(gt_frames):
            raise ValueError("similarity_scores must contain one matrix per frame")
        similarities = tuple(
            _similarity_matrix(
                values,
                (len(gt_ids), len(tracker_ids)),
                f"similarity_scores[{index}]",
            )
            for index, (values, gt_ids, tracker_ids) in enumerate(
                zip(self.similarity_scores, gt_frames, tracker_frames, strict=True)
            )
        )
        object.__setattr__(self, "gt_ids", gt_frames)
        object.__setattr__(self, "tracker_ids", tracker_frames)
        object.__setattr__(self, "similarity_scores", similarities)
        object.__setattr__(self, "num_gt_ids", num_gt_ids)
        object.__setattr__(self, "num_tracker_ids", num_tracker_ids)

    @property
    def frame_count(self) -> int:
        return len(self.gt_ids)

    @property
    def num_gt_detections(self) -> int:
        return sum(len(ids) for ids in self.gt_ids)

    @property
    def num_tracker_detections(self) -> int:
        return sum(len(ids) for ids in self.tracker_ids)


def unit_interval_scalar(value: object, *, name: str) -> float:
    """Validate a finite non-boolean scalar in ``[0, 1]``."""

    message = f"{name} must be a finite scalar in [0, 1]"
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        array = np.asarray(value)
        result = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if array.shape != () or array.dtype.kind in "bMmSUc" or not np.isfinite(result):
        raise ValueError(message)
    if not 0.0 <= result <= 1.0:
        raise ValueError(message)
    return result


def _identity_array(values: object, upper_bound: int, name: str) -> np.ndarray:
    message = f"{name} must be a one-dimensional integer array"
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 1:
        raise ValueError(message)
    if array.size == 0:
        result = np.empty(0, dtype=int)
    else:
        if array.dtype.kind not in "iu" or array.dtype.kind == "b":
            raise ValueError(message)
        result = np.array(array, dtype=int, copy=True)
        if np.any(result < 0) or np.any(result >= upper_bound):
            raise ValueError(f"{name} contains an identity outside the declared range")
        if len(np.unique(result)) != len(result):
            raise ValueError(f"{name} contains duplicate identities within one frame")
    result.setflags(write=False)
    return result


def _similarity_matrix(
    values: object, expected_shape: tuple[int, int], name: str
) -> np.ndarray:
    message = f"{name} must be a finite numeric matrix"
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if raw.size == 0 and 0 in expected_shape:
        result = np.zeros(expected_shape, dtype=float)
    else:
        if raw.dtype.kind in "bMmSUc":
            raise ValueError(message)
        if raw.dtype.kind == "O":
            invalid_types = (
                bool,
                np.bool_,
                str,
                bytes,
                bytearray,
                complex,
                np.complexfloating,
                np.datetime64,
                np.timedelta64,
                type(None),
            )
            if any(isinstance(value, invalid_types) for value in raw.reshape(-1)):
                raise ValueError(message)
        try:
            result = np.array(values, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
    if result.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {result.shape}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError(f"{name} values must be finite and lie in [0, 1]")
    result.setflags(write=False)
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    message = f"{name} must be a non-negative integer"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(message)
    result = int(array.item())
    if result < 0:
        raise ValueError(message)
    return result
