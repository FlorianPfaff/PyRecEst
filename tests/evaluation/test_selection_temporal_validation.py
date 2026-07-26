from __future__ import annotations

import numpy as np
import pytest
from pyrecest.evaluation.selection import (
    retained_count_from_fraction,
    tail_rescue_quota_count,
    top_count_mask,
    top_fraction_mask,
)


def _temporal_values(payload: int = 1):
    timestamp = np.datetime64(f"1970-01-01T00:00:00.00000000{payload}")
    duration = np.timedelta64(payload, "ns")
    return (
        duration,
        timestamp,
        np.asarray(duration),
        np.asarray(timestamp),
        np.asarray(duration, dtype=object),
        np.asarray(timestamp, dtype=object),
    )


def test_selection_helpers_reject_temporal_integer_controls() -> None:
    for temporal in _temporal_values():
        with pytest.raises(ValueError, match="retained_count"):
            top_count_mask([1.0, 2.0], temporal)
        with pytest.raises(ValueError, match="item_count"):
            retained_count_from_fraction(temporal, 0.5)
        with pytest.raises(ValueError, match="min_count"):
            retained_count_from_fraction(2, 0.5, min_count=temporal)


def test_selection_helpers_reject_temporal_fraction_controls() -> None:
    for temporal in _temporal_values():
        with pytest.raises(ValueError, match="retention_fraction"):
            retained_count_from_fraction(2, temporal)
        with pytest.raises(ValueError, match="retention_fraction"):
            top_fraction_mask([1.0, 2.0], temporal)
        with pytest.raises(ValueError, match="rescue_fraction"):
            tail_rescue_quota_count(2, rescue_fraction=temporal)
