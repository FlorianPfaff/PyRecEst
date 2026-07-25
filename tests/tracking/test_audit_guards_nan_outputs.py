from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import assert_selector_invariant_under_forbidden_key_changes

_ROWS = [
    {"candidate_id": "a", "score": 2.0, "audit_delta": -100.0},
    {"candidate_id": "b", "score": 1.0, "audit_delta": 100.0},
]
_FORBIDDEN = {"audit_delta"}


def test_selector_invariance_accepts_identical_scalar_nan_outputs() -> None:
    def selector(_candidate_rows):
        return float("nan")

    assert_selector_invariant_under_forbidden_key_changes(
        selector,
        _ROWS,
        _FORBIDDEN,
    )


def test_selector_invariance_accepts_aligned_nan_array_outputs() -> None:
    def selector(candidate_rows):
        scores = [row["score"] for row in candidate_rows]
        return np.asarray([max(scores), np.nan])

    assert_selector_invariant_under_forbidden_key_changes(
        selector,
        _ROWS,
        _FORBIDDEN,
    )


def test_selector_invariance_still_rejects_finite_differences_next_to_nan() -> None:
    def selector(candidate_rows):
        return np.asarray([len(candidate_rows[0]), np.nan])

    with pytest.raises(AssertionError, match="selector output changed"):
        assert_selector_invariant_under_forbidden_key_changes(
            selector,
            _ROWS,
            _FORBIDDEN,
        )
