from __future__ import annotations

import pytest
from pyrecest.tracking import (
    ForbiddenKeyAccessError,
    assert_selector_invariant_under_forbidden_key_changes,
    guarded_mapping,
    guarded_mappings,
    poison_forbidden_keys,
    poison_forbidden_keys_in_mappings,
    strip_forbidden_keys,
    strip_forbidden_keys_from_mappings,
)

FORBIDDEN = {"audit_label", "audit_delta"}


def test_guarded_mapping_raises_on_forbidden_key_access() -> None:
    row = guarded_mapping({"score": 1.0, "audit_label": "x"}, FORBIDDEN)

    assert row["score"] == 1.0
    with pytest.raises(ForbiddenKeyAccessError):
        _ = row["audit_label"]
    with pytest.raises(ForbiddenKeyAccessError):
        row.get("audit_label")
    with pytest.raises(ForbiddenKeyAccessError):
        "audit_label" in row
    assert list(row) == ["score"]


def test_strip_and_poison_forbidden_keys() -> None:
    row = {"candidate_id": "a", "score": 1.0, "audit_delta": 99.0}

    assert strip_forbidden_keys(row, FORBIDDEN) == {"candidate_id": "a", "score": 1.0}
    assert (
        poison_forbidden_keys(row, FORBIDDEN)["audit_delta"]
        == "__PYRECEST_FORBIDDEN_AUDIT_VALUE__"
    )


def test_sequence_helpers_strip_poison_and_guard_rows() -> None:
    rows = [
        {"candidate_id": "a", "score": 2.0, "audit_delta": 1.0},
        {"candidate_id": "b", "score": 1.0, "audit_label": "x"},
    ]

    stripped = strip_forbidden_keys_from_mappings(rows, FORBIDDEN)
    poisoned = poison_forbidden_keys_in_mappings(
        rows, FORBIDDEN, poison_value="SENTINEL"
    )
    guarded = guarded_mappings(rows, FORBIDDEN)

    assert all("audit_delta" not in row for row in stripped)
    assert poisoned[0]["audit_delta"] == "SENTINEL"
    with pytest.raises(ForbiddenKeyAccessError):
        guarded[0].get("audit_delta")


def test_selector_invariance_helper_passes_for_allowed_feature_selector() -> None:
    rows = [
        {"candidate_id": "a", "score": 2.0, "audit_delta": -100.0},
        {"candidate_id": "b", "score": 1.0, "audit_label": "x"},
    ]

    def selector(candidate_rows):
        return max(candidate_rows, key=lambda row: row["score"])["candidate_id"]

    assert_selector_invariant_under_forbidden_key_changes(selector, rows, FORBIDDEN)


def test_selector_invariance_helper_catches_forbidden_feature_selector() -> None:
    rows = [
        {"candidate_id": "a", "score": 2.0, "audit_delta": -100.0},
        {"candidate_id": "b", "score": 1.0, "audit_delta": 100.0},
    ]

    def selector(candidate_rows):
        return max(candidate_rows, key=lambda row: row.get("audit_delta", 0.0))[
            "candidate_id"
        ]

    with pytest.raises((AssertionError, ForbiddenKeyAccessError)):
        assert_selector_invariant_under_forbidden_key_changes(selector, rows, FORBIDDEN)
