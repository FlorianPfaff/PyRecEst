from __future__ import annotations

import pytest
from pyrecest.diagnostics import EvidenceSupport, coerce_evidence_support


def test_evidence_support_exact_full_grid_is_headline_comparable() -> None:
    support = EvidenceSupport.exact_full_grid({"state_count": 5})

    assert support.support_type == "exact_full_grid"
    assert support.comparable
    assert not support.lower_bound
    assert support.headline_comparable
    assert support.to_dict()["diagnostics"]["state_count"] == 5
    assert support["headline_comparable"]


def test_evidence_support_truncated_lower_bound_is_not_headline_comparable() -> None:
    support = EvidenceSupport.truncated_lower_bound({"retained_candidates": 8})

    assert support.support_type == "truncated_lower_bound"
    assert not support.comparable
    assert support.lower_bound
    assert not support.headline_comparable


def test_evidence_support_from_mapping_preserves_unknown_fields_as_diagnostics() -> (
    None
):
    support = EvidenceSupport.from_mapping(
        {
            "support_type": "exact_sparse",
            "comparable": True,
            "lower_bound": False,
            "support_size": 17,
            "diagnostics": {"transition": "finite_radius"},
        }
    )

    assert support.support_type == "exact_sparse"
    assert support.headline_comparable
    assert support.diagnostics["transition"] == "finite_radius"
    assert support.diagnostics["support_size"] == 17


def test_coerce_evidence_support_rejects_unknown_support_type() -> None:
    with pytest.raises(ValueError, match="unsupported evidence support type"):
        coerce_evidence_support("candidate_magic")

    assert (
        coerce_evidence_support({"support_type": "unknown"}).support_type == "unknown"
    )
