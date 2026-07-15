import pytest
from pyrecest.evidence import resolve_evidence_computation_mode


class StringifyingMode:
    def __str__(self):
        return "evidence"


def test_resolver_rejects_non_string_modes_before_stringification():
    with pytest.raises(ValueError, match="unknown evidence computation mode"):
        resolve_evidence_computation_mode(StringifyingMode())


def test_resolver_accepts_documented_string_aliases():
    mode = resolve_evidence_computation_mode("evidence")

    assert mode.evidence_only_requested
