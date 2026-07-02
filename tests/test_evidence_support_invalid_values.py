import pytest

from pyrecest.diagnostics import EvidenceSupport


def test_evidence_support_rejects_list_value() -> None:
    bad_value = list(("unknown",))

    with pytest.raises(ValueError, match="unsupported evidence support type"):
        EvidenceSupport(bad_value)
