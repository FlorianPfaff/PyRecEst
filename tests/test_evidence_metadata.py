import pytest
from pyrecest.evidence import EvidenceComputationMode


def test_evidence_metadata_rejects_nonstring_keys():
    with pytest.raises(ValueError, match="metadata keys must be strings"):
        EvidenceComputationMode.full_smoothing(metadata={1: "integer key"})


def test_evidence_metadata_prevents_stringification_collisions():
    with pytest.raises(ValueError, match="metadata keys must be strings"):
        EvidenceComputationMode.evidence_only(metadata={1: "first", "1": "second"})


def test_evidence_metadata_keeps_string_keys_in_diagnostics():
    mode = EvidenceComputationMode.evidence_only(metadata={"source": "forward"})

    diagnostics = mode.to_diagnostics()

    assert diagnostics["evidence_source"] == "forward"
