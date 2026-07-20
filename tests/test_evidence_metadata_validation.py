import pytest
from pyrecest.evidence import EvidenceComputationMode


def test_evidence_computation_mode_rejects_text_metadata():
    with pytest.raises(ValueError, match="metadata must be a mapping or None"):
        EvidenceComputationMode(metadata="ab")


def test_evidence_computation_mode_rejects_sequence_metadata():
    with pytest.raises(ValueError, match="metadata must be a mapping or None"):
        EvidenceComputationMode(metadata=["ab"])
