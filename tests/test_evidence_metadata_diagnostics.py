import pytest
from pyrecest.evidence import EvidenceComputationMode


@pytest.mark.parametrize(
    "key", ("computation_mode", "only", "return_smoothed", "terminal_posterior")
)
def test_evidence_metadata_rejects_reserved_diagnostic_keys(key):
    with pytest.raises(ValueError, match="metadata keys would overwrite"):
        EvidenceComputationMode.evidence_only(metadata={key: "shadow"})


def test_evidence_metadata_cannot_overwrite_stable_diagnostics():
    diagnostics = EvidenceComputationMode.evidence_only(
        metadata={"scenario": "demo"}
    ).to_diagnostics()

    assert diagnostics["evidence_only"] == 1
    assert diagnostics["evidence_return_smoothed"] == 0
    assert diagnostics["evidence_terminal_posterior"] == 1
    assert diagnostics["evidence_scenario"] == "demo"
