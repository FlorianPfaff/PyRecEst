from pyrecest.evidence import EvidenceComputationMode


def test_reserved_evidence_diagnostics_survive_metadata_mutation():
    mode = EvidenceComputationMode.evidence_only(metadata={"source": "regression"})
    assert mode.metadata is not None

    mode.metadata.update(
        {
            "computation_mode": "full_smoothing",
            "only": 0,
            "return_smoothed": 1,
            "terminal_posterior": 0,
        }
    )

    diagnostics = mode.to_diagnostics()

    assert diagnostics["evidence_computation_mode"] == "evidence_only"
    assert diagnostics["evidence_only"] == 1
    assert diagnostics["evidence_return_smoothed"] == 0
    assert diagnostics["evidence_terminal_posterior"] == 1
    assert diagnostics["evidence_source"] == "regression"
