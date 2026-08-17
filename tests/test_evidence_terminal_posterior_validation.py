import pytest
from pyrecest.evidence import EvidenceComputationMode


def test_full_smoothing_requires_terminal_posterior():
    with pytest.raises(ValueError, match="terminal posterior"):
        EvidenceComputationMode(
            mode="full_smoothing",
            return_smoothed=True,
            terminal_posterior=False,
        )


def test_evidence_only_may_omit_terminal_posterior():
    mode = EvidenceComputationMode(
        mode="evidence_only",
        return_smoothed=False,
        terminal_posterior=False,
    )

    assert mode.terminal_posterior is False
