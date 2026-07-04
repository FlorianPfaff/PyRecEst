import pytest

from pyrecest.evidence import EvidenceComputationMode, resolve_evidence_computation_mode


class FailingArrayConversion:
    def __array__(self, dtype=None):
        del dtype
        raise RuntimeError("array conversion failed")


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EvidenceComputationMode(return_smoothed=value),
        lambda value: EvidenceComputationMode(terminal_posterior=value),
        lambda value: EvidenceComputationMode.from_return_smoothed(value),
        lambda value: resolve_evidence_computation_mode(return_smoothed=value),
    ],
)
def test_evidence_bool_flags_normalize_array_conversion_errors(factory):
    with pytest.raises(ValueError, match="must be a bool"):
        factory(FailingArrayConversion())
