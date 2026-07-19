import pytest
from pyrecest.evidence import EvidenceComputationMode, resolve_evidence_computation_mode


class FailingArrayConversion:
    def __init__(self, error_type):
        self.error_type = error_type

    def __array__(self, dtype=None):
        del dtype
        raise self.error_type("array conversion failed")


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EvidenceComputationMode(return_smoothed=value),
        lambda value: EvidenceComputationMode(terminal_posterior=value),
        lambda value: EvidenceComputationMode.from_return_smoothed(value),
        lambda value: resolve_evidence_computation_mode(return_smoothed=value),
    ],
)
@pytest.mark.parametrize("error_type", [RuntimeError, OverflowError])
def test_evidence_bool_flags_normalize_array_conversion_errors(factory, error_type):
    with pytest.raises(ValueError, match="must be a bool"):
        factory(FailingArrayConversion(error_type))
