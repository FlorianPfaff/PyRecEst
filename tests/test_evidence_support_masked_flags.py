import numpy as np
import pytest
from pyrecest.diagnostics import EvidenceSupport


@pytest.mark.parametrize("field", ["comparable", "lower_bound"])
@pytest.mark.parametrize("payload", [True, False])
def test_evidence_support_rejects_masked_boolean_flags(field, payload):
    with pytest.raises(ValueError, match=rf"{field} must be a boolean value"):
        EvidenceSupport(**{field: np.ma.array(payload, mask=True)})


def test_evidence_support_accepts_fully_unmasked_boolean_scalars():
    support = EvidenceSupport(
        comparable=np.ma.array(True, mask=False),
        lower_bound=np.ma.array(False, mask=False),
    )

    assert support.comparable is True
    assert support.lower_bound is False
