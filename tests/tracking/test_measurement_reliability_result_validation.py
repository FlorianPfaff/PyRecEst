from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import MeasurementReliabilityResult


def test_reliability_result_rejects_non_boolean_accepted_flags() -> None:
    invalid_values = ("no", 1, np.array([True]))

    for accepted in invalid_values:
        with pytest.raises(ValueError, match="accepted must be a boolean"):
            MeasurementReliabilityResult(
                reliability=0.5,
                covariance_scale=1.0,
                covariance=np.eye(1),
                accepted=accepted,
                action="reliability_policy",
                mode="hard",
            )
