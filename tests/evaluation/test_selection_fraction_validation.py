from __future__ import annotations

import numpy as np
import pytest
from pyrecest.evaluation.selection import (
    retained_count_from_fraction,
    tail_rescue_quota_count,
)


def test_retained_count_from_fraction_rejects_bool_and_vector_fractions() -> None:
    for retention_fraction in (True, False, np.array([0.5])):
        with pytest.raises(ValueError, match="retention_fraction"):
            retained_count_from_fraction(10, retention_fraction)

    assert retained_count_from_fraction(10, np.array(0.5)) == 5


def test_tail_rescue_quota_count_rejects_bool_and_vector_fractions() -> None:
    for rescue_fraction in (True, False, np.array([0.5])):
        with pytest.raises(ValueError, match="rescue_fraction"):
            tail_rescue_quota_count(10, rescue_fraction=rescue_fraction)

    assert tail_rescue_quota_count(10, rescue_fraction=np.array(0.2)) == 2
