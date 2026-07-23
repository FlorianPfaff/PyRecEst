import numpy as np
import pytest
from pyrecest.exceptions import NumericalStabilityError
from pyrecest.numerics import jittered_cholesky


def test_jittered_cholesky_rejects_overflowed_jitter_and_factor():
    matrix = np.array([[-1e308]])

    with pytest.raises(NumericalStabilityError, match="Cholesky factorization failed"):
        jittered_cholesky(matrix, initial_jitter=1e307, max_attempts=3)
