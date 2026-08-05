import numpy as np
import pytest

from pyrecest.exceptions import NumericalStabilityError
from pyrecest.numerics import assert_covariance_matrix, is_symmetric


def test_is_symmetric_rejects_extreme_finite_asymmetry_without_overflow():
    maximum = np.finfo(float).max
    matrix = np.array([[1.0, maximum], [-maximum, 1.0]])

    with np.errstate(all="raise"):
        assert is_symmetric(matrix) is False


def test_is_symmetric_accepts_extreme_finite_symmetric_matrix():
    maximum = np.finfo(float).max
    matrix = np.array([[1.0, maximum], [maximum, 1.0]])

    with np.errstate(all="raise"):
        assert is_symmetric(matrix) is True


def test_is_symmetric_preserves_absolute_tolerance_across_zero():
    epsilon = 2.0e-12
    matrix = np.array([[1.0, epsilon], [-epsilon, 1.0]])

    with np.errstate(all="raise"):
        assert is_symmetric(matrix, atol=4.0e-12) is True
        assert is_symmetric(matrix, atol=3.0e-12) is False


def test_covariance_validation_reports_extreme_asymmetry_cleanly():
    maximum = np.finfo(float).max
    matrix = np.array([[1.0, maximum], [-maximum, 1.0]])

    with np.errstate(all="raise"):
        with pytest.raises(NumericalStabilityError, match="must be symmetric"):
            assert_covariance_matrix(matrix)
