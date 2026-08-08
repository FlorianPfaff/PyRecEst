import warnings

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from pyrecest._backend.jax import random  # noqa: E402


def test_multivariate_normal_accepts_numpy_validation_keywords():
    random.seed(0)

    sample = random.multivariate_normal(
        [0.0, 1.0],
        [[2.0, 0.0], [0.0, 1.0]],
        size=3,
        check_valid="raise",
        tol=np.float64(1e-8),
    )

    assert sample.shape == (3, 2)


@pytest.mark.parametrize(
    "bad_check_valid",
    ["error", None, 1, [], {}, bytearray(b"warn")],
)
def test_multivariate_normal_rejects_invalid_check_valid_keyword(bad_check_valid):
    with pytest.raises(ValueError, match="check_valid"):
        random.multivariate_normal(
            [0.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            check_valid=bad_check_valid,
        )


@pytest.mark.parametrize(
    "bad_tol",
    [-1.0, np.nan, np.inf, True, [1e-8], "1e-8"],
)
def test_multivariate_normal_rejects_invalid_tol_keyword(bad_tol):
    with pytest.raises(ValueError, match="tol"):
        random.multivariate_normal(
            [0.0, 1.0],
            [[1.0, 0.0], [0.0, 1.0]],
            tol=bad_tol,
        )


def test_multivariate_normal_honors_custom_tolerance_for_raise():
    covariance = [[1.0, 0.0], [0.0, -1.0e-9]]

    with pytest.raises(ValueError, match="positive semidefinite"):
        random.multivariate_normal(
            [0.0, 0.0],
            covariance,
            check_valid="raise",
            tol=1.0e-10,
        )

    sample = random.multivariate_normal(
        [0.0, 0.0],
        covariance,
        size=4,
        check_valid="raise",
        tol=1.0e-8,
    )

    assert np.all(np.isfinite(np.asarray(sample)))


def test_multivariate_normal_warns_for_indefinite_covariance():
    with pytest.warns(RuntimeWarning, match="positive semidefinite"):
        sample = random.multivariate_normal(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, -1.0e-4]],
            size=4,
            check_valid="warn",
            tol=1.0e-8,
        )

    assert np.all(np.isfinite(np.asarray(sample)))


def test_multivariate_normal_ignore_skips_indefinite_covariance_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample = random.multivariate_normal(
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, -1.0e-4]],
            size=4,
            check_valid="ignore",
            tol=1.0e-8,
        )

    assert np.all(np.isfinite(np.asarray(sample)))
