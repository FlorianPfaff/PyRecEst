import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pyrecest._backend.pytorch import random  # noqa: E402


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


@pytest.mark.parametrize("bad_check_valid", ["error", None, 1])
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
