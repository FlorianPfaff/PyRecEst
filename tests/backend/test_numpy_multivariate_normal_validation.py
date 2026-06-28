import numpy as np
import pytest
from pyrecest._backend.numpy import random


@pytest.mark.parametrize(
    ("mean", "cov"),
    [
        ([True, False], np.eye(2)),
        (np.array([0.0, np.bool_(True)], dtype=object), np.eye(2)),
        ([0.0, 0.0], [[True, False], [False, True]]),
        (
            [0.0, 0.0],
            np.array([[1.0, np.bool_(False)], [0.0, 1.0]], dtype=object),
        ),
    ],
)
def test_multivariate_normal_rejects_boolean_parameters(mean, cov):
    with pytest.raises(TypeError, match="real numeric"):
        random.multivariate_normal(mean, cov)


@pytest.mark.parametrize(
    ("mean", "cov"),
    [
        (["0.0", "1.0"], np.eye(2)),
        ([0.0, 0.0], np.array([["1.0", "0.0"], ["0.0", "1.0"]])),
    ],
)
def test_multivariate_normal_rejects_text_parameters(mean, cov):
    with pytest.raises(TypeError, match="real numeric"):
        random.multivariate_normal(mean, cov)


@pytest.mark.parametrize(
    ("mean", "cov"),
    [
        ([np.nan, 0.0], np.eye(2)),
        ([0.0, 0.0], [[1.0, np.inf], [0.0, 1.0]]),
    ],
)
def test_multivariate_normal_rejects_nonfinite_parameters(mean, cov):
    with pytest.raises(ValueError, match="finite"):
        random.multivariate_normal(mean, cov)
