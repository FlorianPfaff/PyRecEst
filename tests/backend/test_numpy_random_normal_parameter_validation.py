import numpy as np
import pytest
from pyrecest._backend.numpy import random


@pytest.mark.parametrize(
    "loc",
    [
        np.nan,
        np.inf,
        -np.inf,
        np.array([0.0, np.inf]),
    ],
)
def test_normal_rejects_nonfinite_loc(loc):
    with pytest.raises(ValueError, match="loc must be finite"):
        random.normal(loc=loc)


@pytest.mark.parametrize(
    "scale",
    [
        np.nan,
        np.inf,
        -np.inf,
        np.array([1.0, np.inf]),
    ],
)
def test_normal_rejects_nonfinite_scale(scale):
    with pytest.raises(ValueError, match="scale must be finite"):
        random.normal(scale=scale)
