import numpy as np
import pytest


torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    ("low", "high", "dtype", "message"),
    [
        ([-1], [1], torch.uint8, "low is out of bounds for uint8"),
        ([0], [257], torch.uint8, "high is out of bounds for uint8"),
        ([-129], [-128], torch.int8, "low is out of bounds for int8"),
        ([0], [129], np.int8, "high is out of bounds for int8"),
    ],
)
def test_array_randint_rejects_bounds_outside_output_dtype(
    low, high, dtype, message
):
    with pytest.raises(ValueError, match=message):
        random.randint(low, high, dtype=dtype)


@pytest.mark.parametrize(
    ("low", "high", "dtype", "expected"),
    [
        ([255], [256], torch.uint8, 255),
        ([127], [128], torch.int8, 127),
    ],
)
def test_array_randint_accepts_exclusive_endpoint_above_dtype_max(
    low, high, dtype, expected
):
    sample = random.randint(low, high, dtype=dtype)

    assert sample.dtype == dtype
    assert sample.item() == expected
