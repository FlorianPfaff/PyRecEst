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
def test_array_randint_rejects_bounds_outside_output_dtype(low, high, dtype, message):
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


@pytest.mark.parametrize(
    ("bound_dtype", "low", "high"),
    [
        (torch.int8, -128, 127),
        (torch.int16, -32768, 32767),
        (torch.int32, -(2**31), 2**31 - 1),
    ],
)
def test_array_randint_avoids_narrow_bound_span_overflow(bound_dtype, low, high):
    torch.manual_seed(0)
    samples = random.randint(
        torch.tensor([low], dtype=bound_dtype),
        torch.tensor([high], dtype=bound_dtype),
        size=(1024,),
        dtype=bound_dtype,
    )

    assert bool(torch.all(samples >= low))
    assert bool(torch.all(samples < high))
    assert bool(torch.any(samples < 0))
    assert bool(torch.any(samples > 0))
