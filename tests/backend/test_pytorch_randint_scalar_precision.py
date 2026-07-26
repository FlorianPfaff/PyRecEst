import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (np.int64(0), np.int64(2**62)),
        (np.array(0, dtype=np.int64), np.array(2**62, dtype=np.int64)),
        (
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(2**62, dtype=torch.int64),
        ),
    ],
)
def test_randint_integer_like_scalar_bounds_retain_full_precision(low, high):
    random.seed(0)

    samples = random.randint(low, high, size=64)

    assert samples.dtype == torch.int64
    assert torch.any(torch.remainder(samples, 256) != 0)


@pytest.mark.parametrize(
    "high",
    [
        np.int64(2**62),
        np.array(2**62, dtype=np.int64),
        torch.tensor(2**62, dtype=torch.int64),
    ],
)
def test_randint_integer_like_scalar_high_retains_full_precision(high):
    random.seed(0)

    samples = random.randint(high, size=64)

    assert samples.dtype == torch.int64
    assert torch.any(torch.remainder(samples, 256) != 0)
