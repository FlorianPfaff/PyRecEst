"""Regression tests for exact PyTorch array-bound randint sampling."""

import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


def test_array_randint_retains_low_integer_bits_for_large_ranges():
    torch.manual_seed(0)
    low = torch.zeros(64, dtype=torch.int64)
    high = torch.full((64,), 2**62, dtype=torch.int64)

    samples = random.randint(low, high)

    # Float32 scaling would constrain every sample to multiples of 2**38.
    assert bool(torch.any(torch.remainder(samples, 256) != 0))


def test_array_randint_handles_nearly_full_int64_span_without_overflow():
    torch.manual_seed(0)
    low = torch.full((128,), -(2**63), dtype=torch.int64)
    high = torch.full((128,), 2**63 - 1, dtype=torch.int64)

    samples = random.randint(low, high)

    assert bool(torch.all(samples >= low))
    assert bool(torch.all(samples < high))
    assert samples.unique().numel() > 2
