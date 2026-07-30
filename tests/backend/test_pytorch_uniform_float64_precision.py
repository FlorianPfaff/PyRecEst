"""Regression tests for PyTorch uniform sampling precision."""

import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


def test_uniform_preserves_float64_bounds_and_random_precision():
    torch.manual_seed(0)
    samples = random.uniform(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
        size=(256,),
    )

    assert samples.dtype == torch.float64

    # A float32-generated uniform sample promoted afterwards is always on
    # the binary grid k / 2**24. Genuine float64 draws are not restricted
    # to that grid.
    scaled = samples * float(2**24)
    assert bool(torch.any(scaled != torch.round(scaled)))
