import importlib.util

import numpy as np
import pytest


def _pytorch_random_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    import torch  # pylint: disable=import-outside-toplevel
    from pyrecest._backend.pytorch import (  # pylint: disable=import-outside-toplevel
        random,
    )

    return torch, random


def test_choice_normalizes_tiny_float64_probabilities_without_underflow():
    torch, random = _pytorch_random_backend()

    random.seed(0)
    probabilities = np.array([1.0e-300, 1.0e-300], dtype=np.float64)

    samples = random.choice(2, size=8, p=probabilities)

    assert samples.shape == (8,)
    assert torch.all((samples == 0) | (samples == 1))


def test_multinomial_normalizes_tiny_float64_probabilities_without_underflow():
    torch, random = _pytorch_random_backend()

    random.seed(0)
    probabilities = np.array([1.0e-300, 1.0e-300], dtype=np.float64)

    counts = random.multinomial(5, probabilities, size=3)

    assert counts.shape == (3, 2)
    assert torch.all(counts.sum(dim=-1) == 5)
