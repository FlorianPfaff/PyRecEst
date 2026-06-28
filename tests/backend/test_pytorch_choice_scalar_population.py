import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    "population",
    [np.array(3, dtype=np.int32), np.array(3, dtype=np.int64)],
)
def test_choice_accepts_zero_dimensional_numpy_integer_population(population):
    random.seed(0)

    samples = random.choice(population, size=16)

    assert samples.shape == (16,)
    assert torch.all(samples >= 0)
    assert torch.all(samples < int(population.item()))


def test_choice_accepts_zero_dimensional_numpy_integer_population_without_replacement():
    random.seed(0)

    samples = random.choice(np.array(3, dtype=np.int64), size=3, replace=False)

    assert samples.shape == (3,)
    assert torch.equal(torch.sort(samples).values, torch.arange(3))


def test_choice_allows_zero_sized_sample_from_zero_dimensional_numpy_zero_population():
    samples = random.choice(np.array(0, dtype=np.int64), size=(0,))

    assert samples.shape == (0,)


@pytest.mark.parametrize("population", [np.array(True), np.array(3.0)])
def test_choice_rejects_zero_dimensional_numpy_non_integer_population(population):
    with pytest.raises(ValueError, match="positive integer or an array"):
        random.choice(population)
