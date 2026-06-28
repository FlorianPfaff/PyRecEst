import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


def _size_aware_samplers():
    return (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(3, size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
        lambda size: random.multinomial(3, [0.25, 0.75], size=size),
    )


@pytest.mark.parametrize(
    "scalar_size",
    [np.array(3, dtype=np.int64), torch.tensor(3, dtype=torch.int64)],
)
def test_size_arguments_accept_zero_dimensional_integer_arrays_and_tensors(scalar_size):
    random.seed(0)

    for sampler in _size_aware_samplers():
        sample = sampler(scalar_size)

        assert sample.shape[0] == 3


@pytest.mark.parametrize(
    "bad_size",
    [
        np.array(True),
        torch.tensor(True),
        np.array(3.0),
        torch.tensor(3.0),
    ],
)
def test_size_arguments_reject_zero_dimensional_non_integer_arrays_and_tensors(
    bad_size,
):
    for sampler in _size_aware_samplers():
        with pytest.raises(TypeError, match="size must"):
            sampler(bad_size)


@pytest.mark.parametrize(
    "probabilities",
    [
        np.array([0.5 + 0.0j, 0.5 + 0.0j]),
        torch.tensor([0.5 + 0.0j, 0.5 + 0.0j]),
    ],
)
def test_choice_rejects_complex_probabilities(probabilities):
    with pytest.raises(TypeError, match="real numeric"):
        random.choice(2, size=1, p=probabilities)
