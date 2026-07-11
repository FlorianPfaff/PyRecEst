import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    "population",
    [
        4,
        torch.tensor([10, 20, 30, 40]),
    ],
)
def test_weighted_choice_without_replacement_rejects_insufficient_support(population):
    probabilities = torch.tensor([1.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Fewer non-zero entries in p than size"):
        random.choice(
            population,
            size=2,
            replace=False,
            p=probabilities,
        )


def test_weighted_choice_without_replacement_can_exhaust_positive_support():
    values = torch.tensor([10, 20, 30])

    samples = random.choice(
        values,
        size=2,
        replace=False,
        p=torch.tensor([0.25, 0.75, 0.0]),
    )

    assert set(samples.tolist()) == {10, 20}
