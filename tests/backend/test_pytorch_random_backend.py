import pytest

torch = pytest.importorskip("torch")

from pyrecest._backend.pytorch import random  # noqa: E402


@pytest.mark.parametrize(
    "bad_size",
    [True, False, (True,), [False, 2], 1.5, (2.0,), "3"],
)
def test_size_arguments_reject_bool_and_non_integral_dimensions(bad_size):
    samplers = (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(3, size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
    )

    for sampler in samplers:
        with pytest.raises(TypeError):
            sampler(bad_size)


@pytest.mark.parametrize("bad_size", [-1, (2, -1)])
def test_size_arguments_reject_negative_dimensions(bad_size):
    samplers = (
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.randint(0, 3, size=size),
        lambda size: random.choice(3, size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
    )

    for sampler in samplers:
        with pytest.raises(ValueError):
            sampler(bad_size)


def test_rand_accepts_numpy_positional_dimensions():
    random.seed(0)

    assert random.rand(2, 3).shape == (2, 3)
    assert random.rand(4).shape == (4,)


def test_rand_rejects_ambiguous_positional_and_size_arguments():
    with pytest.raises(TypeError, match="positional dimensions or size"):
        random.rand(2, size=(3,))


def test_randint_accepts_numpy_broadcasted_bounds_without_explicit_size():
    random.seed(0)
    low = torch.tensor([0, 10])
    high = torch.tensor([3, 13])

    samples = random.randint(low, high)

    assert samples.shape == (2,)
    assert torch.all(samples >= low)
    assert torch.all(samples < high)


def test_randint_accepts_numpy_broadcasted_bounds_with_explicit_size():
    random.seed(0)
    low = torch.tensor([0, 10])
    high = torch.tensor([3, 13])

    samples = random.randint(low, high, size=(4, 2))

    assert samples.shape == (4, 2)
    assert torch.all(samples >= low)
    assert torch.all(samples < high)


def test_randint_accepts_array_high_only():
    random.seed(0)
    high = torch.tensor([3, 13])

    samples = random.randint(high)

    assert samples.shape == (2,)
    assert torch.all(samples >= 0)
    assert torch.all(samples < high)


def test_randint_rejects_incompatible_array_bounds_and_size():
    with pytest.raises(ValueError, match="broadcast"):
        random.randint(torch.tensor([0, 10]), torch.tensor([3, 13]), size=(3,))


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (torch.tensor([0, 10]), torch.tensor([3])),
        (torch.tensor([0, 10]), torch.tensor([0, 13])),
    ],
)
def test_randint_rejects_invalid_array_bounds(low, high):
    with pytest.raises(ValueError):
        random.randint(low, high)


@pytest.mark.parametrize("bad_size", [(), (3,), (3, 1)])
def test_normal_rejects_array_parameters_incompatible_with_explicit_size(bad_size):
    with pytest.raises(ValueError, match="broadcast"):
        random.normal(torch.tensor([1.0, 2.0]), 1.0, size=bad_size)


@pytest.mark.parametrize("bad_size", [(), (3,), (3, 1)])
def test_uniform_rejects_array_parameters_incompatible_with_explicit_size(bad_size):
    with pytest.raises(ValueError, match="broadcast"):
        random.uniform(torch.tensor([1.0, 2.0]), 3.0, size=bad_size)


def test_normal_accepts_array_parameters_with_compatible_explicit_size():
    random.seed(0)

    samples = random.normal(torch.tensor([1.0, 2.0]), 1.0, size=(4, 2))

    assert samples.shape == (4, 2)


def test_uniform_accepts_array_parameters_with_compatible_explicit_size():
    random.seed(0)

    samples = random.uniform(torch.tensor([1.0, 2.0]), 3.0, size=(4, 2))

    assert samples.shape == (4, 2)
    assert torch.all(samples >= torch.tensor([1.0, 2.0]))
    assert torch.all(samples <= 3.0)


def test_scalar_and_empty_tuple_sizes_keep_scalar_shape():
    assert random.rand().shape == ()
    assert random.rand(size=()).shape == ()
    assert random.normal(size=()).shape == ()
    assert random.uniform(size=()).shape == ()
    assert random.randint(0, 3, size=()).shape == ()
    assert random.multivariate_normal([0.0], [[1.0]], size=()).shape == (1,)


@pytest.mark.parametrize(
    "population",
    [True, False, torch.tensor(True), torch.tensor(False)],
)
def test_choice_rejects_boolean_scalar_population(population):
    with pytest.raises(ValueError, match="positive integer or an array"):
        random.choice(population)


def test_zero_sized_choice_still_works_for_empty_population():
    sample = random.choice(0, size=(0,))

    assert sample.shape == (0,)


def test_choice_without_replacement_shuffle_false_preserves_order():
    values = torch.tensor([10, 20, 30, 40, 50])
    matrix = torch.tensor([[10, 20, 30], [40, 50, 60]])

    random.seed(0)
    samples = random.choice(values, size=values.shape[0], replace=False, shuffle=False)
    column_samples = random.choice(
        matrix,
        size=matrix.shape[1],
        replace=False,
        axis=1,
        shuffle=False,
    )

    assert torch.equal(samples, values)
    assert torch.equal(column_samples, matrix)


@pytest.mark.parametrize("control", ["replace", "shuffle"])
def test_choice_rejects_non_boolean_controls(control):
    kwargs = {control: "False"}

    with pytest.raises(TypeError, match=f"{control} must be a boolean"):
        random.choice(torch.arange(3), size=2, **kwargs)
