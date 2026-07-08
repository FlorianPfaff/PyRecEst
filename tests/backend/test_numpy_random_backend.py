import numpy as np
import pytest
from pyrecest._backend.numpy import random


def test_rand_accepts_backend_size_keyword():
    random.seed(0)

    samples = random.rand(size=(2, 3))

    assert samples.shape == (2, 3)
    assert samples.dtype == np.float64


def test_rand_keeps_numpy_positional_dimensions():
    random.seed(0)

    assert random.rand(2, 3).shape == (2, 3)
    assert random.rand(4).shape == (4,)


def test_rand_rejects_ambiguous_positional_and_size_arguments():
    with pytest.raises(TypeError, match="positional dimensions or size"):
        random.rand(2, size=(3,))


@pytest.mark.parametrize(
    "bad_size",
    [
        True,
        False,
        np.bool_(True),
        np.array(True),
        (True,),
        [np.bool_(False), 2],
        np.array([True, 2], dtype=object),
    ],
)
@pytest.mark.parametrize(
    "sampler",
    [
        lambda size: random.rand(size=size),
        lambda size: random.uniform(size=size),
        lambda size: random.normal(size=size),
        lambda size: random.multivariate_normal([0.0], [[1.0]], size=size),
        lambda size: random.choice(np.arange(5), size=size),
    ],
)
def test_random_wrappers_reject_boolean_size_arguments(sampler, bad_size):
    with pytest.raises(TypeError, match="size"):
        sampler(bad_size)


@pytest.mark.parametrize(
    "dims",
    [
        (True,),
        (np.bool_(True),),
        (2, False),
        ([True, 2],),
    ],
)
def test_rand_rejects_boolean_positional_dimensions(dims):
    with pytest.raises(TypeError, match="size"):
        random.rand(*dims)


def test_random_wrappers_accept_integer_array_size_arguments():
    random.seed(0)

    assert random.rand(size=np.array([2, 3], dtype=np.int64)).shape == (2, 3)
    assert random.uniform(size=np.array(2, dtype=np.int64)).shape == (2,)
    assert random.normal(size=np.array([2, 3], dtype=np.int64)).shape == (2, 3)
    assert random.multivariate_normal(
        [0.0], [[1.0]], size=np.array(2, dtype=np.int64)
    ).shape == (2, 1)
    assert random.choice(np.arange(5), size=np.array([2], dtype=np.int64)).shape == (2,)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (0.2, 3),
        (0, 3.7),
        (False, 3),
        (0, True),
        (np.array([0.0, 10.0]), np.array([3, 13])),
        ([0, 10], [3.0, 13.0]),
        (np.array([False, 0], dtype=object), np.array([3, 13])),
        (np.array([0, 10]), np.array([3, np.bool_(True)], dtype=object)),
        (np.array([0 + 0j, 10]), np.array([3, 13])),
    ],
)
def test_randint_rejects_non_integer_bounds(low, high):
    with pytest.raises(TypeError, match="integer"):
        random.randint(low, high)


@pytest.mark.parametrize(
    "high",
    [
        3.0,
        True,
        np.array([3.0, 13.0]),
        [3, 13.0],
        np.array([True, False]),
    ],
)
def test_randint_rejects_non_integer_high_only(high):
    with pytest.raises(TypeError, match="integer"):
        random.randint(high)


def test_randint_accepts_integer_array_bounds():
    random.seed(0)
    low = np.array([0, 10])
    high = np.array([3, 13])

    samples = random.randint(low, high, size=(4, 2))

    assert samples.shape == (4, 2)
    assert np.all(samples >= low)
    assert np.all(samples < high)


@pytest.mark.parametrize(
    "n",
    [
        True,
        False,
        np.bool_(True),
        np.array(True),
        np.array([1]),
        1.5,
        "1",
    ],
)
def test_multinomial_rejects_non_integer_or_boolean_sample_counts(n):
    with pytest.raises(TypeError, match="non-negative integer"):
        random.multinomial(n, [1.0])


def test_multinomial_rejects_negative_sample_counts():
    with pytest.raises(ValueError, match="non-negative"):
        random.multinomial(-1, [1.0])


def test_multinomial_accepts_integer_like_scalar_sample_counts():
    random.seed(0)

    samples = random.multinomial(np.array(2, dtype=np.int64), [0.25, 0.75], size=3)

    assert samples.shape == (3, 2)
    assert np.all(samples.sum(axis=1) == 2)


@pytest.mark.parametrize(
    "bad_size",
    [
        True,
        False,
        np.bool_(True),
        (True,),
        [np.bool_(False), 2],
        np.array(True),
        np.array([True, 2], dtype=object),
    ],
)
def test_multinomial_rejects_boolean_size_arguments(bad_size):
    with pytest.raises(TypeError, match="size"):
        random.multinomial(2, [0.25, 0.75], size=bad_size)


def test_multinomial_accepts_integer_like_size_argument():
    random.seed(0)

    samples = random.multinomial(2, [0.25, 0.75], size=np.array(3, dtype=np.int64))

    assert samples.shape == (3, 2)
    assert np.all(samples.sum(axis=1) == 2)


def test_multinomial_accepts_large_finite_unnormalized_probabilities():
    random.seed(0)

    samples = random.multinomial(5, np.array([1e308, 1e308]), size=3)

    assert samples.shape == (3, 2)
    assert np.all(samples.sum(axis=1) == 5)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        (False, 1.0),
        (0.0, True),
        (np.array([False, False]), np.array([1.0, 2.0])),
        ([False, 0.0], [1.0, 2.0]),
        ([0.0, 0.5], [1.0, np.bool_(True)]),
        (
            np.array([0.0, np.bool_(False)], dtype=object),
            np.array([1.0, 2.0], dtype=object),
        ),
    ],
)
def test_uniform_rejects_boolean_bounds(low, high):
    with pytest.raises(TypeError, match="real numeric"):
        random.uniform(low, high)


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ("0.0", 1.0),
        (0.0, "1.0"),
        (np.array(["0.0", "0.5"]), np.array([1.0, 1.5])),
    ],
)
def test_uniform_rejects_text_bounds(low, high):
    with pytest.raises(TypeError, match="real numeric"):
        random.uniform(low, high)


@pytest.mark.parametrize(
    "scale",
    [
        True,
        False,
        np.bool_(True),
        np.array(False),
        [1.0, False],
        np.array([1.0, np.bool_(True)], dtype=object),
    ],
)
def test_normal_rejects_boolean_scale(scale):
    with pytest.raises(TypeError, match="scale.*boolean"):
        random.normal(scale=scale)


def test_normal_rejects_negative_scale():
    with pytest.raises(ValueError, match="non-negative"):
        random.normal(scale=-1.0)

    with pytest.raises(ValueError, match="non-negative"):
        random.normal(scale=np.array([1.0, -0.1]))


def test_choice_without_replacement_shuffle_false_preserves_order():
    values = np.array([10, 20, 30, 40, 50])
    matrix = np.array([[10, 20, 30], [40, 50, 60]])

    random.seed(0)
    samples = random.choice(values, size=values.shape[0], replace=False, shuffle=False)
    column_samples = random.choice(
        matrix,
        size=matrix.shape[1],
        replace=False,
        axis=1,
        shuffle=False,
    )

    np.testing.assert_array_equal(samples, values)
    np.testing.assert_array_equal(column_samples, matrix)


def test_choice_without_replacement_shuffle_false_preserves_population_order_for_unsorted_values():
    values = np.array([30, 10, 20, 40])

    random.seed(1)
    expected_indices = np.sort(np.random.choice(values.shape[0], size=3, replace=False))
    expected = values[expected_indices]

    random.seed(1)
    samples = random.choice(values, size=3, replace=False, shuffle=False)

    np.testing.assert_array_equal(samples, expected)


@pytest.mark.parametrize("control", ["replace", "shuffle"])
@pytest.mark.parametrize("value", [np.array(False), np.array(True)])
def test_choice_accepts_numpy_scalar_boolean_controls(control, value):
    kwargs = {control: value}
    size = 2
    if control == "shuffle":
        kwargs["replace"] = False
        size = 3

    samples = random.choice(np.arange(3), size=size, **kwargs)

    assert samples.shape == (size,)


@pytest.mark.parametrize("control", ["replace", "shuffle"])
def test_choice_rejects_non_boolean_controls(control):
    kwargs = {control: "False"}

    with pytest.raises(TypeError, match=f"{control} must be a boolean"):
        random.choice(np.arange(3), size=2, **kwargs)


def test_choice_accepts_large_finite_unnormalized_probabilities():
    random.seed(0)

    samples = random.choice(np.arange(2), size=8, p=np.array([1e308, 1e308]))

    assert samples.shape == (8,)
    assert np.all((samples == 0) | (samples == 1))


@pytest.mark.parametrize(
    "probabilities",
    [
        [True, False, False],
        np.array([True, False, False]),
        np.array([0.5, np.bool_(False), 0.5], dtype=object),
    ],
)
def test_choice_rejects_boolean_probabilities(probabilities):
    with pytest.raises(TypeError, match="real numeric"):
        random.choice(np.arange(3), p=probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        ["1.0", "0.0", "0.0"],
        np.array(["0.2", "0.3", "0.5"]),
        np.array([0.5, "0.5", 0.0], dtype=object),
    ],
)
def test_choice_rejects_text_probabilities(probabilities):
    with pytest.raises(TypeError, match="real numeric"):
        random.choice(np.arange(3), p=probabilities)
