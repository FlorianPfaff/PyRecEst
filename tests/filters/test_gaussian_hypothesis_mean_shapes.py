import numpy as np
import pytest
from pyrecest.filters.gaussian_hypothesis_mixture import WeightedGaussianHypothesis


@pytest.mark.parametrize(
    "mean",
    [
        np.array([[1.0, 2.0]]),
        np.array([[1.0], [2.0]]),
    ],
)
def test_weighted_gaussian_hypothesis_rejects_matrix_shaped_means(mean) -> None:
    with pytest.raises(ValueError, match="mean.*one-dimensional"):
        WeightedGaussianHypothesis(mean, np.eye(2))


def test_weighted_gaussian_hypothesis_preserves_scalar_mean_compatibility() -> None:
    hypothesis = WeightedGaussianHypothesis(1.0, np.array([[2.0]]))

    assert hypothesis.mean.shape == (1,)
    assert np.array_equal(hypothesis.mean, np.array([1.0]))


def test_weighted_gaussian_hypothesis_accepts_one_dimensional_means() -> None:
    hypothesis = WeightedGaussianHypothesis(np.array([1.0, 2.0]), np.eye(2))

    assert hypothesis.mean.shape == (2,)
