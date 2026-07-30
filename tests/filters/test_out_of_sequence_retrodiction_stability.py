import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, float64
from pyrecest.filters import retrodict_linear_gaussian


def test_retrodiction_preserves_extreme_finite_covariance():
    largest = np.finfo(np.float64).max
    covariance = array(
        [[largest, 0.0], [0.0, largest]],
        dtype=float64,
    )

    with np.errstate(over="raise", invalid="raise"):
        previous_mean, previous_covariance = retrodict_linear_gaussian(
            mean=array([0.0, 0.0], dtype=float64),
            covariance=covariance,
            system_matrix=array(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=float64,
            ),
        )

    assert allclose(
        previous_mean,
        array([0.0, 0.0], dtype=float64),
    )
    assert allclose(previous_covariance, covariance)
