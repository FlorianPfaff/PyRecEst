import numpy as np
from pyrecest.evaluation import get_distance_function


def test_euclidean_mtt_distance_preserves_ambiguous_row_oriented_3d_targets():
    distance = get_distance_function("euclidean_mtt", {"cutoff_distance": 100.0})
    first = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]
    )
    second = np.array(
        [
            [0.0, 0.0, 0.0],
            [13.0, 4.0, 0.0],
        ]
    )

    np.testing.assert_allclose(distance(first, second), 5.0)


def test_euclidean_mtt_distance_handles_empty_against_dim_first_targets():
    distance = get_distance_function("euclidean_mtt", {"cutoff_distance": 7.0})
    no_targets = np.empty((0, 2))
    dim_first_targets = np.vstack((10.0 * np.arange(5), np.zeros(5)))

    np.testing.assert_allclose(distance(no_targets, dim_first_targets), 35.0)
    np.testing.assert_allclose(distance(dim_first_targets, no_targets), 35.0)


def test_empty_target_dimension_disambiguates_small_dim_first_target_set():
    distance = get_distance_function("euclidean_mtt", {"cutoff_distance": 7.0})
    no_targets = np.empty((0, 2))
    dim_first_targets = np.array(
        [
            [0.0, 10.0, 20.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(distance(no_targets, dim_first_targets), 21.0)
    np.testing.assert_allclose(distance(dim_first_targets, no_targets), 21.0)
