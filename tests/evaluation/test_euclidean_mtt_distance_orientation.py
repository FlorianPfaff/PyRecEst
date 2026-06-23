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
