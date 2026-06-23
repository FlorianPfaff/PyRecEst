import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import nearest_neighbor_distances


def test_nearest_neighbor_distances_rejects_noninteger_chunk_sizes():
    points = np.array([[0.0], [1.0]])

    for chunk_size in (0, -1, 1.5, np.nan, np.inf, True, np.array([1])):
        with pytest.raises(ValueError, match="positive integer"):
            nearest_neighbor_distances(
                points,
                points,
                query_chunk_size=chunk_size,
            )


def test_nearest_neighbor_distances_accepts_integer_like_chunk_size():
    points = np.array([[0.0], [1.0]])

    distances = nearest_neighbor_distances(
        points,
        points,
        query_chunk_size=np.array(1.0),
    )

    np.testing.assert_allclose(distances, [0.0, 0.0])
