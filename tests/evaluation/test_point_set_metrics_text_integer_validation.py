import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import (
    deterministic_subsample,
    nearest_neighbor_distances,
)


def test_point_set_integer_validators_reject_text_scalars():
    points = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="positive integer"):
        nearest_neighbor_distances(points, points, query_chunk_size="1")

    with pytest.raises(ValueError, match="max_points"):
        deterministic_subsample(points, max_points="1")
