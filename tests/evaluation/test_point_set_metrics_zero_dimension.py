import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import (
    as_point_set,
    nearest_neighbor_distances,
)


def test_as_point_set_rejects_zero_coordinate_dimension():
    with pytest.raises(ValueError, match="at least one coordinate"):
        as_point_set(np.empty((2, 0)))


def test_nearest_neighbor_distances_rejects_zero_coordinate_dimension_cleanly():
    with pytest.raises(ValueError, match="at least one coordinate"):
        nearest_neighbor_distances(np.empty((2, 0)), np.empty((3, 0)))
