import numpy as np
import pytest
from pyrecest.evaluation.point_set_metrics import nearest_neighbor_distances


@pytest.mark.parametrize(
    "invalid_points",
    [
        [["0.0"], ["1.0"]],
        np.array([[True], [False]], dtype=np.bool_),
        np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex128),
        np.array([[1.0], [2.0]], dtype=object),
    ],
)
def test_point_set_metrics_rejects_nonreal_coordinate_dtypes(invalid_points):
    reference = np.array([[0.0]])

    with pytest.raises(ValueError, match="finite real numeric"):
        nearest_neighbor_distances(invalid_points, reference)


def test_point_set_metrics_still_accepts_numeric_integer_coordinates():
    distances = nearest_neighbor_distances(
        np.array([[0], [2]], dtype=np.int64),
        np.array([[1]], dtype=np.uint64),
    )

    np.testing.assert_allclose(distances, [1.0, 1.0])
