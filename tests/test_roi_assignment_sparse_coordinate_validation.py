import pytest
from pyrecest.utils.roi_assignment import roi_iou


@pytest.mark.parametrize(
    ("roi", "coordinate_name"),
    [
        ({"ypix": [1.5], "xpix": [0]}, "ypix"),
        ({"ypix": [-1], "xpix": [0]}, "ypix"),
        ({"ypix": [0], "xpix": [float("nan")]}, "xpix"),
        (([0], [float("inf")]), "xpix"),
        (([True], [0]), "ypix"),
        ({"ypix": ["1"], "xpix": [0]}, "ypix"),
        ({"ypix": [0], "xpix": [1 + 0j]}, "xpix"),
    ],
)
def test_sparse_roi_rejects_lossy_or_invalid_coordinates(roi, coordinate_name):
    valid_roi = {"ypix": [0], "xpix": [0]}

    with pytest.raises(ValueError, match=coordinate_name):
        roi_iou(roi, valid_roi)


def test_sparse_roi_accepts_nonnegative_integer_valued_floats():
    sparse_mapping = {"ypix": [0.0, 1.0], "xpix": [2.0, 3.0]}
    sparse_tuple = ([0, 1], [2, 3])

    assert roi_iou(sparse_mapping, sparse_tuple) == 1.0
