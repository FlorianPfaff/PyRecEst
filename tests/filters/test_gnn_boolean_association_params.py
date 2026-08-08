import numpy as np
import pytest

from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


@pytest.mark.parametrize("name", ("square_dist", "maximize_cardinality"))
@pytest.mark.parametrize(
    "value",
    (
        "false",
        0,
        1,
        [False],
        np.array([False]),
        np.ma.array(True, mask=True),
    ),
)
def test_gnn_rejects_non_boolean_association_flags(name, value):
    with pytest.raises(ValueError, match=rf"{name} must be a boolean"):
        GlobalNearestNeighbor(association_param={name: value})


@pytest.mark.parametrize("name", ("square_dist", "maximize_cardinality"))
@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.array(False, dtype=bool), False),
    ),
)
def test_gnn_accepts_scalar_boolean_association_flags(name, value, expected):
    tracker = GlobalNearestNeighbor(association_param={name: value})

    assert tracker.association_param[name] is expected
