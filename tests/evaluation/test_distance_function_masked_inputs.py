import numpy as np
import pytest
from pyrecest.evaluation.get_distance_function import get_distance_function


def test_rejects_masked_symmetry_count():
    with pytest.raises(ValueError, match="nSymm.*positive integer"):
        get_distance_function("circle", nSymm=np.ma.array(2, mask=True))


def test_rejects_masked_symmetry_offsets():
    offsets = np.ma.array([0.0, np.pi], mask=[False, True])

    with pytest.raises(ValueError, match="symmetryOffsets.*finite"):
        get_distance_function("circle", symmetryOffsets=offsets)


def test_rejects_masked_mtt_cutoff_distance():
    with pytest.raises(ValueError, match="cutoff_distance.*finite"):
        get_distance_function(
            "euclidean_mtt",
            {"cutoff_distance": np.ma.array(2.0, mask=True)},
        )


def test_rejects_masked_mtt_target_coordinates():
    distance = get_distance_function(
        "euclidean_mtt",
        {"cutoff_distance": 2.0},
    )
    masked_targets = np.ma.array(
        [[0.0, 1.0]],
        mask=[[False, True]],
    )

    with pytest.raises(ValueError, match="x1.*finite real numeric"):
        distance(masked_targets, np.array([[0.0, 1.0]]))


def test_clear_mask_wrappers_remain_supported():
    distance = get_distance_function(
        "euclidean_mtt",
        {"cutoff_distance": np.ma.array(2.0, mask=False)},
    )
    first = np.ma.array([[0.0, 0.0]], mask=False)
    second = np.ma.array([[1.0, 0.0]], mask=False)

    assert distance(first, second) == pytest.approx(1.0)

    symmetric_distance = get_distance_function(
        "circle",
        nSymm=np.ma.array(2, mask=False),
    )
    assert symmetric_distance(0.0, np.pi) == pytest.approx(0.0)
