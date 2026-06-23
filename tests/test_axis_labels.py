import pytest
from pyrecest.evaluation import get_axis_label


@pytest.mark.parametrize(
    ("manifold_name", "expected_label"),
    [
        ("hypersphereSymmetric", "Angular error in radian"),
        ("se2bounded", "Error in radian"),
        ("se3bounded", "Error in radian"),
    ],
)
def test_specific_manifolds_are_not_shadowed_by_generic_substrings(
    manifold_name, expected_label
):
    assert get_axis_label(manifold_name) == expected_label


@pytest.mark.parametrize(
    ("manifold_name", "expected_label"),
    [
        ("hypersphere", "Error (orthodromic distance) in radian"),
        ("se2", "Error in meters"),
        ("se3", "Error in meters"),
    ],
)
def test_generic_manifold_axis_labels_are_preserved(manifold_name, expected_label):
    assert get_axis_label(manifold_name) == expected_label
