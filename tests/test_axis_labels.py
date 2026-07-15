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


@pytest.mark.parametrize(
    ("manifold_name", "expected_label"),
    [
        ("SE(2)", "Error in meters"),
        ("SE(2)-linear", "Error in meters"),
        ("SE(3)-bounded", "Error in radian"),
        ("hypersphere_symmetric", "Angular error in radian"),
    ],
)
def test_axis_label_accepts_common_mathematical_notation(manifold_name, expected_label):
    assert get_axis_label(manifold_name) == expected_label


@pytest.mark.parametrize("manifold_name", [None, "", "   ", "---"])
def test_axis_label_rejects_invalid_manifold_names(manifold_name):
    with pytest.raises(ValueError, match="manifold_name must be a non-empty string"):
        get_axis_label(manifold_name)


def test_axis_label_strips_manifold_name_whitespace():
    assert get_axis_label("  hypersphere  ") == "Error (orthodromic distance) in radian"
