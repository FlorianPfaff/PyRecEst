import pytest
from pyrecest.backend import array, pi, to_numpy
from pyrecest.evaluation.get_distance_function import (
    get_distance_function,
    register_distance_function,
)
from pyrecest.evaluation.get_extract_mean import get_extract_mean, register_extract_mean


def _as_float(value):
    return float(to_numpy(value))


def test_custom_distance_function_registry():
    register_distance_function(
        "unit-test-manifold", lambda _name, _params: lambda x, y: 42.0
    )

    assert (
        get_distance_function("unit-test-manifold")(array([0.0]), array([1.0])) == 42.0
    )


def test_euclidean_mtt_distance_uses_assignment_with_cutoff():
    distance = get_distance_function("euclideanMTT", {"cutoff_distance": 10.0})

    assert distance(array([[0.0], [10.0]]), array([[0.0], [12.0]])) == 2.0


def test_underscored_symmetric_hypersphere_distance_is_antipodal_invariant():
    distance = get_distance_function("hypersphere_symmetric")

    assert _as_float(distance(array([1.0, 0.0]), array([-1.0, 0.0]))) == pytest.approx(
        0.0
    )


def test_se2bounded_distance_uses_angular_component_before_linear_dispatch():
    distance = get_distance_function("se2bounded")

    x_estimate = array([[0.0], [1.0], [2.0]])
    x_true = array([[pi], [1.0], [2.0]])

    assert _as_float(distance(x_estimate, x_true)) == pytest.approx(float(pi))


def test_se3bounded_distance_uses_quaternion_component_before_linear_dispatch():
    distance = get_distance_function("se3bounded")

    x_estimate = array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    x_true = array([0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0])

    assert _as_float(distance(x_estimate, x_true)) == pytest.approx(float(pi / 2.0))


def test_custom_extract_mean_registry():
    register_extract_mean(
        "unit-test-mean", lambda _name, _mtt: lambda state: state["mean"]
    )

    assert get_extract_mean("unit-test-mean")({"mean": 3}) == 3


def test_symmetric_hypersphere_extract_mean_requires_custom_extractor():
    with pytest.raises(NotImplementedError, match="custom extractor"):
        get_extract_mean("hypersphereSymmetric")
