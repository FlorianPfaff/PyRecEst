from pyrecest.backend import array
from pyrecest.evaluation.get_distance_function import (
    get_distance_function,
    register_distance_function,
)
from pyrecest.evaluation.get_extract_mean import get_extract_mean, register_extract_mean


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


def test_custom_extract_mean_registry():
    register_extract_mean(
        "unit-test-mean", lambda _name, _mtt: lambda state: state["mean"]
    )

    assert get_extract_mean("unit-test-mean")({"mean": 3}) == 3
