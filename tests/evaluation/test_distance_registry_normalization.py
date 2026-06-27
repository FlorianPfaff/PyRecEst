from pyrecest.backend import array
from pyrecest.evaluation.get_distance_function import (
    available_distance_functions,
    get_distance_function,
    register_distance_function,
)


def test_custom_distance_registry_strips_names_for_registration_and_lookup():
    register_distance_function(
        "  unit-test-distance-trimmed  ",
        lambda _name, _params: lambda _x, _y: 13.0,
    )

    assert "unit-test-distance-trimmed" in available_distance_functions()
    assert "  unit-test-distance-trimmed  " not in available_distance_functions()
    assert (
        get_distance_function("unit-test-distance-trimmed")(array([0.0]), array([1.0]))
        == 13.0
    )
    assert (
        get_distance_function("  unit-test-distance-trimmed  ")(
            array([0.0]), array([1.0])
        )
        == 13.0
    )
