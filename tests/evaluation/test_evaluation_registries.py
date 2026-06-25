import pytest
from pyrecest.backend import array, pi, to_numpy
from pyrecest.evaluation.get_distance_function import (
    available_distance_functions,
    get_distance_function,
    register_distance_function,
)
from pyrecest.evaluation.get_extract_mean import (
    available_extract_mean_functions,
    get_extract_mean,
    register_extract_mean,
)


def _as_float(value):
    return float(to_numpy(value))


def test_custom_distance_function_registry():
    register_distance_function(
        "unit-test-manifold", lambda _name, _params: lambda x, y: 42.0
    )

    assert (
        get_distance_function("unit-test-manifold")(array([0.0]), array([1.0])) == 42.0
    )


def test_custom_distance_function_registry_strips_names_for_registration_and_lookup():
    register_distance_function(
        "  unit-test-manifold-trimmed  ",
        lambda _name, _params: lambda x, y: 17.0,
    )

    assert "unit-test-manifold-trimmed" in available_distance_functions()
    assert "  unit-test-manifold-trimmed  " not in available_distance_functions()
    assert (
        get_distance_function("unit-test-manifold-trimmed")(array([0.0]), array([1.0]))
        == 17.0
    )
    assert (
        get_distance_function("  unit-test-manifold-trimmed  ")(
            array([0.0]), array([1.0])
        )
        == 17.0
    )


def test_custom_registries_validate_names_and_factories():
    registries = (register_distance_function, register_extract_mean)

    for register in registries:
        with pytest.raises(ValueError, match="non-empty string"):
            register("", lambda *_args: None)
        with pytest.raises(ValueError, match="non-empty string"):
            register("   ", lambda *_args: None)
        with pytest.raises(ValueError, match="non-empty string"):
            register(123, lambda *_args: None)
        with pytest.raises(TypeError, match="factory must be callable"):
            register("invalid-factory", None)


def test_euclidean_mtt_distance_uses_assignment_with_cutoff():
    distance = get_distance_function("euclideanMTT", {"cutoff_distance": 10.0})

    assert distance(array([[0.0], [10.0]]), array([[0.0], [12.0]])) == 2.0


def test_underscored_symmetric_hypersphere_distance_is_antipodal_invariant():
    distance = get_distance_function("hypersphere_symmetric")

    assert _as_float(distance(array([1.0, 0.0]), array([-1.0, 0.0]))) == pytest.approx(
        0.0
    )


def test_angular_distances_clip_inner_product_roundoff():
    cases = (
        ("hypersphere", array([1.0, 0.0]), array([1.0 + 1e-12, 0.0])),
        ("hypersphere_symmetric", array([1.0, 0.0]), array([1.0 + 1e-12, 0.0])),
        (
            "se3bounded",
            array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
            array([1.0 + 1e-12, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
        ),
    )

    for manifold_name, estimate, truth in cases:
        distance = get_distance_function(manifold_name)

        assert _as_float(distance(estimate, truth)) == pytest.approx(0.0)


def test_se2bounded_distance_uses_angular_component_before_linear_dispatch():
    distance = get_distance_function("se2bounded")

    x_estimate = array([[0.0], [1.0], [2.0]])
    x_true = array([[pi], [1.0], [2.0]])

    assert _as_float(distance(x_estimate, x_true)) == pytest.approx(float(pi))


def test_se2bounded_distance_accepts_vector_states():
    distance = get_distance_function("se2bounded")

    x_estimate = array([0.0, 1.0, 2.0])
    x_true = array([pi, 1.0, 2.0])

    assert _as_float(distance(x_estimate, x_true)) == pytest.approx(float(pi))


def test_se2_and_se3_linear_distances_accept_vector_states():
    se2_distance = get_distance_function("se2")
    assert _as_float(
        se2_distance(array([0.0, 1.0, 2.0]), array([0.0, 4.0, 6.0]))
    ) == pytest.approx(5.0)

    se3_distance = get_distance_function("se3")
    assert _as_float(
        se3_distance(
            array([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
            array([1.0, 0.0, 0.0, 0.0, 4.0, 6.0, 3.0]),
        )
    ) == pytest.approx(5.0)


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


def test_custom_extract_mean_registry_strips_names_for_registration_and_lookup():
    register_extract_mean(
        "  unit-test-mean-trimmed  ",
        lambda _name, _mtt: lambda state: state["mean"],
    )

    assert "unit-test-mean-trimmed" in available_extract_mean_functions()
    assert "  unit-test-mean-trimmed  " not in available_extract_mean_functions()
    assert get_extract_mean("unit-test-mean-trimmed")({"mean": 7}) == 7
    assert get_extract_mean("  unit-test-mean-trimmed  ")({"mean": 8}) == 8


def test_euclidean_extract_mean_preserves_raw_array_state():
    state = array([1.0, 2.0])

    extracted = get_extract_mean("euclidean")(state)

    assert to_numpy(extracted).tolist() == pytest.approx([1.0, 2.0])


def test_symmetric_hypersphere_extract_mean_requires_custom_extractor():
    with pytest.raises(NotImplementedError, match="custom extractor"):
        get_extract_mean("hypersphereSymmetric")


def test_underscored_symmetric_hypersphere_extract_mean_requires_convention():
    with pytest.raises(NotImplementedError, match="explicit convention"):
        get_extract_mean("hypersphere_symmetric")
