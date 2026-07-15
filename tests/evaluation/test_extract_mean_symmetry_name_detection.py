import pytest

from pyrecest.evaluation.get_extract_mean import get_extract_mean


class _DirectionalState:
    def mean_direction(self):
        return "direction"


@pytest.mark.parametrize(
    "manifold_name",
    ("asymmetric_circle", "nonsymmetric_hypertorus"),
)
def test_non_symmetric_substrings_do_not_trigger_symmetry_guard(manifold_name):
    extractor = get_extract_mean(manifold_name)

    assert extractor(_DirectionalState()) == "direction"


@pytest.mark.parametrize(
    "manifold_name",
    (
        "circle_symmetric",
        "symmetric_circle",
        "hypertorus_symmetric",
        "symm_hypertorus",
        "symmetriccircle",
        "circlesymm",
    ),
)
def test_explicit_symmetric_names_still_require_a_convention(manifold_name):
    with pytest.raises(NotImplementedError, match="explicit convention"):
        get_extract_mean(manifold_name)
