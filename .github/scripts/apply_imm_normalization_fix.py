from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


source_path = Path("src/pyrecest/filters/interacting_multiple_model_filter.py")
source = source_path.read_text(encoding="utf-8")

class_anchor = "\n\nclass InteractingMultipleModelFilter(AbstractFilter, EuclideanFilterMixin):\n"
helper = '''


def _normalize_nonnegative_weights(weights, positive_mass_message):
    """Normalize finite nonnegative weights without overflowing their sum."""

    weight_scale = pyrecest.backend.max(weights)
    if not bool(isfinite(weight_scale)) or not bool(weight_scale > 0.0):
        raise ValueError(positive_mass_message)

    scale_root = pyrecest.backend.sqrt(weight_scale)
    # Splitting by sqrt(scale) avoids reciprocal underflow on JAX for values near
    # the largest representable floating-point number while preserving ratios.
    scaled_weights = (weights / scale_root) / scale_root
    scaled_sum = scaled_weights.sum()
    if not bool(isfinite(scaled_sum)) or not bool(scaled_sum > 0.0):
        raise ValueError(positive_mass_message)

    normalized_weights = scaled_weights / scaled_sum
    return normalized_weights, bool(allclose(normalized_weights, weights))
'''
source = replace_once(source, class_anchor, helper + class_anchor)

old_transition = '''        row_sums = transition_matrix.sum(axis=1)
        if pyrecest.backend.any(row_sums <= 0.0):
            raise ValueError(
                "Each row of transition_matrix must sum to a positive value."
            )
        if not allclose(row_sums, 1.0):
            warnings.warn(
                "Rows of transition_matrix do not sum to one. Renormalizing rows.",
                UserWarning,
            )
            transition_matrix = transition_matrix / row_sums[:, None]
        return transition_matrix
'''
new_transition = '''        normalized_rows = []
        rows_sum_to_one = True
        for row in transition_matrix:
            normalized_row, row_sums_to_one = _normalize_nonnegative_weights(
                row,
                "Each row of transition_matrix must sum to a positive value.",
            )
            normalized_rows.append(normalized_row)
            rows_sum_to_one = rows_sum_to_one and row_sums_to_one

        transition_matrix = stack(normalized_rows)
        if not rows_sum_to_one:
            warnings.warn(
                "Rows of transition_matrix do not sum to one. Renormalizing rows.",
                UserWarning,
            )
        return transition_matrix
'''
source = replace_once(source, old_transition, new_transition)

old_modes = '''            curr_sum = mode_probabilities.sum()
            if curr_sum <= 0.0:
                raise ValueError(
                    "At least one model probability must be strictly positive."
                )
            if not isclose(curr_sum, 1.0):
                warnings.warn(
                    "mode_probabilities do not sum to one. Renormalizing.",
                    UserWarning,
                )
                mode_probabilities = mode_probabilities / curr_sum
'''
new_modes = '''            mode_probabilities, probabilities_sum_to_one = (
                _normalize_nonnegative_weights(
                    mode_probabilities,
                    "At least one model probability must be strictly positive.",
                )
            )
            if not probabilities_sum_to_one:
                warnings.warn(
                    "mode_probabilities do not sum to one. Renormalizing.",
                    UserWarning,
                )
'''
source = replace_once(source, old_modes, new_modes)

old_moment = '''        if not bool(pyrecest.backend.all(isfinite(weights))):
            raise ValueError("weights must be finite.")
        curr_sum = weights.sum()
        if curr_sum <= 0.0:
            raise ValueError("At least one mixture weight must be strictly positive.")
        if not isclose(curr_sum, 1.0):
            weights = weights / curr_sum
'''
new_moment = '''        if not bool(pyrecest.backend.all(isfinite(weights))):
            raise ValueError("weights must be finite.")
        if pyrecest.backend.any(weights < 0.0):
            raise ValueError("weights must be nonnegative.")
        weights, _ = _normalize_nonnegative_weights(
            weights,
            "At least one mixture weight must be strictly positive.",
        )
'''
source = replace_once(source, old_moment, new_moment)
source_path.write_text(source, encoding="utf-8")

test_path = Path("tests/filters/test_imm_extreme_weight_normalization.py")
test_path.write_text(
    '''import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class _GaussianStateHolder:
    def __init__(self, mean):
        self.filter_state = GaussianDistribution(
            array([mean]),
            array([[1.0]]),
            check_validity=False,
        )


def _filter_bank():
    return [_GaussianStateHolder(0.0), _GaussianStateHolder(2.0)]


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme binary64 regression is specific to the NumPy backend",
)
def test_extreme_finite_mode_probabilities_preserve_ratios():
    largest = np.finfo(np.float64).max

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        with pytest.warns(UserWarning, match="mode_probabilities"):
            imm = InteractingMultipleModelFilter(
                _filter_bank(),
                transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
                mode_probabilities=array([largest, largest / 2.0]),
            )

    npt.assert_allclose(imm.mode_probabilities, array([2.0 / 3.0, 1.0 / 3.0]))
    npt.assert_allclose(imm.get_point_estimate(), array([2.0 / 3.0]))


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme binary64 regression is specific to the NumPy backend",
)
def test_extreme_finite_transition_rows_remain_stochastic():
    largest = np.finfo(np.float64).max

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        with pytest.warns(UserWarning, match="transition_matrix"):
            imm = InteractingMultipleModelFilter(
                _filter_bank(),
                transition_matrix=array(
                    [[largest, largest], [largest, 0.0]],
                ),
                mode_probabilities=array([0.25, 0.75]),
            )
        mixing = imm.interact()

    npt.assert_allclose(
        imm.transition_matrix,
        array([[0.5, 0.5], [1.0, 0.0]]),
    )
    npt.assert_allclose(imm.mode_probabilities, array([0.875, 0.125]))
    npt.assert_allclose(mixing.sum(axis=0), array([1.0, 1.0]))


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme binary64 regression is specific to the NumPy backend",
)
def test_moment_matching_normalizes_extreme_weights_and_rejects_negative_weights():
    largest = np.finfo(np.float64).max
    gaussians = [holder.filter_state for holder in _filter_bank()]

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        matched = InteractingMultipleModelFilter._moment_match_gaussians(
            gaussians,
            array([largest, largest]),
        )

    npt.assert_allclose(matched.mu, array([1.0]))
    npt.assert_allclose(matched.C, array([[2.0]]))

    with pytest.raises(ValueError, match="nonnegative"):
        InteractingMultipleModelFilter._moment_match_gaussians(
            gaussians,
            array([-1.0, 2.0]),
        )
''',
    encoding="utf-8",
)
