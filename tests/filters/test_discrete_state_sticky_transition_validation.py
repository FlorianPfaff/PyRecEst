import numpy as np
import pytest
from pyrecest.filters.discrete_state import sticky_mode_transition_matrix


@pytest.mark.parametrize(
    "stickiness",
    [
        True,
        False,
        "0.5",
        0.5 + 0j,
        np.array([0.5]),
        np.array([0.2, 0.8]),
        np.nan,
        np.inf,
        -np.inf,
        -0.1,
        1.1,
    ],
)
def test_sticky_mode_transition_matrix_rejects_invalid_stickiness(stickiness):
    with pytest.raises(ValueError, match="stickiness"):
        sticky_mode_transition_matrix(3, stickiness)


@pytest.mark.parametrize(
    "stickiness",
    [0.0, 1.0, 0.25, np.float64(0.75), np.array(0.5)],
)
def test_sticky_mode_transition_matrix_accepts_real_scalar_stickiness(stickiness):
    parsed = float(np.asarray(stickiness).item())
    result = sticky_mode_transition_matrix(3, stickiness)

    expected = np.full((3, 3), (1.0 - parsed) / 2.0)
    np.fill_diagonal(expected, parsed)
    np.testing.assert_allclose(result, expected)
