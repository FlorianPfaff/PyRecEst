import numpy as np
import pytest
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)


@pytest.mark.parametrize(
    "xs",
    [
        True,
        "0.1",
        0.1 + 0.2j,
        np.nan,
        np.inf,
        np.datetime64("2026-07-23"),
        np.ma.array([0.1], mask=[True]),
        [0.1, False],
    ],
)
def test_pdf_rejects_non_real_or_non_finite_inputs(xs):
    distribution = HypertoroidalUniformDistribution(1)

    with pytest.raises(ValueError, match="xs must contain only finite real values"):
        distribution.pdf(xs)


@pytest.mark.parametrize(
    "shift_by",
    [
        [0.1, True],
        [0.1, "0.2"],
        [0.1, 0.2 + 0.3j],
        [0.1, np.nan],
        [0.1, np.inf],
    ],
)
def test_shift_rejects_non_real_or_non_finite_values(shift_by):
    distribution = HypertoroidalUniformDistribution(2)

    with pytest.raises(
        ValueError, match="shift_by must contain only finite real values"
    ):
        distribution.shift(shift_by)


@pytest.mark.parametrize(
    ("left", "right", "invalid_name"),
    [
        (np.nan, 1.0, "left"),
        (0.0, np.inf, "right"),
        (0.0 + 1.0j, 1.0, "left"),
        (0.0, "1.0", "right"),
        (np.datetime64("2026-07-23"), 1.0, "left"),
    ],
)
def test_integrate_rejects_non_real_or_non_finite_boundaries(
    left, right, invalid_name
):
    distribution = HypertoroidalUniformDistribution(1)

    with pytest.raises(
        ValueError,
        match=rf"{invalid_name} must contain only finite real values",
    ):
        distribution.integrate((left, right))
