import numpy as np
import pytest
from pyrecest.distributions.hypertorus._input_validation import (
    as_hypertoroidal_points,
    as_shift_vector,
)
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


@pytest.mark.parametrize(
    ("value", "dim"),
    [
        (0.25 + 0.5j, 1),
        ([0.25 + 0.5j], 1),
        ([0.0, 0.25 + 0.5j], 2),
        (np.array([0.0, 0.25 + 0.5j], dtype=object), 2),
    ],
)
def test_shift_vector_rejects_complex_angles(value, dim):
    with pytest.raises(ValueError, match="complex"):
        as_shift_vector(value, dim)


@pytest.mark.parametrize(
    ("value", "dim"),
    [
        (0.25 + 0.5j, 1),
        ([0.25 + 0.5j], 1),
        ([0.0, 0.25 + 0.5j], 2),
        (np.array([[0.0, 0.25 + 0.5j]], dtype=object), 2),
    ],
)
def test_hypertoroidal_points_reject_complex_angles(value, dim):
    with pytest.raises(ValueError, match="complex"):
        as_hypertoroidal_points(value, dim)


def test_wrapped_normal_pdf_rejects_complex_queries_before_wrapping():
    distribution = HypertoroidalWrappedNormalDistribution([0.0], [[1.0]])

    with pytest.raises(ValueError, match="complex"):
        distribution.pdf([0.25 + 0.5j])
