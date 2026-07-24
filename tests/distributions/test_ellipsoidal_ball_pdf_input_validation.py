import numpy as np
import pytest
from pyrecest.distributions import EllipsoidalBallUniformDistribution
from pyrecest.exceptions import ValidationError


@pytest.fixture
def distribution():
    return EllipsoidalBallUniformDistribution(
        center=np.zeros(2),
        shape_matrix=np.eye(2),
    )


@pytest.mark.parametrize(
    "xs",
    [
        [np.nan, 0.0],
        [np.inf, 0.0],
        [[0.0, 0.0], [0.0, -np.inf]],
    ],
)
def test_pdf_rejects_nonfinite_points(distribution, xs):
    with pytest.raises(ValidationError, match="finite"):
        distribution.pdf(xs)


def test_pdf_rejects_textual_points(distribution):
    with pytest.raises(ValidationError, match="finite real-valued"):
        distribution.pdf(["0", "1"])
