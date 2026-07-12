import pytest
from pyrecest.backend import pi, zeros
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)


def test_pdf_preserves_all_leading_batch_dimensions():
    dist = HypertoroidalUniformDistribution(2)

    values = dist.pdf(zeros((2, 3, 2)))

    assert values.shape == (2, 3)
    assert float(values[1, 2]) == pytest.approx(1.0 / (2.0 * pi) ** 2)
