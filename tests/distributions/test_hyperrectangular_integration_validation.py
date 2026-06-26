from __future__ import annotations

import numpy as np
import pytest
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)


def _unit_square_distribution() -> HyperrectangularUniformDistribution:
    return HyperrectangularUniformDistribution([[0.0, 1.0], [0.0, 1.0]])


def test_integrate_rejects_nonfinite_integration_boundaries() -> None:
    distribution = _unit_square_distribution()

    with pytest.raises(ValueError, match="finite"):
        distribution.integrate([[0.0, np.inf], [0.0, 1.0]])


def test_integrate_rejects_reversed_integration_boundaries() -> None:
    distribution = _unit_square_distribution()

    with pytest.raises(ValueError, match="strictly increasing"):
        distribution.integrate([[1.0, 0.0], [0.0, 1.0]])


def test_integrate_preserves_flat_boundary_input_contract() -> None:
    distribution = _unit_square_distribution()

    assert distribution.integrate([0.0, 0.5, 0.0, 0.5]) == pytest.approx(0.25)
