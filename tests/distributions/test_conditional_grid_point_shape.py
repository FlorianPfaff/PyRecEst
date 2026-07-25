import numpy as np
import pytest
from pyrecest.backend import array
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)


def test_fix_dim_rejects_column_point():
    density = 1.0 / (2.0 * np.pi) ** 2
    distribution = TdCondTdGridDistribution(
        array([[0.0, 0.0], [1.0, 1.0]]),
        array([[density, density], [density, density]]),
    )

    with pytest.raises(ValueError, match=r"point must have shape \(2,\)"):
        distribution.fix_dim(1, array([[0.0], [0.0]]))
