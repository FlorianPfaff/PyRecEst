import numpy as np
import pytest
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("enforce_pdf_nonnegative", [False, True])
def test_circular_grid_distribution_rejects_nonfinite_grid_values(
    invalid_value, enforce_pdf_nonnegative
):
    grid_values = np.array([0.25, invalid_value, 0.5])

    with pytest.raises(ValueError, match="finite"):
        CircularGridDistribution(
            grid_values,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )
