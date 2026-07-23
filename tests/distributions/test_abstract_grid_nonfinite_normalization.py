import pytest
from pyrecest.backend import array
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution


class _DummyGridDistribution(AbstractGridDistribution):
    def __init__(self, grid_values):
        super().__init__(
            grid_values,
            grid_type="custom",
            grid=array([[0.0], [1.0]]),
            dim=1,
        )

    def get_closest_point(self, xs):
        raise NotImplementedError

    def get_manifold_size(self):
        return 1.0


@pytest.mark.parametrize(
    "grid_values",
    [
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [-float("inf"), 1.0],
    ],
)
def test_normalize_rejects_nonfinite_integral_without_mutating_values(grid_values):
    dist = _DummyGridDistribution(array(grid_values))
    original_values = dist.grid_values

    with pytest.raises(ValueError, match="Integral of grid values must be finite"):
        dist.normalize_in_place(warn_unnorm=False)

    assert dist.grid_values is original_values
