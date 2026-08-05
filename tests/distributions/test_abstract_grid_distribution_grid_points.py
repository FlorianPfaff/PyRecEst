import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution


class _OneDimensionalGridDistribution(AbstractGridDistribution):
    def __init__(self):
        super().__init__(
            grid_values=array([1.0, 1.0, 1.0]),
            grid_type="custom",
            grid=array([0.0, 0.5, 1.0]),
            dim=1,
        )

    def get_closest_point(self, xs):
        raise NotImplementedError

    def get_manifold_size(self):
        return 1.0


def test_default_get_grid_point_supports_one_dimensional_coordinates():
    distribution = _OneDimensionalGridDistribution()

    npt.assert_allclose(distribution.get_grid_point(1), 0.5)
    npt.assert_allclose(
        distribution.get_grid_point(array([0, 2])),
        array([0.0, 1.0]),
    )
