import numpy as np
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution


class _ConcreteGridDistribution(AbstractGridDistribution):
    def get_closest_point(self, xs):
        raise NotImplementedError

    def get_manifold_size(self):
        return 1.0


def test_multidimensional_grid_values_report_total_point_count():
    grid_values = np.ones((2, 3))
    grid = np.zeros((6, 2))

    distribution = _ConcreteGridDistribution(
        grid_values,
        grid_type="custom",
        grid=grid,
        dim=2,
    )

    assert distribution.n_grid_points == 6
    assert distribution.grid_density_description["n_grid_values"] == 6
