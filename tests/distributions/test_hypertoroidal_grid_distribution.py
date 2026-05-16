import unittest

import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, max, min, pi, random
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


class HypertoroidalGridDistributionTest(unittest.TestCase):

    def test_get_grid(self):
        grid = array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]).T
        hgd = HypertoroidalGridDistribution(
            array([[1.0, 1.0], [1.0, 1.09]]) / ((2.0 * pi) ** 2), grid=grid
        )
        npt.assert_allclose(hgd.get_grid(), grid)
        npt.assert_allclose(hgd.get_grid().shape, (4, 2))

    def test_custom_flat_grid_honors_explicit_dim(self):
        grid = array([[0.0, 0.0], [pi, 0.0], [0.0, pi], [pi, pi]])
        grid_values = array([1.0, 1.0, 1.0, 1.0]) / ((2.0 * pi) ** 2)

        hgd = HypertoroidalGridDistribution(grid_values, grid=grid, dim=2)

        self.assertEqual(hgd.dim, 2)
        npt.assert_allclose(hgd.get_grid(), grid)

    def test_custom_flat_grid_infers_dim_from_grid(self):
        grid = array([[0.0, 0.0], [pi, 0.0], [0.0, pi], [pi, pi]])
        grid_values = array([1.0, 1.0, 1.0, 1.0]) / ((2.0 * pi) ** 2)

        hgd = HypertoroidalGridDistribution(grid_values, grid=grid)

        self.assertEqual(hgd.dim, 2)
        npt.assert_allclose(hgd.get_grid(), grid)

    def test_pdf_custom_grid_flattens_grid_values_for_nearest_neighbor(self):
        grid = array(
            [
                [0.0, 0.0],
                [0.0, pi],
                [pi, 0.0],
                [pi, pi],
            ]
        )
        normalizer = ((2.0 * pi) ** 2) * 2.5
        grid_values = array([[1.0, 2.0], [3.0, 4.0]]) / normalizer
        hgd = HypertoroidalGridDistribution(grid_values, grid=grid)

        query_points = array([[pi, 0.0], [pi, pi]])
        expected = array([3.0, 4.0]) / normalizer
        npt.assert_allclose(hgd.pdf(query_points), expected, rtol=2e-7)

    def test_custom_grid_uses_elementwise_toroidal_distance(self):
        grid = array(
            [
                [0.0, 0.0],
                [0.0, pi],
                [pi, 0.0],
                [pi, pi],
            ]
        )
        normalizer = ((2.0 * pi) ** 2) * 2.5
        grid_values = array([[4.0, 1.0], [2.0, 3.0]]) / normalizer
        hgd = HypertoroidalGridDistribution(grid_values, grid=grid)

        query = array([[2.0 * pi - 0.1, 0.05]])
        npt.assert_allclose(hgd.get_closest_point(query), array([0.0, 0.0]))
        npt.assert_allclose(hgd.pdf(query), array([4.0]) / normalizer)

    def test_custom_one_dimensional_flat_grid_is_reshaped_to_column_grid(self):
        grid = array([0.0, pi, 2.0 * pi - 0.2])
        # Mean is 1 / (2*pi), so the 1-D torus integral is already one.
        grid_values = array([1.0, 2.0, 3.0]) / (4.0 * pi)

        hgd = HypertoroidalGridDistribution(grid_values, grid=grid)

        self.assertEqual(hgd.dim, 1)
        npt.assert_allclose(
            hgd.get_grid(), array([[0.0], [pi], [2.0 * pi - 0.2]])
        )

        query_points = array([[2.0 * pi - 0.3], [pi + 0.1], [0.1]])
        expected = array([3.0, 2.0, 1.0]) / (4.0 * pi)

        npt.assert_allclose(hgd.pdf(query_points), expected)
        npt.assert_allclose(
            hgd.get_closest_point(array([2.0 * pi - 0.3])),
            array([2.0 * pi - 0.2]),
        )

    def test_approx_vmmixture_t2(self):
        dist = HypertoroidalMixture(
            [
                ToroidalWrappedNormalDistribution(
                    array([1.0, 1.0]), array([[1.0, 0.5], [0.5, 1.0]])
                ),
                ToroidalWrappedNormalDistribution(
                    array([3.0, 3.0]), array([[1.0, -0.5], [-0.5, 1.0]])
                ),
            ],
            array([0.5, 0.5]),
        )

        hgd = HypertoroidalGridDistribution.from_distribution(dist, (31, 31))
        npt.assert_allclose(
            hgd.grid_values.reshape((-1,)), dist.pdf(hgd.get_grid()), atol=6
        )
        npt.assert_allclose(min(hgd.get_grid(), 0), array([0, 0]))
        npt.assert_allclose(
            max(hgd.get_grid(), 0), array([30 / 31 * 2 * pi, 30 / 31 * 2 * pi])
        )
        self.assertEqual(hgd.grid_type, "cartesian_prod")

    def test_from_function_3D(self):
        random.seed(0)
        test_points = 2 * pi * random.uniform(size=(30, 3))
        C = array([[2, 0.4, 0.2], [0.4, 2, 0.1], [0.2, 0.1, 3]])
        mu = 2.0 * pi * random.uniform(size=(3,))
        hwnd = HypertoroidalWrappedNormalDistribution(mu, C)
        n_grid_points = [21, 21, 21]
        hfd_id = HypertoroidalGridDistribution.from_function(
            hwnd.pdf, n_grid_points, "cartesian_prod"
        )
        self.assertIsInstance(hfd_id, HypertoroidalGridDistribution)
        npt.assert_allclose(hfd_id.pdf(test_points), hwnd.pdf(test_points), rtol=0.3)


if __name__ == "__main__":
    unittest.main()
