import unittest

from pyrecest.backend import array
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution

_DEFAULT_GRID = object()


class DummyGridDistribution(AbstractGridDistribution):
    def __init__(
        self,
        grid_values,
        grid_type="custom",
        grid=_DEFAULT_GRID,
        dim=1,
        enforce_pdf_nonnegative=True,
    ):
        if grid is _DEFAULT_GRID:
            grid = array([[0.0], [1.0]])
        super().__init__(
            grid_values,
            grid_type=grid_type,
            grid=grid,
            dim=dim,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )

    def get_closest_point(self, xs):
        raise NotImplementedError

    def get_manifold_size(self):
        return 1.0


class AbstractGridDistributionTest(unittest.TestCase):
    def test_constructor_rejects_custom_grid_without_coordinates(self):
        with self.assertRaisesRegex(ValueError, "Custom grids"):
            DummyGridDistribution(array([1.0, 1.0]), grid=None)

    def test_constructor_rejects_grid_value_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "number of grid coordinates"):
            DummyGridDistribution(array([1.0, 1.0]), grid=array([[0.0]]))

    def test_constructor_rejects_grid_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension 1"):
            DummyGridDistribution(
                array([1.0, 1.0]), grid=array([[0.0, 0.0], [1.0, 1.0]])
            )

    def test_integrate_rejects_custom_boundaries(self):
        dist = DummyGridDistribution(array([1.0, 1.0]))

        with self.assertRaisesRegex(NotImplementedError, "boundaries"):
            dist.integrate((0.0, 1.0))

    def test_multiply_rejects_incompatible_grids(self):
        dist = DummyGridDistribution(array([1.0, 1.0]))

        with self.assertRaisesRegex(TypeError, "AbstractGridDistribution"):
            dist.multiply(object())

        with self.assertRaisesRegex(ValueError, "enforce_pdf_nonnegative"):
            dist.multiply(
                DummyGridDistribution(array([1.0, 1.0]), enforce_pdf_nonnegative=False)
            )

        with self.assertRaisesRegex(ValueError, "Grid value shapes"):
            dist.multiply(DummyGridDistribution(array([[1.0, 1.0]])))

        with self.assertRaisesRegex(ValueError, "Grid types"):
            dist.multiply(
                DummyGridDistribution(array([1.0, 1.0]), grid_type="cartesian_prod")
            )

        with self.assertRaisesRegex(ValueError, "Grid coordinates"):
            dist.multiply(
                DummyGridDistribution(array([1.0, 1.0]), grid=array([[0.0], [2.0]]))
            )

    def test_multiply_combines_compatible_values(self):
        dist = DummyGridDistribution(array([1.0, 1.0]))
        other = DummyGridDistribution(array([2.0, 2.0]))

        result = dist.multiply(other)

        self.assertEqual(result.grid_values.shape, (2,))

    def test_normalize_rejects_near_zero_integral_even_with_negative_values(self):
        dist = DummyGridDistribution(array([1.0, -1.0]))

        with self.assertRaisesRegex(ValueError, "too close to zero"):
            dist.normalize_in_place(warn_unnorm=False)
