import unittest
import warnings

import numpy.testing as npt
import pyrecest
from pyrecest.backend import array, ones
from pyrecest.distributions.conditional.sd_cond_sd_grid_distribution import (
    SdCondSdGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)


def _make_uniform_conditional(no_of_grid_points=100, dim=6):
    """Helper: build a properly normalized SdCondSdGridDistribution with uniform values."""
    from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

    manifold_dim = dim // 2 - 1
    grid, _ = get_grid_hypersphere("leopardi", no_of_grid_points, manifold_dim)
    d = grid.shape[1]  # embedding dim
    surface = AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
        d - 1
    )
    # Uniform pdf value for each column so that mean * surface == 1
    uniform_val = 1.0 / surface
    grid_values = ones((no_of_grid_points, no_of_grid_points)) * uniform_val
    return SdCondSdGridDistribution(grid, grid_values)


class TestSdCondSdGridDistributionInit(unittest.TestCase):
    """Tests for __init__ validation."""

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_basic_construction_s2(self):
        """Construct a valid SdCondSdGridDistribution on S2 x S2."""
        sc = _make_uniform_conditional(no_of_grid_points=50)
        self.assertEqual(sc.dim, 6)
        self.assertEqual(sc.grid.shape[1], 3)
        self.assertEqual(sc.grid_values.shape, (50, 50))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_wrong_grid_shape_raises(self):
        """Providing a 1-D grid must raise ValueError."""
        from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

        grid, _ = get_grid_hypersphere("leopardi", 10, 2)
        n = grid.shape[0]
        surface = (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(2)
        )
        gv = ones((n, n)) / surface
        with self.assertRaises(ValueError):
            SdCondSdGridDistribution(grid.ravel(), gv)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_non_square_grid_values_raises(self):
        from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

        grid, _ = get_grid_hypersphere("leopardi", 10, 2)
        n = grid.shape[0]
        with self.assertRaises(ValueError):
            SdCondSdGridDistribution(grid, ones((n, n + 1)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_negative_grid_values_raises(self):
        from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

        grid, _ = get_grid_hypersphere("leopardi", 10, 2)
        n = grid.shape[0]
        surface = (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(2)
        )
        gv = ones((n, n)) / surface
        gv[0, 0] = -1.0
        with self.assertRaises(ValueError):
            SdCondSdGridDistribution(grid, gv)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_unnormalized_warns(self):
        from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

        grid, _ = get_grid_hypersphere("leopardi", 10, 2)
        n = grid.shape[0]
        # Deliberately unnormalized (all ones instead of 1/surface)
        gv = ones((n, n))
        with self.assertWarns(UserWarning):
            SdCondSdGridDistribution(grid, gv)


class TestSdCondSdGridDistributionNormalize(unittest.TestCase):
    """normalize() is a no-op and returns self."""

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_normalize_returns_self(self):
        sc = _make_uniform_conditional(50)
        result = sc.normalize()
        self.assertIs(result, sc)


class TestSdCondSdGridDistributionMultiply(unittest.TestCase):
    """Tests for multiply()."""

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_multiply_same_grid(self):
        sc1 = _make_uniform_conditional(20)
        sc2 = _make_uniform_conditional(20)
        # Force identical grids
        sc2.grid = sc1.grid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sc1.multiply(sc2)
        npt.assert_allclose(
            result.grid_values, sc1.grid_values * sc2.grid_values, rtol=1e-10
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_multiply_incompatible_grid_raises(self):
        sc1 = _make_uniform_conditional(20)
        sc2 = _make_uniform_conditional(30)  # Different number of grid points
        with self.assertRaises(ValueError) as ctx:
            sc1.multiply(sc2)
        self.assertIn("IncompatibleGrid", str(ctx.exception))


class TestSdCondSdGridDistributionMarginalizeOut(unittest.TestCase):
    """Tests for marginalize_out()."""

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_marginalize_out_returns_hgd(self):
        sc = _make_uniform_conditional(20)
        sgd1 = sc.marginalize_out(1)
        sgd2 = sc.marginalize_out(2)
        self.assertIsInstance(sgd1, HypersphericalGridDistribution)
        self.assertIsInstance(sgd2, HypersphericalGridDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_marginalize_out_shape(self):
        n = 20
        sc = _make_uniform_conditional(n)
        sgd1 = sc.marginalize_out(1)
        sgd2 = sc.marginalize_out(2)
        self.assertEqual(sgd1.grid_values.shape, (n,))
        self.assertEqual(sgd2.grid_values.shape, (n,))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_marginalize_out_uniform_is_uniform(self):
        """For a uniform conditional, marginalizing either way gives uniform."""
        n = 30
        sc = _make_uniform_conditional(n)
        sgd1 = sc.marginalize_out(1)
        sgd2 = sc.marginalize_out(2)
        # After normalisation in HypersphericalGridDistribution, all values equal
        npt.assert_allclose(
            sgd1.grid_values,
            sgd1.grid_values[0] * ones(n),
            rtol=1e-10,
        )
        npt.assert_allclose(
            sgd2.grid_values,
            sgd2.grid_values[0] * ones(n),
            rtol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_marginalize_out_invalid_raises(self):
        sc = _make_uniform_conditional(10)
        with self.assertRaises(ValueError):
            sc.marginalize_out(0)
        with self.assertRaises(ValueError):
            sc.marginalize_out(3)


class TestSdCondSdGridDistributionFixDim(unittest.TestCase):
    """Tests for fix_dim()."""

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fix_dim_returns_hgd(self):
        n = 20
        sc = _make_uniform_conditional(n)
        point = sc.grid[0, :]
        sgd1 = sc.fix_dim(1, point)
        sgd2 = sc.fix_dim(2, point)
        self.assertIsInstance(sgd1, HypersphericalGridDistribution)
        self.assertIsInstance(sgd2, HypersphericalGridDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fix_dim_values_correct(self):
        n = 20
        sc = _make_uniform_conditional(n)
        # Fix first dim at grid[5]
        sgd = sc.fix_dim(1, sc.grid[5, :])
        npt.assert_allclose(
            sgd.grid_values,
            sc.fix_dim(1, sc.grid[5, :]).grid_values,
            rtol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fix_dim_off_grid_raises(self):
        sc = _make_uniform_conditional(20)
        # Supply a point that is not on the grid
        off_grid = array([0.1, 0.2, 0.97])  # arbitrary off-grid S2 point
        # Normalise to unit sphere
        from pyrecest.backend import linalg

        off_grid = off_grid / linalg.norm(off_grid)
        with self.assertRaises(ValueError):
            sc.fix_dim(1, off_grid)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_fix_dim_invalid_raises(self):
        sc = _make_uniform_conditional(10)
        point = sc.grid[0, :]
        with self.assertRaises(ValueError):
            sc.fix_dim(0, point)
        with self.assertRaises(ValueError):
            sc.fix_dim(3, point)


class TestSdCondSdGridDistributionFromFunction(unittest.TestCase):
    """Tests for from_function()."""

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_from_function_uniform(self):
        """Build a uniform conditional from a constant function."""
        d = 3  # S2 embedding dim; manifold dim = 2
        surface = (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                d - 1
            )
        )

        def uniform_fun(a, _b):
            return ones(a.shape[0]) / surface

        n = 50
        sc = SdCondSdGridDistribution.from_function(
            uniform_fun,
            n,
            fun_does_cartesian_product=False,
            grid_type="leopardi",
            dim=6,
        )
        self.assertEqual(sc.dim, 6)
        self.assertEqual(sc.grid_values.shape, (n, n))
        # All values should be equal to 1/surface
        npt.assert_allclose(sc.grid_values, ones((n, n)) / surface, rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_from_function_cartesian_product(self):
        """from_function with fun_does_cartesian_product=True."""
        d = 3
        surface = (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                d - 1
            )
        )
        n = 30

        def uniform_fun_cp(grid_a, grid_b):
            # grid_a and grid_b are both (n, d); return (n, n)
            return ones((grid_a.shape[0], grid_b.shape[0])) / surface

        sc = SdCondSdGridDistribution.from_function(
            uniform_fun_cp,
            n,
            fun_does_cartesian_product=True,
            grid_type="leopardi",
            dim=6,
        )
        self.assertEqual(sc.grid_values.shape, (n, n))
        npt.assert_allclose(sc.grid_values, ones((n, n)) / surface, rtol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_from_function_wrong_shape_raises(self):
        """Function that returns wrong shape should raise ValueError."""
        n = 10

        def bad_fun(a, b):
            # Return (n*n, n*n) – as if doing Cartesian product
            return ones((a.shape[0], b.shape[0]))

        with self.assertRaises(ValueError):
            SdCondSdGridDistribution.from_function(
                bad_fun,
                n,
                fun_does_cartesian_product=False,
                grid_type="leopardi",
                dim=6,
            )


if __name__ == "__main__":
    unittest.main()
