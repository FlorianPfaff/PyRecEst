import unittest
import warnings

import numpy.testing as npt
import pyrecest
from pyrecest.backend import (  # pylint: disable=redefined-builtin
    array,
    column_stack,
    ones,
    pi,
    sum,
    tile,
    zeros,
)
from pyrecest.distributions.conditional.s2_cond_s2_grid_distribution import (
    S2CondS2GridDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)


def _skip_jax(test_fn):
    return unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )(test_fn)


class TestS2CondS2GridDistributionInit(unittest.TestCase):
    """Basic construction and validation."""

    @_skip_jax
    def test_basic_construction(self):
        no_grid_points = 50

        def uniform_trans(xkk, xk):
            # xkk: (n1, 3), xk: (n2, 3)  -> (n1, n2)
            from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_distribution import (
                AbstractHypersphereSubsetDistribution,
            )

            surface = (
                AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                    2
                )
            )
            return ones((xkk.shape[0], xk.shape[0])) / surface

        s2s2 = S2CondS2GridDistribution.from_function(
            uniform_trans, no_grid_points, True
        )
        self.assertEqual(s2s2.dim, 6)
        self.assertEqual(s2s2.grid.shape[1], 3)
        self.assertEqual(s2s2.grid_values.shape, (no_grid_points, no_grid_points))

    @_skip_jax
    def test_wrong_grid_dim_raises(self):
        from pyrecest.sampling.hyperspherical_sampler import get_grid_hypersphere

        # Build a 2-sphere grid and misshape it to 4D
        grid, _ = get_grid_hypersphere("leopardi", 10, 2)
        n = grid.shape[0]

        surface = 4 * pi
        gv = ones((n, n)) / surface
        # Simulate a non-S2 grid (embed in 4D instead of 3D) - should raise

        grid_4d = column_stack([grid, zeros(n)])
        with self.assertRaises(ValueError):
            S2CondS2GridDistribution(grid_4d, gv)


class TestS2CondS2GridDistributionFromFunction(unittest.TestCase):
    """Tests mirroring the MATLAB S2CondS2GridDistributionTest class."""

    @_skip_jax
    def test_warning_free_normalized_vmf(self):
        """testWarningFreeNormalizedVMF: VMF-based conditional should warn-free."""
        no_grid_points = 112

        def trans(xkk, xk):
            # xkk: (n1, 3), xk: (n2, 3) -> (n1, n2)
            result = zeros((xkk.shape[0], xk.shape[0]))
            for i in range(xk.shape[0]):
                vmf = VonMisesFisherDistribution(xk[i], 1.0)
                result[:, i] = vmf.pdf(xkk)
            return result

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            S2CondS2GridDistribution.from_function(
                trans, no_grid_points, True, "leopardi"
            )

    @_skip_jax
    def test_warning_unnormalized(self):
        """testWarningUnnormalized: unnormalized transition should emit UserWarning."""
        no_grid_points = 112

        def trans(xkk, xk):
            # xkk, xk both (n_pairs, 3) when fun_does_cartesian_product=False
            D = array([0.1, 0.15, 1.0])
            diff = (xkk - xk) * D[None, :]
            return 1.0 / (sum(diff**2, axis=1) + 0.01)

        with self.assertWarns(UserWarning):
            S2CondS2GridDistribution.from_function(
                trans, no_grid_points, False, "leopardi"
            )

    @_skip_jax
    def test_warning_free_custom_normalized(self):
        """testWarningFreeCustomNormalized: manually normalized transition should be warn-free."""
        no_grid_points = 1000

        def trans(xkk, xk):
            # xkk: (n1, 3), xk: (n2, 3) -> (n1, n2)  (cartesian product mode)
            from pyrecest.distributions.hypersphere_subset.custom_hyperspherical_distribution import (
                CustomHypersphericalDistribution,
            )

            D = array([0.1, 0.15, 0.3])

            def trans_unnorm(pts, fixed):
                diff = (pts - fixed[None, :]) * D[None, :]
                return 1.0 / (sum(diff**2, axis=1) + 0.01)

            p = zeros((xkk.shape[0], xk.shape[0]))
            for i in range(xk.shape[0]):
                chd = CustomHypersphericalDistribution(
                    lambda pts, fi=xk[i]: trans_unnorm(pts, fi), 2
                )
                norm_const = chd.integrate_numerically()
                p[:, i] = trans_unnorm(xkk, xk[i]) / norm_const
            return p

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            S2CondS2GridDistribution.from_function(
                trans, no_grid_points, True, "leopardi"
            )

    @_skip_jax
    def test_equal_with_and_without_cart(self):
        """testEqualWithAndWithoutCart: Cartesian and non-Cartesian modes should agree."""
        no_grid_points = 100
        dist = VonMisesFisherDistribution(array([0.0, -1.0, 0.0]), 100.0)

        def f_trans1(xkk, xk):
            vals = dist.pdf(xkk)  # (n1,)
            return tile(vals[:, None], (1, xk.shape[0]))  # (n1, n2)

        def f_trans2(xkk, _xk):
            return dist.pdf(xkk)  # (n_pairs,) in non-Cartesian mode

        s2s2_1 = S2CondS2GridDistribution.from_function(f_trans1, no_grid_points, True)
        s2s2_2 = S2CondS2GridDistribution.from_function(f_trans2, no_grid_points, False)

        npt.assert_array_equal(s2s2_1.grid, s2s2_2.grid)
        npt.assert_allclose(s2s2_1.grid_values, s2s2_2.grid_values, rtol=1e-10)

    @_skip_jax
    def test_fix_dim_returns_spherical_grid_distribution(self):
        """fix_dim should return SphericalGridDistribution instances."""
        no_grid_points = 50

        def trans(xkk, xk):
            result = zeros((xkk.shape[0], xk.shape[0]))
            for i in range(xk.shape[0]):
                vmf = VonMisesFisherDistribution(xk[i], 1.0)
                result[:, i] = vmf.pdf(xkk)
            return result

        s2s2 = S2CondS2GridDistribution.from_function(
            trans, no_grid_points, True, "leopardi"
        )

        point = s2s2.grid[0]
        sgd1 = s2s2.fix_dim(1, point)
        sgd2 = s2s2.fix_dim(2, point)
        self.assertIsInstance(sgd1, SphericalGridDistribution)
        self.assertIsInstance(sgd2, SphericalGridDistribution)

    @_skip_jax
    def test_fix_dim_mean_direction(self):
        """
        testFixDim: fixing dim 2 at a grid point and computing mean_direction
        should give back the conditioning point (approx).
        """
        no_grid_points = 112

        def trans(xkk, xk):
            result = zeros((xkk.shape[0], xk.shape[0]))
            for i in range(xk.shape[0]):
                vmf = VonMisesFisherDistribution(xk[i], 1.0)
                result[:, i] = vmf.pdf(xkk)
            return result

        s2s2 = S2CondS2GridDistribution.from_function(
            trans, no_grid_points, True, "leopardi"
        )

        for point in s2s2.grid:
            sgd = s2s2.fix_dim(2, point)
            npt.assert_allclose(sgd.mean_direction(), point, atol=1e-1)

    @_skip_jax
    def test_marginalize_out_returns_spherical_grid_distribution(self):
        """marginalize_out should return SphericalGridDistribution."""
        no_grid_points = 50

        def trans(xkk, xk):
            result = zeros((xkk.shape[0], xk.shape[0]))
            for i in range(xk.shape[0]):
                vmf = VonMisesFisherDistribution(xk[i], 1.0)
                result[:, i] = vmf.pdf(xkk)
            return result

        s2s2 = S2CondS2GridDistribution.from_function(
            trans, no_grid_points, True, "leopardi"
        )
        sgd1 = s2s2.marginalize_out(1)
        sgd2 = s2s2.marginalize_out(2)
        self.assertIsInstance(sgd1, SphericalGridDistribution)
        self.assertIsInstance(sgd2, SphericalGridDistribution)


if __name__ == "__main__":
    unittest.main()
