import unittest

import numpy.testing as npt
from pyrecest.backend import (  # pylint: disable=redefined-builtin
    abs,
    array,
    asarray,
    exp,
    linspace,
    minimum,
    ones,
    pi,
    random,
    zeros,
)
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)


def _make_normalized_grid_values(n: int):
    """Return an (n x n) matrix whose columns are normalized (integrate to 1)."""
    random.seed(0)
    gv = 0.5 + random.uniform(size=(n, n))
    # Normalize each column so that mean(col) * (2*pi)^1 == 1
    gv = gv / (gv.mean(axis=0) * 2.0 * pi)
    return gv


class TdCondTdGridDistributionTest(unittest.TestCase):
    # -------------------------------------------------------------- construction

    def test_construction_t1(self):
        """Basic construction for T1 x T1."""
        n = 5
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid, gv)
        self.assertEqual(td.dim, 2)
        npt.assert_allclose(td.grid, grid, rtol=1e-6)
        npt.assert_allclose(td.grid_values, gv, rtol=1e-6)

    def test_construction_wrong_shape_raises(self):
        n = 4
        grid = zeros((n, 1))
        with self.assertRaises(ValueError):
            # Non-square grid_values
            TdCondTdGridDistribution(grid, ones((n, n + 1)) / (n * 2 * pi))

    def test_construction_wrong_order_raises(self):
        """Transposed (row-normalized) matrix should raise an error."""
        n = 6
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        # Transpose → rows are normalized, columns are not
        with self.assertRaises(ValueError):
            TdCondTdGridDistribution(grid, gv.T)

    def test_construction_unnormalized_warns(self):
        """An unnormalized matrix that cannot be fixed by transposing should warn."""
        n = 5
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = ones((n, n))  # neither rows nor cols sum to 1/(2pi)
        with self.assertWarns(UserWarning):
            TdCondTdGridDistribution(grid, gv)

    # -------------------------------------------------------------- normalize

    def test_normalize_returns_self(self):
        n = 4
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid, gv)
        self.assertIs(td.normalize(), td)

    # -------------------------------------------------------------- multiply

    def test_multiply_same_grid(self):
        import warnings

        n = 6
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td1 = TdCondTdGridDistribution(grid, gv)
        td2 = TdCondTdGridDistribution(grid, gv)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = td1.multiply(td2)
        npt.assert_allclose(
            asarray(result.grid_values),
            asarray(td1.grid_values) * asarray(td2.grid_values),
            rtol=1e-10,
        )

    def test_multiply_incompatible_grid_raises(self):
        n1, n2 = 4, 6
        grid1 = linspace(0.0, 2.0 * pi - 2.0 * pi / n1, n1).reshape(-1, 1)
        grid2 = linspace(0.0, 2.0 * pi - 2.0 * pi / n2, n2).reshape(-1, 1)
        gv1 = _make_normalized_grid_values(n1)
        gv2 = _make_normalized_grid_values(n2)
        td1 = TdCondTdGridDistribution(grid1, gv1)
        td2 = TdCondTdGridDistribution(grid2, gv2)
        with self.assertRaises(ValueError) as ctx:
            td1.multiply(td2)
        self.assertIn("IncompatibleGrid", str(ctx.exception))

    # -------------------------------------------------------------- from_function

    def test_from_function_t1(self):
        """from_function should recover a wrapped-normal-like conditional."""
        n = 20
        dim = 2  # T1 x T1

        def cond_fun(a, b):
            # Simple Gaussian-like conditional (unnormalized, normalized per column)
            diff = asarray(a)[:, 0] - asarray(b)[:, 0]
            return exp(-0.5 * minimum(diff**2, (2 * pi - abs(diff)) ** 2))

        td = TdCondTdGridDistribution.from_function(
            cond_fun,
            n,
            fun_does_cartesian_product=False,
            grid_type="CartesianProd",
            dim=dim,
        )
        self.assertIsInstance(td, TdCondTdGridDistribution)
        self.assertEqual(td.dim, dim)
        self.assertEqual(asarray(td.grid_values).shape, (n, n))
        self.assertEqual(asarray(td.grid).shape, (n, 1))

    def test_from_function_cartesian_product_flag(self):
        """from_function with fun_does_cartesian_product=True."""
        n = 8
        dim = 2

        def cond_fun_cp(a, b):
            # a: (n, 1), b: (n, 1) → return (n, n)
            a_arr = asarray(a)[:, 0]
            b_arr = asarray(b)[:, 0]
            diff = a_arr[:, None] - b_arr[None, :]
            return exp(-0.5 * minimum(diff**2, (2 * pi - abs(diff)) ** 2))

        td = TdCondTdGridDistribution.from_function(
            cond_fun_cp,
            n,
            fun_does_cartesian_product=True,
            grid_type="CartesianProd",
            dim=dim,
        )
        self.assertIsInstance(td, TdCondTdGridDistribution)
        self.assertEqual(asarray(td.grid_values).shape, (n, n))

    def test_from_function_unknown_grid_raises(self):
        n = 4
        with self.assertRaises(ValueError):
            TdCondTdGridDistribution.from_function(
                lambda a, b: ones(len(a)),
                n,
                fun_does_cartesian_product=False,
                grid_type="unknownGrid",
                dim=2,
            )

    def test_from_function_odd_dim_raises(self):
        with self.assertRaises(ValueError):
            TdCondTdGridDistribution.from_function(
                lambda a, b: ones(len(a)), 4, False, "CartesianProd", dim=3
            )

    # --------------------------------------------------------- marginalize_out

    def test_marginalize_out_returns_hgd(self):
        n = 6
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid, gv)

        for first_or_second in (1, 2):
            with self.subTest(first_or_second=first_or_second):
                marginal = td.marginalize_out(first_or_second)
                self.assertIsInstance(marginal, HypertoroidalGridDistribution)
                self.assertEqual(marginal.dim, 1)

    def test_marginalize_out_sums(self):
        n = 5
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid, gv)

        # marginalize_out(1) sums rows; HGD normalizes, so check proportionality
        m1 = td.marginalize_out(1)
        expected_unnorm = gv.sum(axis=0)
        actual_unnorm = asarray(m1.grid_values) * float(m1.integrate())
        # Check proportionality (ratio should be constant)
        ratio = actual_unnorm / expected_unnorm
        npt.assert_allclose(ratio, ratio[0] * ones(n), atol=1e-12)

        # marginalize_out(2) sums cols
        m2 = td.marginalize_out(2)
        expected_unnorm2 = gv.sum(axis=1)
        actual_unnorm2 = asarray(m2.grid_values) * float(m2.integrate())
        ratio2 = actual_unnorm2 / expected_unnorm2
        npt.assert_allclose(ratio2, ratio2[0] * ones(n), atol=1e-12)

    def test_marginalize_out_invalid_raises(self):
        n = 5
        grid = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid, gv)
        with self.assertRaises(ValueError):
            td.marginalize_out(0)
        with self.assertRaises(ValueError):
            td.marginalize_out(3)

    # -------------------------------------------------------------- fix_dim

    def test_fix_dim_returns_hgd(self):
        n = 5
        grid_np = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid_np, gv)

        for first_or_second in (1, 2):
            with self.subTest(first_or_second=first_or_second):
                point = grid_np[2]  # third grid point
                result = td.fix_dim(first_or_second, point)
                self.assertIsInstance(result, HypertoroidalGridDistribution)
                self.assertEqual(result.dim, 1)

    def test_fix_dim_off_grid_raises(self):
        n = 5
        grid_np = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid_np, gv)
        with self.assertRaises(ValueError):
            td.fix_dim(1, array([1.23456789]))

    def test_fix_dim_values_correct(self):
        """fix_dim(2, grid[j]) should give a distribution proportional to col j."""
        n = 6
        grid_np = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid_np, gv)

        j = 3
        slice_dist = td.fix_dim(2, grid_np[j])
        expected = gv[:, j]
        expected = expected / expected.mean() / (2.0 * pi)  # normalize
        npt.assert_allclose(
            asarray(slice_dist.grid_values),
            expected,
            atol=1e-12,
        )

    def test_fix_dim_invalid_raises(self):
        n = 5
        grid_np = linspace(0.0, 2.0 * pi - 2.0 * pi / n, n).reshape(-1, 1)
        gv = _make_normalized_grid_values(n)
        td = TdCondTdGridDistribution(grid_np, gv)
        point = grid_np[0]
        with self.assertRaises(ValueError):
            td.fix_dim(0, point)
        with self.assertRaises(ValueError):
            td.fix_dim(3, point)

    # --------------------------------------------------------- from_function + fix_dim round-trip

    def test_from_function_fix_dim_roundtrip(self):
        """fix_dim on a from_function object should return a HypertoroidalGridDistribution."""
        n = 10
        dim = 2

        def cond_fun(a, b):
            diff = asarray(a)[:, 0] - asarray(b)[:, 0]
            return exp(-0.5 * diff**2)

        td = TdCondTdGridDistribution.from_function(
            cond_fun, n, fun_does_cartesian_product=False, dim=dim
        )
        grid_np = asarray(td.grid)
        slice_dist = td.fix_dim(2, grid_np[0])
        self.assertIsInstance(slice_dist, HypertoroidalGridDistribution)


if __name__ == "__main__":
    unittest.main()
