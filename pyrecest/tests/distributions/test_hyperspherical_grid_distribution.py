import unittest
import warnings
import numpy as np

from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions import HypersphericalMixture


from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)


def _grid_for_pdf(grid, dim):
    """
    Return grid in (batch_dim, space_dim) form for pdf evaluation.
    Accepts either:
    - (n_points, dim)  -> returned as-is
    - (dim, n_points)  -> transposed
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {grid.shape}")
    if grid.shape[1] == dim and grid.shape[0] != dim:
        return grid
    if grid.shape[0] == dim and grid.shape[1] != dim:
        return grid.T
    if grid.shape[0] == dim and grid.shape[1] == dim:
        # ambiguous, but return as-is
        return grid
    raise ValueError(
        f"Grid shape {grid.shape} not compatible with dimension {dim}"
    )


def _standardize_grid(grid, dim):
    """Standardize grid orientation to (n_points, dim) for comparisons."""
    return _grid_for_pdf(grid, dim)


class HypersphericalGridDistributionTest(unittest.TestCase):
    # --------------------------------------------------------------
    # Helper: PDF equality on a small cartesian grid (S^2)
    # --------------------------------------------------------------
    def verify_pdf_equal(self, dist1, dist2, tol):
        """
        Compare pdfs of two distributions on a simple grid on S^2.
        """
        # S^2 grid
        phi, theta = np.meshgrid(
            np.linspace(0.0, 2 * np.pi, 10),
            np.linspace(-np.pi / 2, np.pi / 2, 10),
        )
        phi = phi.ravel()
        theta = theta.ravel()
        r = 1.0

        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)

        pts = np.vstack((x, y, z)).T  # (n_points, 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = dist1.pdf(pts)
            p2 = dist2.pdf(pts)

        np.testing.assert_allclose(p1, p2, atol=tol, rtol=0)

    # --------------------------------------------------------------
    # Approximation tests
    # --------------------------------------------------------------
    @unittest.skipIf(
        SphericalGridDistribution is None,
        "SphericalGridDistribution not available",
    )
    def test_approx_vmf_mixture_s2(self):
        """
        MATLAB: testApproxVMMixtureS2

        Compare HypersphericalGridDistribution with SphericalGridDistribution
        on S^2 for a VMF mixture.
        """
        mu1 = 1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0])
        mu2 = np.array([0.0, -1.0, 0.0])
        dist1 = VonMisesFisherDistribution(mu1, 2.0)
        dist2 = VonMisesFisherDistribution(mu2, 2.0)
        dist = HypersphericalMixture([dist1, dist2], [0.5, 0.5])

        hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)
        sgd = SphericalGridDistribution.from_distribution(dist, 1012)

        dim = hgd.dim
        grid_hgd = _standardize_grid(hgd.get_grid(), dim)
        grid_sgd = _standardize_grid(sgd.get_grid(), dim)

        np.testing.assert_allclose(grid_hgd, grid_sgd, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            hgd.grid_values, sgd.grid_values, atol=1e-12, rtol=0
        )

    def test_approx_vmf_mixture_sd(self):
        """
        MATLAB: testApproxVMMixtureSd

        For dimensions 2..6:
        - Sample two random VMFs
        - Multiply them analytically (VMF.multiply)
        - HypersphericalGridDistribution approximating the mixture of the two
          should have mean direction close to vmfMult.mu.
        """
        for dim in range(2, 7):
            # Re-seed to mimic MATLAB rng(1) per dimension
            np.random.seed(1)

            mu1_raw = HypersphericalUniformDistribution(dim).sample(1)
            mu2_raw = HypersphericalUniformDistribution(dim).sample(1)
            mu1 = np.asarray(mu1_raw, dtype=float).reshape(-1)[:dim]
            mu2 = np.asarray(mu2_raw, dtype=float).reshape(-1)[:dim]

            vmf1 = VonMisesFisherDistribution(mu1, 2.0)
            vmf2 = VonMisesFisherDistribution(mu2, 2.0)
            vmf_mult = vmf1.multiply(vmf2)

            dist = HypersphericalMixture([vmf1, vmf2], [0.5, 0.5])
            hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)

            # For VMF, mean direction is mu
            expected_mu = np.asarray(vmf_mult.mu, dtype=float).reshape(-1)
            np.testing.assert_allclose(
                hgd.mean_direction(),
                expected_mu,
                atol=5e-2,
                rtol=0,
            )

    @unittest.skipIf(
        SphericalGridDistribution is None,
        "SphericalGridDistribution not available",
    )
    def test_approx_bingham_s2(self):
        """
        MATLAB: testApproxBinghamS2

        Bingham on S^2: first verify SphericalGridDistribution approximates it,
        then check that HypersphericalGridDistribution matches SphericalGridDistribution.
        """
        M = np.eye(3)
        Z = np.array([-2.0, -1.0, 0.0])
        dist = BinghamDistribution(Z, M)

        # Optional: improve normalization constant if present
        if hasattr(dist, "integralNumerical"):
            dist.F = dist.F * dist.integralNumerical
        elif hasattr(dist, "integral_numerical"):
            dist.F = dist.F * dist.integral_numerical

        hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)
        sgd = SphericalGridDistribution.from_distribution(dist, 1012)

        # First verify that SphericalGridDistribution approximates Bingham
        self.verify_pdf_equal(sgd, dist, tol=1e-6)

        dim = hgd.dim
        grid_hgd = _standardize_grid(hgd.get_grid(), dim)
        grid_sgd = _standardize_grid(sgd.get_grid(), dim)

        np.testing.assert_allclose(grid_hgd, grid_sgd, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            hgd.grid_values, sgd.grid_values, atol=1e-12, rtol=0
        )

    @unittest.skipIf(
        SphericalGridDistribution is None,
        "SphericalGridDistribution not available",
    )
    def test_approx_bingham_s3(self):
        """
        MATLAB: testApproxBinghamS3

        Bingham on S^3 (dim=4).
        """
        M = np.eye(4)
        Z = np.array([-2.0, -1.0, -0.5, 0.0])
        dist = BinghamDistribution(Z, M)

        if hasattr(dist, "integralNumerical"):
            dist.F = dist.F * dist.integralNumerical
        elif hasattr(dist, "integral_numerical"):
            dist.F = dist.F * dist.integral_numerical

        hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)
        # No SphericalGridDistribution in S^3 here in MATLAB; they only checked
        # S^3 via hemisphere tests elsewhere. We'll just check that the grid is
        # consistent with its own pdf:  pdf(grid) ~ grid_values.
        grid = _grid_for_pdf(hgd.get_grid(), hgd.dim)
        np.testing.assert_allclose(
            hgd.grid_values,
            dist.pdf(grid),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_mean_direction_sd(self):
        """
        MATLAB: testMeanDirectionSd

        For each dimension, VMF mean direction ~ mu used to build it.
        """
        for dim in range(2, 6):
            np.random.seed(1)
            mu_raw = HypersphericalUniformDistribution(dim).sample(1)
            mu = np.asarray(mu_raw, dtype=float).reshape(-1)[:dim]

            vmf = VonMisesFisherDistribution(mu, 2.0)
            hgd = HypersphericalGridDistribution.from_distribution(vmf, 1012)

            np.testing.assert_allclose(
                hgd.mean_direction(),
                mu,
                atol=2e-2,
                rtol=0,
            )

    # --------------------------------------------------------------
    # Multiply tests
    # --------------------------------------------------------------
    @unittest.skipIf(
        SphericalGridDistribution is None,
        "SphericalGridDistribution not available",
    )
    def test_multiply_vmf_s2(self):
        """
        MATLAB: testMultiplyVMFS2

        Validate HypersphericalGridDistribution.multiply against
        SphericalGridDistribution.multiply for VMFs on S^2.
        """
        kappas = [0.1 + 0.3 * i for i in range(14)]  # 0.1:0.3:4

        for kappa1 in kappas:
            for kappa2 in kappas:
                dist1 = VonMisesFisherDistribution(
                    1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0]), kappa1
                )
                dist2 = VonMisesFisherDistribution(
                    np.array([0.0, -1.0, 0.0]), kappa2
                )

                hgd1 = HypersphericalGridDistribution.from_distribution(
                    dist1, 1000, "eq_point_set"
                )
                hgd2 = HypersphericalGridDistribution.from_distribution(
                    dist2, 1000, "eq_point_set"
                )
                hgd_filtered = hgd1.multiply(hgd2)

                sgd1 = SphericalGridDistribution.from_distribution(
                    dist1, 1000, "eq_point_set"
                )
                sgd2 = SphericalGridDistribution.from_distribution(
                    dist2, 1000, "eq_point_set"
                )
                sgd_filtered = sgd1.multiply(sgd2)

                np.testing.assert_allclose(
                    hgd_filtered.grid_values,
                    sgd_filtered.grid_values,
                    atol=1e-6,
                    rtol=0,
                )

    def test_multiply_vmf_sd(self):
        """
        MATLAB: testMultiplyVMFSd

        For general dim, the mean direction of the product of two grid
        distributions approximates the mean direction of the analytic
        product VMF (vmf1.multiply(vmf2)).
        """
        kappa1 = 2.0
        kappa2 = 1.0

        for dim in range(2, 7):
            np.random.seed(1)
            mu1_raw = HypersphericalUniformDistribution(dim).sample(1)
            mu2_raw = HypersphericalUniformDistribution(dim).sample(1)
            mu1 = np.asarray(mu1_raw, dtype=float).reshape(-1)[:dim]
            mu2 = np.asarray(mu2_raw, dtype=float).reshape(-1)[:dim]

            vmf1 = VonMisesFisherDistribution(mu1, kappa1)
            vmf2 = VonMisesFisherDistribution(mu2, kappa2)
            vmf_mult = vmf1.multiply(vmf2)

            hgd1 = HypersphericalGridDistribution.from_distribution(
                vmf1, 1000, "eq_point_set"
            )
            hgd2 = HypersphericalGridDistribution.from_distribution(
                vmf2, 1000, "eq_point_set"
            )
            hgd_mult = hgd1.multiply(hgd2)

            expected_mu = np.asarray(vmf_mult.mu, dtype=float).reshape(-1)
            np.testing.assert_allclose(
                hgd_mult.mean_direction(),
                expected_mu,
                atol=5e-2,
                rtol=0,
            )

    def test_multiply_error(self):
        """
        MATLAB: testMultiplyError

        Two grid distributions with incompatible grids must trigger
        a 'Multiply:IncompatibleGrid' error.
        """
        dist1 = VonMisesFisherDistribution(
            1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0]), 1.0
        )
        f1 = HypersphericalGridDistribution.from_distribution(
            dist1, 84, "eq_point_set"
        )

        # Make an independent copy and truncate its grid
        f2 = HypersphericalGridDistribution(
            f1.grid.copy(), f1.grid_values.copy()
        )
        f2.grid_values = f2.grid_values[:-1]
        grid_full = f2.get_grid()
        # Standardize then drop last point
        grid_full_std = _standardize_grid(grid_full, f2.dim)
        # convert back to (dim, n_points-1)
        if grid_full_std.shape[1] == f2.dim:
            # (n_points, dim)
            f2.grid = grid_full_std[:-1, :].T
        else:
            # already (dim, n_points)
            f2.grid = grid_full_std[:, :-1]

        with self.assertRaises(ValueError) as cm:
            f1.multiply(f2)
        self.assertIn("IncompatibleGrid", str(cm.exception))

    # --------------------------------------------------------------
    # Symmetrize tests
    # --------------------------------------------------------------
    def test_symmetrize_vmf_mixture_s2(self):
        """
        MATLAB: testSymmetrizeVMFMixtureS2
        """
        dist = HypersphericalMixture(
            [
                VonMisesFisherDistribution(np.array([0.0, 1.0, 0.0]), 2.0),
                VonMisesFisherDistribution(np.array([0.0, -1.0, 0.0]), 2.0),
            ],
            [0.5, 0.5],
        )
        f = HypersphericalGridDistribution.from_distribution(
            dist, 50, "eq_point_set_symm"
        )

        n = f.grid_values.shape[0]
        half = n // 2
        self.assertEqual(n % 2, 0)

        # For symmetric VMF mixture + symmetric grid, second half equals first half
        np.testing.assert_allclose(
            f.grid_values[half:], f.grid_values[:half], atol=1e-10, rtol=0
        )

        # Break symmetry in the values
        f_asymm = HypersphericalGridDistribution(
            f.grid.copy(), f.grid_values.copy()
        )
        # swap indices 25 and 26 (1-based 26 and 27 in MATLAB)
        i1, i2 = 25, 26
        f_asymm.grid_values[i1], f_asymm.grid_values[i2] = (
            f_asymm.grid_values[i2],
            f_asymm.grid_values[i1],
        )

        self.assertFalse(
            np.allclose(
                f_asymm.grid_values[half:], f_asymm.grid_values[:half]
            )
        )

        f_symm = f_asymm.symmetrize()

        np.testing.assert_allclose(
            f_symm.grid_values[half:], f_symm.grid_values[:half], atol=1e-10, rtol=0
        )
        self.assertFalse(
            np.allclose(f_symm.grid_values, f_asymm.grid_values)
        )
        self.assertFalse(
            np.allclose(f_symm.grid_values, f.grid_values)
        )

    def test_symmetrize_watson_s3(self):
        """
        MATLAB: testSymmetrizeWatsonS3
        """
        dist = WatsonDistribution(
            1 / np.sqrt(2) * np.array([1.0, 1.0, 0.0]), 1.0
        )
        f = HypersphericalGridDistribution.from_distribution(
            dist, 50, "eq_point_set_symm"
        )
        n = f.grid_values.shape[0]
        half = n // 2
        self.assertEqual(n % 2, 0)

        np.testing.assert_allclose(
            f.grid_values[half:], f.grid_values[:half], atol=1e-10, rtol=0
        )

        f_asymm = HypersphericalGridDistribution(
            f.grid.copy(), f.grid_values.copy()
        )
        # swap two arbitrary entries (matching MATLAB indices [26,46])
        idx1, idx2 = 25, 45
        f_asymm.grid_values[idx1], f_asymm.grid_values[idx2] = (
            f_asymm.grid_values[idx2],
            f_asymm.grid_values[idx1],
        )
        self.assertFalse(
            np.allclose(
                f_asymm.grid_values[half:], f_asymm.grid_values[:half]
            )
        )

        f_symm = f_asymm.symmetrize()
        np.testing.assert_allclose(
            f_symm.grid_values[half:], f_symm.grid_values[:half], atol=1e-10, rtol=0
        )
        self.assertFalse(
            np.allclose(f_symm.grid_values, f_asymm.grid_values)
        )
        self.assertFalse(
            np.allclose(f_symm.grid_values, f.grid_values)
        )

    def test_symmetrize_error(self):
        """
        MATLAB: testSymmetrizeError
        """
        dist = VonMisesFisherDistribution(
            1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0]), 1.0
        )
        f = HypersphericalGridDistribution.from_distribution(
            dist, 84, "eq_point_set"
        )
        with self.assertRaises(ValueError) as cm:
            f.symmetrize()
        self.assertIn("Symmetrize:AsymmetricGrid", str(cm.exception))

    # --------------------------------------------------------------
    # Link between full sphere and hemisphere (sanity)
    # --------------------------------------------------------------
    def test_to_hemisphere_and_back_full_sphere(self):
        """
        Sanity check: for a symmetric mixture of antipodal VMFs on S^2,
        converting to a hemisphere and back via HyperhemisphericalGridDistribution
        should reproduce the original full-sphere grid (up to interpolation).
        """
        base_mu = 1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0])
        dist = HypersphericalMixture(
            [
                VonMisesFisherDistribution(base_mu, 1.0),
                VonMisesFisherDistribution(-base_mu, 1.0),
            ],
            [0.5, 0.5],
        )

        hgd = HypersphericalGridDistribution.from_distribution(
            dist, 84, "eq_point_set_symm"
        )
        hhgd = hgd.to_hemisphere()
        hgd_back = hhgd.to_full_sphere()

        dim = hgd.dim
        grid_hgd = _standardize_grid(hgd.get_grid(), dim)
        grid_back = _standardize_grid(hgd_back.get_grid(), dim)

        np.testing.assert_allclose(grid_back, grid_hgd, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            hgd_back.grid_values, hgd.grid_values, atol=1e-12, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
