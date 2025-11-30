import unittest
import warnings
import copy
import pyrecest

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
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions import HypersphericalMixture


from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)
from pyrecest.backend import meshgrid, linspace, cos, sin, vstack, sqrt, array, random, allclose, eye
from math import pi
import numpy.testing as npt

class HypersphericalGridDistributionTest(unittest.TestCase):
    # --------------------------------------------------------------
    # Helper: PDF equality on a small cartesian grid (S²)
    # --------------------------------------------------------------
    def verify_pdf_equal(self, dist1, dist2, tol):
        """
        Compare pdfs of two distributions on a simple grid on S².
        """
        # S² grid
        phi, theta = meshgrid(
            linspace(0.0, 2 * pi, 10),
            linspace(-pi / 2, pi / 2, 10),
        )
        phi = phi.ravel()
        theta = theta.ravel()
        r = array(1.0)

        x = r * cos(theta) * cos(phi)
        y = r * cos(theta) * sin(phi)
        z = r * sin(theta)

        pts = vstack((x, y, z)).T  # (n_points, 3)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = dist1.pdf(pts)
            p2 = dist2.pdf(pts)

        npt.assert_allclose(p1, p2, atol=tol, rtol=0.1)

    # --------------------------------------------------------------
    # Approximation tests
    # --------------------------------------------------------------
    def test_approx_vmf_mixture_s2(self):
        """
        Compare HypersphericalGridDistribution with SphericalGridDistribution
        on S² for a VMF mixture.
        """
        mu1 = 1 / sqrt(2) * array([-1.0, 0.0, 1.0])
        mu2 = array([0.0, -1.0, 0.0])
        dist1 = VonMisesFisherDistribution(mu1, array(2.0))
        dist2 = VonMisesFisherDistribution(mu2, array(2.0))
        dist = HypersphericalMixture([dist1, dist2], array([0.5, 0.5]))

        hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)
        sgd = SphericalGridDistribution.from_distribution(dist, 1012)

        grid_hgd = hgd.get_grid()
        grid_sgd = sgd.get_grid()

        npt.assert_allclose(grid_hgd, grid_sgd, atol=1e-12, rtol=0)
        npt.assert_allclose(
            hgd.grid_values, sgd.grid_values, atol=1e-12, rtol=0
        )

    def test_approx_vmf_mixture_sd(self):
        """
        For dimensions 2..6:
        - Sample two random VMFs
        - Multiply them analytically (.multiply)
        - HypersphericalGridDistribution approximating the mixture of the two
          should have mean direction close to vmf_mult.mu.
        """
        for dim in range(2, 6):
            random.seed(1)

            mu1 = HypersphericalUniformDistribution(dim).sample(1).reshape((-1,))
            mu2 = HypersphericalUniformDistribution(dim).sample(1).reshape((-1,))

            vmf1 = VonMisesFisherDistribution(mu1, array(2.0))
            vmf2 = VonMisesFisherDistribution(mu2, array(2.0))
            vmf_mult = vmf1.multiply(vmf2)

            dist = HypersphericalMixture([vmf1, vmf2], array([0.5, 0.5]))
            hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)

            # For VMF, mean direction is mu
            expected_mu = vmf_mult.mu
            npt.assert_allclose(
                hgd.mean_direction(),
                expected_mu,
                atol=5e-2,
                rtol=0,
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_approx_bingham_s2(self):
        """
        Bingham on S²: first verify SphericalGridDistribution approximates it,
        then check that HypersphericalGridDistribution matches SphericalGridDistribution.
        """
        M = eye(3)
        Z = array([-2.0, -1.0, 0.0])
        dist = BinghamDistribution(Z, M)

        # Optional: improve normalization constant if present
        dist.F = dist.F * dist.integrate_numerically()

        hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)
        sgd = SphericalGridDistribution.from_distribution(dist, 1012)

        # First verify that SphericalGridDistribution approximates Bingham
        self.verify_pdf_equal(sgd, dist, tol=1e-6)

        grid_hgd = hgd.get_grid()
        grid_sgd = sgd.get_grid()

        npt.assert_allclose(grid_hgd, grid_sgd, atol=1e-12, rtol=0)
        npt.assert_allclose(
            hgd.grid_values, sgd.grid_values, atol=1e-12, rtol=0
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_approx_bingham_s3(self):
        """
        Bingham on S^3 (dim=4).
        """
        M = eye(4)
        Z = array([-2.0, -1.0, -0.5, 0.0])
        dist = BinghamDistribution(Z, M)

        dist.F = dist.F * dist.integrate_numerically()

        hgd = HypersphericalGridDistribution.from_distribution(dist, 1012)
        # We just check that the grid is consistent with its own pdf:  pdf(grid) ~ grid_values.
        grid = hgd.get_grid()
        npt.assert_allclose(
            hgd.grid_values,
            dist.pdf(grid),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_mean_direction_sd(self):
        """
        For each dimension, VMF mean direction ~ mu used to build it.
        """
        for dim in range(2, 6):
            random.seed(1)
            mu = HypersphericalUniformDistribution(dim).sample(1).reshape((-1,))

            vmf = VonMisesFisherDistribution(mu, 2.0)
            hgd = HypersphericalGridDistribution.from_distribution(vmf, 1012)

            npt.assert_allclose(
                hgd.mean_direction(),
                mu,
                rtol=0.1,
            )

    # --------------------------------------------------------------
    # Multiply tests
    # --------------------------------------------------------------
    def test_multiply_vmf_s2(self):
        """
        Validate HypersphericalGridDistribution.multiply against
        SphericalGridDistribution.multiply for VMFs on S².
        """
        kappas = [0.1 + 0.3 * i for i in range(14)]  # 0.1:0.3:4

        for kappa1 in kappas:
            for kappa2 in kappas:
                dist1 = VonMisesFisherDistribution(
                    1 / sqrt(2) * array([-1.0, 0.0, 1.0]), kappa1
                )
                dist2 = VonMisesFisherDistribution(
                    array([0.0, -1.0, 0.0]), kappa2
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

                npt.assert_allclose(
                    hgd_filtered.grid_values,
                    sgd_filtered.grid_values,
                    atol=1e-6,
                )

    def test_multiply_vmf_sd(self):
        """
        For general dim, the mean direction of the product of two grid
        distributions approximates the mean direction of the analytic
        product VMF (vmf1.multiply(vmf2)).
        """
        kappa1 = 2.0
        kappa2 = 1.0

        for dim in range(2, 7):
            random.seed(1)
            mu1 = HypersphericalUniformDistribution(dim).sample(1).reshape((-1,))
            mu2 = HypersphericalUniformDistribution(dim).sample(1).reshape((-1,))

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

            expected_mu = vmf_mult.mu
            npt.assert_allclose(
                hgd_mult.mean_direction(),
                expected_mu,
                atol=0.08,
            )

    def test_multiply_error(self):
        """
        Two grid distributions with incompatible grids must trigger
        a 'Multiply:IncompatibleGrid' error.
        """
        dist1 = VonMisesFisherDistribution(
            1 / sqrt(2) * array([-1.0, 0.0, 1.0]), 1.0
        )
        f1 = HypersphericalGridDistribution.from_distribution(
            dist1, 84, "eq_point_set"
        )

        # Make an independent copy and truncate its grid
        f2 = HypersphericalGridDistribution(
            copy.deepcopy(f1.grid), copy.deepcopy(f1.grid_values)
        )
        f2.grid_values = f2.grid_values[:-1]
        grid_full = f2.get_grid()
        # Standardize then drop last point
        grid_full_std = grid_full
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
        dist = HypersphericalMixture(
            [
                VonMisesFisherDistribution(array([0.0, 1.0, 0.0]), 2.0),
                VonMisesFisherDistribution(array([0.0, -1.0, 0.0]), 2.0),
            ],
            array([0.5, 0.5]),
        )
        f = HypersphericalGridDistribution.from_distribution(
            dist, 50, "eq_point_set_symm"
        )

        n = f.grid_values.shape[0]
        half = n // 2
        self.assertEqual(n % 2, 0)

        # For symmetric VMF mixture + symmetric grid, second half equals first half
        npt.assert_allclose(
            f.grid_values[half:], f.grid_values[:half], atol=1e-10, rtol=0
        )

        # Break symmetry in the values
        f_asymm = HypersphericalGridDistribution(
            copy.deepcopy(f.grid), copy.deepcopy(f.grid_values)
        )
        # swap indices 25 and 26
        i1, i2 = 25, 26
        f_asymm.grid_values[i1], f_asymm.grid_values[i2] = (
            f_asymm.grid_values[i2],
            f_asymm.grid_values[i1],
        )

        self.assertFalse(
            allclose(
                f_asymm.grid_values[half:], f_asymm.grid_values[:half]
            )
        )

        f_symm = f_asymm.symmetrize()

        npt.assert_allclose(
            f_symm.grid_values[half:], f_symm.grid_values[:half], atol=1e-10, rtol=0
        )
        self.assertFalse(
            allclose(f_symm.grid_values, f_asymm.grid_values)
        )
        self.assertFalse(
            allclose(f_symm.grid_values, f.grid_values)
        )

    def test_symmetrize_watson_s3(self):
        dist = WatsonDistribution(
            1 / sqrt(2) * array([1.0, 1.0, 0.0]), 1.0
        )
        f = HypersphericalGridDistribution.from_distribution(
            dist, 50, "eq_point_set_symm"
        )
        n = f.grid_values.shape[0]
        half = n // 2
        self.assertEqual(n % 2, 0)

        npt.assert_allclose(
            f.grid_values[half:], f.grid_values[:half], atol=1e-10, rtol=0
        )

        f_asymm = HypersphericalGridDistribution(
            copy.deepcopy(f.grid), copy.deepcopy(f.grid_values)
        )
        # swap two arbitrary entries
        idx1, idx2 = 25, 45
        f_asymm.grid_values[idx1], f_asymm.grid_values[idx2] = (
            f_asymm.grid_values[idx2],
            f_asymm.grid_values[idx1],
        )
        self.assertFalse(
            allclose(
                f_asymm.grid_values[half:], f_asymm.grid_values[:half]
            )
        )

        f_symm = f_asymm.symmetrize()
        npt.assert_allclose(
            f_symm.grid_values[half:], f_symm.grid_values[:half], atol=1e-10, rtol=0
        )
        self.assertFalse(
            allclose(f_symm.grid_values, f_asymm.grid_values)
        )
        self.assertFalse(
            allclose(f_symm.grid_values, f.grid_values)
        )

    def test_symmetrize_error(self):
        dist = VonMisesFisherDistribution(
            1 / sqrt(2) * array([-1.0, 0.0, 1.0]), 1.0
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
        Sanity check: for a symmetric mixture of antipodal VMFs on S²,
        converting to a hemisphere and back via HyperhemisphericalGridDistribution
        should reproduce the original full-sphere grid (up to interpolation).
        """
        base_mu = 1 / sqrt(2) * array([-1.0, 0.0, 1.0])
        dist = HypersphericalMixture(
            [
                VonMisesFisherDistribution(base_mu, 1.0),
                VonMisesFisherDistribution(-base_mu, 1.0),
            ],
            array([0.5, 0.5]),
        )

        hgd = HypersphericalGridDistribution.from_distribution(
            dist, 84, "eq_point_set_symm"
        )
        hhgd = hgd.to_hemisphere()
        hgd_back = hhgd.to_full_sphere()

        grid_hgd = hgd.get_grid()
        grid_back = hgd_back.get_grid()

        npt.assert_allclose(grid_back, grid_hgd, atol=1e-12, rtol=0)
        npt.assert_allclose(
            hgd_back.grid_values, hgd.grid_values, atol=1e-12, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
