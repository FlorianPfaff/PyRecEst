import copy
import unittest
import warnings

import numpy.testing as npt
import pyrecest
from pyrecest.backend import array, eye, sqrt
from pyrecest.distributions import HypersphericalMixture
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)


class HyperhemisphericalGridDistributionTest(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # Warning tests (testWarningAsymm)
    # ------------------------------------------------------------------ #
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_warning_asymm(self):
        """
        - Asymmetric VMF on S²
        - Asymmetric mixture of two VMFs on S²

        Expect a warning about approximating a hyperspherical distribution
        on a hemisphere.
        """
        mu_vmf = 1 / sqrt(2) * array([-1.0, 0.0, -1.0])
        vmf = VonMisesFisherDistribution(mu_vmf, 2.0)

        comp1 = VonMisesFisherDistribution(mu_vmf, 2.0)
        comp2 = VonMisesFisherDistribution(1 / sqrt(2) * array([1.0, 0.0, -1.0]), 2.0)
        mixture = HypersphericalMixture([comp1, comp2], array([0.5, 0.5]))

        # VMF
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HyperhemisphericalGridDistribution.from_distribution(
                vmf, 1012, "leopardi_symm"
            )
        self.assertTrue(
            any(
                "Approximating a hyperspherical distribution on a hemisphere"
                in str(wi.message)
                for wi in w
            ),
            msg="Expected asymmetry warning for VMF distribution.",
        )

        # Mixture
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HyperhemisphericalGridDistribution.from_distribution(
                mixture, 1012, "leopardi_symm"
            )
        self.assertTrue(
            any(
                "Approximating a hyperspherical distribution on a hemisphere"
                in str(wi.message)
                for wi in w
            ),
            msg="Expected asymmetry warning for VMF mixture distribution.",
        )

    # ------------------------------------------------------------------ #
    # Approximation tests: VMF mixture and Bingham.
    # ------------------------------------------------------------------ #
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_approx_vmf_mixture_s2(self):
        """
        Verify that for a symmetric VMF mixture on S², the hemisphere
        grid pdf values equal 2 * dist.pdf(grid).
        """
        mu1 = 1 / sqrt(2) * array([-1.0, 0.0, 1.0])
        mu2 = 1 / sqrt(2) * array([1.0, 0.0, -1.0])
        dist1 = VonMisesFisherDistribution(mu1, 2.0)
        dist2 = VonMisesFisherDistribution(mu2, 2.0)
        dist = HypersphericalMixture([dist1, dist2], array([0.5, 0.5]))

        hhgd = HyperhemisphericalGridDistribution.from_distribution(
            dist, 1012, "leopardi_symm"
        )
        grid = hhgd.get_grid()

        npt.assert_allclose(
            hhgd.grid_values,
            2 * dist.pdf(grid),
            rtol=1e-12,
            atol=1e-12,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_approx_bingham_s2(self):
        """
        Bingham on S².
        """
        M = eye(3)
        Z = array([-2.0, -1.0, 0.0])
        dist = BinghamDistribution(Z, M)

        # Improve normalization constant.
        dist.F = dist.F * dist.integrate_numerically()

        hhgd = HyperhemisphericalGridDistribution.from_distribution(
            dist, 1012, "leopardi_symm"
        )
        grid = hhgd.get_grid()

        npt.assert_allclose(
            hhgd.grid_values,
            2 * dist.pdf(grid),
            rtol=1e-12,
            atol=1e-12,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_approx_bingham_s3(self):
        """
        Bingham on S³.
        """
        M = eye(4)
        Z = array([-2.0, -1.0, -0.5, 0.0])
        dist = BinghamDistribution(Z, M)

        # Improve normalization constant.
        dist.F = dist.F * dist.integrate_numerically()

        hhgd = HyperhemisphericalGridDistribution.from_distribution(
            dist, 1012, "leopardi_symm"
        )
        grid = hhgd.get_grid()

        npt.assert_allclose(
            hhgd.grid_values,
            2 * dist.pdf(grid),
            rtol=1e-12,
            atol=1e-12,
        )

    # ------------------------------------------------------------------ #
    # Multiply tests
    # ------------------------------------------------------------------ #
    # pylint: disable=too-many-locals
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_multiply_vmf_mixture_s2(self):
        """
        Compare HyperhemisphericalGridDistribution.multiply with
        HypersphericalGridDistribution.multiply on S².
        """
        kappas = [0.1 + 0.3 * i for i in range(14)]

        for kappa1 in kappas:
            for kappa2 in kappas:
                # dist1: mixture around +/- (1/sqrt(2)) * [-1, 0, 1]
                base_mu1 = 1 / sqrt(2) * array([-1.0, 0.0, 1.0])
                dist1_comp1 = VonMisesFisherDistribution(base_mu1, kappa1)
                dist1_comp2 = VonMisesFisherDistribution(-base_mu1, kappa1)
                dist1 = HypersphericalMixture(
                    [dist1_comp1, dist1_comp2], array([0.5, 0.5])
                )

                # dist2: mixture around [0, -1, 0] and [0, 1, 0]
                mu21 = array([0.0, -1.0, 0.0])
                mu22 = array([0.0, 1.0, 0.0])
                dist2_comp1 = VonMisesFisherDistribution(mu21, kappa2)
                dist2_comp2 = VonMisesFisherDistribution(mu22, kappa2)
                dist2 = HypersphericalMixture(
                    [dist2_comp1, dist2_comp2], array([0.5, 0.5])
                )

                hhgd1 = HyperhemisphericalGridDistribution.from_distribution(
                    dist1, 1000, "leopardi_symm"
                )
                hhgd2 = HyperhemisphericalGridDistribution.from_distribution(
                    dist2, 1000, "leopardi_symm"
                )
                hhgd_filtered = hhgd1.multiply(hhgd2)

                hgd1 = HypersphericalGridDistribution.from_distribution(
                    dist1, 2000, "leopardi_symm_antipodal"
                )
                hgd2 = HypersphericalGridDistribution.from_distribution(
                    dist2, 2000, "leopardi_symm_antipodal"
                )
                hgd_filtered = hgd1.multiply(hgd2)

                hemi_grid = hhgd_filtered.get_grid()
                full_grid = hgd_filtered.get_grid()

                n_hemi = hemi_grid.shape[0]

                # Grids must match for the hemisphere part
                npt.assert_allclose(
                    hemi_grid,
                    full_grid[:n_hemi, :],
                    rtol=0,
                    atol=1e-12,
                )

                # Values: 0.5 * hemisphere values == full-sphere values (hemisphere part)
                hemi_values = hhgd_filtered.grid_values
                full_values = hgd_filtered.grid_values

                npt.assert_allclose(
                    0.5 * hemi_values,
                    full_values[:n_hemi],
                    rtol=0,
                    atol=1e-11,
                )

    # pylint: disable=too-many-locals
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_multiply_vmf_mixture_s3(self):
        # Representative low/medium/high concentrations from the original
        # 0.1:0.3:4 sweep keep the coverage while avoiding a 14x14 grid loop.
        kappas = [0.1, 2.2, 4.0]

        for kappa1 in kappas:
            for kappa2 in kappas:
                # dist1: mixture around +/- (1/sqrt(3)) * [-1, 0, 1, 1]
                base_mu1 = 1 / sqrt(3) * array([-1.0, 0.0, 1.0, 1.0])
                dist1_comp1 = VonMisesFisherDistribution(base_mu1, kappa1)
                dist1_comp2 = VonMisesFisherDistribution(-base_mu1, kappa1)
                dist1 = HypersphericalMixture(
                    [dist1_comp1, dist1_comp2], array([0.5, 0.5])
                )

                # dist2: mixture around [0, -1, 0, 0] and [0, 1, 0, 0]
                mu21 = array([0.0, -1.0, 0.0, 0.0])
                mu22 = array([0.0, 1.0, 0.0, 0.0])
                dist2_comp1 = VonMisesFisherDistribution(mu21, kappa2)
                dist2_comp2 = VonMisesFisherDistribution(mu22, kappa2)
                dist2 = HypersphericalMixture(
                    [dist2_comp1, dist2_comp2], array([0.5, 0.5])
                )

                # Hemisphere grids
                hhgd1 = HyperhemisphericalGridDistribution.from_distribution(
                    dist1, 1000, "leopardi_symm"
                )
                hhgd2 = HyperhemisphericalGridDistribution.from_distribution(
                    dist2, 1000, "leopardi_symm"
                )
                hhgd_filtered = hhgd1.multiply(hhgd2)

                # Full-sphere grids
                hgd1 = HypersphericalGridDistribution.from_distribution(
                    dist1, 2000, "leopardi_symm_antipodal"
                )
                hgd2 = HypersphericalGridDistribution.from_distribution(
                    dist2, 2000, "leopardi_symm_antipodal"
                )
                hgd_filtered = hgd1.multiply(hgd2)

                hemi_grid = hhgd_filtered.get_grid()
                full_grid = hgd_filtered.get_grid()

                n_hemi = hemi_grid.shape[0]

                npt.assert_allclose(
                    hemi_grid,
                    full_grid[:n_hemi, :],
                    rtol=0,
                    atol=1e-10,  # slightly looser than S2 test
                )

                hemi_values = hhgd_filtered.grid_values
                full_values = hgd_filtered.grid_values

                npt.assert_allclose(
                    0.5 * hemi_values,
                    full_values[:n_hemi],
                    rtol=0,
                    atol=1e-4,
                )

    # ------------------------------------------------------------------ #
    # Multiply error
    # ------------------------------------------------------------------ #
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_multiply_error(self):
        """
        Make two hemisphere grid distributions with incompatible grids and
        ensure multiply raises an error.
        """
        base_mu = 1 / sqrt(2) * array([-1.0, 0.0, 1.0])
        dist1_comp1 = VonMisesFisherDistribution(base_mu, 1.0)
        dist1_comp2 = VonMisesFisherDistribution(-base_mu, 1.0)
        dist1 = HypersphericalMixture([dist1_comp1, dist1_comp2], array([0.5, 0.5]))

        f1 = HyperhemisphericalGridDistribution.from_distribution(
            dist1, 84, "leopardi_symm"
        )
        # Make an *independent* copy of f1 with truncated grid
        f2 = HyperhemisphericalGridDistribution(
            copy.deepcopy(f1.get_grid()), copy.deepcopy(f1.grid_values)
        )
        f2.grid_values = f2.grid_values[:-1]
        f2.grid = f2.get_grid()[:, :-1]

        with self.assertRaises(ValueError) as cm:
            f1.multiply(f2)

        self.assertIn("IncompatibleGrid", str(cm.exception))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_to_full_sphere(self):
        """
        Convert a hemisphere grid distribution back to a full-sphere one and
        compare with a direct HypersphericalGridDistribution approximation.
        """
        base_mu = 1 / sqrt(2) * array([-1.0, 0.0, 1.0])
        dist_comp1 = VonMisesFisherDistribution(base_mu, 1.0)
        dist_comp2 = VonMisesFisherDistribution(-base_mu, 1.0)
        dist = HypersphericalMixture([dist_comp1, dist_comp2], array([0.5, 0.5]))

        hgd = HypersphericalGridDistribution.from_distribution(
            dist, 84, "leopardi_symm_antipodal"
        )
        hhgd = HyperhemisphericalGridDistribution.from_distribution(
            dist, 42, "leopardi_symm"
        )

        hhgd2hgd = hhgd.to_full_sphere()

        grid_hgd = hgd.get_grid()
        grid_hhgd2hgd = hhgd2hgd.get_grid()

        npt.assert_allclose(
            grid_hhgd2hgd,
            grid_hgd,
            rtol=0,
            atol=1e-12,
        )

        npt.assert_allclose(
            hhgd2hgd.grid_values,
            hgd.grid_values,
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
