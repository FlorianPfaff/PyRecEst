import unittest
import warnings
import numpy as np

from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions import HypersphericalMixture


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
        # ambiguous; just return as-is
        return grid
    raise ValueError(
        f"Grid shape {grid.shape} not compatible with dimension {dim}"
    )


def _standardize_grid(grid, dim):
    """
    Standardize grid orientation to (n_points, dim) for comparisons.
    """
    return _grid_for_pdf(grid, dim)


class HyperhemisphericalGridDistributionTest(unittest.TestCase):
    # ------------------------------------------------------------------ #
    # Warning tests (testWarningAsymm)
    # ------------------------------------------------------------------ #

    def test_warning_asymm(self):
        """
        Python analogue of MATLAB testWarningAsymm:

        - Asymmetric VMF on S^2
        - Asymmetric mixture of two VMFs on S^2

        Expect a warning about approximating a hyperspherical distribution
        on a hemisphere.
        """
        mu_vmf = 1 / np.sqrt(2) * np.array([-1.0, 0.0, -1.0])
        vmf = VonMisesFisherDistribution(mu_vmf, 2.0)

        comp1 = VonMisesFisherDistribution(mu_vmf, 2.0)
        comp2 = VonMisesFisherDistribution(
            1 / np.sqrt(2) * np.array([1.0, 0.0, -1.0]), 2.0
        )
        mixture = HypersphericalMixture([comp1, comp2], [0.5, 0.5])

        # VMF
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HyperhemisphericalGridDistribution.from_distribution(vmf, 1012)
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
            HyperhemisphericalGridDistribution.from_distribution(mixture, 1012)
        self.assertTrue(
            any(
                "Approximating a hyperspherical distribution on a hemisphere"
                in str(wi.message)
                for wi in w
            ),
            msg="Expected asymmetry warning for VMF mixture distribution.",
        )

    # ------------------------------------------------------------------ #
    # Approximation tests: VMF mixture and Bingham (testApproxVMMixtureS2,
    # testApproxBinghamS2, testApproxBinghamS3)
    # ------------------------------------------------------------------ #

    def test_approx_vmf_mixture_s2(self):
        """
        MATLAB: testApproxVMMixtureS2

        Verify that for a symmetric VMF mixture on S^2, the hemisphere
        grid pdf values equal 2 * dist.pdf(grid).
        """
        mu1 = 1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0])
        mu2 = 1 / np.sqrt(2) * np.array([1.0, 0.0, -1.0])
        dist1 = VonMisesFisherDistribution(mu1, 2.0)
        dist2 = VonMisesFisherDistribution(mu2, 2.0)
        dist = HypersphericalMixture([dist1, dist2], [0.5, 0.5])

        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, 1012)
        grid = _grid_for_pdf(hhgd.get_grid(), dist.dim)

        np.testing.assert_allclose(
            hhgd.grid_values,
            2 * dist.pdf(grid),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_approx_bingham_s2(self):
        """
        MATLAB: testApproxBinghamS2

        Bingham on S^2 (dim=3).
        """
        M = np.eye(3)
        Z = np.array([-2.0, -1.0, 0.0])
        dist = BinghamDistribution(Z, M)

        # Optional: improve normalization constant if the API exposes it
        if hasattr(dist, "integralNumerical"):
            dist.F = dist.F * dist.integralNumerical
        elif hasattr(dist, "integral_numerical"):
            dist.F = dist.F * dist.integral_numerical

        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, 1012)
        grid = _grid_for_pdf(hhgd.get_grid(), dist.dim)

        np.testing.assert_allclose(
            hhgd.grid_values,
            2 * dist.pdf(grid),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_approx_bingham_s3(self):
        """
        MATLAB: testApproxBinghamS3

        Bingham on S^3 (dim=4).
        """
        M = np.eye(4)
        Z = np.array([-2.0, -1.0, -0.5, 0.0])
        dist = BinghamDistribution(Z, M)

        # Optional normalization improvement as above
        if hasattr(dist, "integralNumerical"):
            dist.F = dist.F * dist.integralNumerical
        elif hasattr(dist, "integral_numerical"):
            dist.F = dist.F * dist.integral_numerical

        hhgd = HyperhemisphericalGridDistribution.from_distribution(dist, 1012)
        grid = _grid_for_pdf(hhgd.get_grid(), dist.dim)

        np.testing.assert_allclose(
            hhgd.grid_values,
            2 * dist.pdf(grid),
            rtol=1e-12,
            atol=1e-12,
        )

    # ------------------------------------------------------------------ #
    # Multiply tests (testMultiplyVMFMixtureS2 and S3)
    # ------------------------------------------------------------------ #

    def test_multiply_vmf_mixture_s2(self):
        """
        MATLAB: testMultiplyVMFMixtureS2

        Compare HyperhemisphericalGridDistribution.multiply with
        HypersphericalGridDistribution.multiply on S^2 (dim=3).
        """
        kappas = [0.1 + 0.3 * i for i in range(14)]  # 0.1:0.3:4 in MATLAB

        for kappa1 in kappas:
            for kappa2 in kappas:
                # dist1: mixture around +/- (1/sqrt(2)) * [-1, 0, 1]
                base_mu1 = 1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0])
                dist1_comp1 = VonMisesFisherDistribution(base_mu1, kappa1)
                dist1_comp2 = VonMisesFisherDistribution(-base_mu1, kappa1)
                dist1 = HypersphericalMixture(
                    [dist1_comp1, dist1_comp2], [0.5, 0.5]
                )

                # dist2: mixture around [0, -1, 0] and [0, 1, 0]
                mu21 = np.array([0.0, -1.0, 0.0])
                mu22 = np.array([0.0, 1.0, 0.0])
                dist2_comp1 = VonMisesFisherDistribution(mu21, kappa2)
                dist2_comp2 = VonMisesFisherDistribution(mu22, kappa2)
                dist2 = HypersphericalMixture(
                    [dist2_comp1, dist2_comp2], [0.5, 0.5]
                )

                # Hemisphere grids
                hhgd1 = HyperhemisphericalGridDistribution.from_distribution(
                    dist1, 1000, "eq_point_set"
                )
                hhgd2 = HyperhemisphericalGridDistribution.from_distribution(
                    dist2, 1000, "eq_point_set"
                )
                hhgd_filtered = hhgd1.multiply(hhgd2)

                # Full-sphere grids
                hgd1 = HypersphericalGridDistribution.from_distribution(
                    dist1, 2000, "eq_point_set_symm"
                )
                hgd2 = HypersphericalGridDistribution.from_distribution(
                    dist2, 2000, "eq_point_set_symm"
                )
                hgd_filtered = hgd1.multiply(hgd2)

                dim = hhgd_filtered.dim
                hemi_grid = _standardize_grid(hhgd_filtered.get_grid(), dim)
                full_grid = _standardize_grid(hgd_filtered.get_grid(), dim)

                n_hemi = hemi_grid.shape[0]

                # Grids must match for the hemisphere part
                np.testing.assert_allclose(
                    hemi_grid,
                    full_grid[:n_hemi, :],
                    rtol=0,
                    atol=1e-12,
                )

                # Values: 0.5 * hemisphere values == full-sphere values (hemisphere part)
                hemi_values = np.asarray(hhgd_filtered.grid_values)
                full_values = np.asarray(hgd_filtered.grid_values)

                np.testing.assert_allclose(
                    0.5 * hemi_values,
                    full_values[:n_hemi],
                    rtol=0,
                    atol=1e-11,
                )

    def test_multiply_vmf_mixture_s3(self):
        """
        MATLAB: testMultiplyVMFMixtureS3

        Same as above, but on S^3 (dim=4).
        """
        kappas = [0.1 + 0.3 * i for i in range(14)]  # 0.1:0.3:4

        for kappa1 in kappas:
            for kappa2 in kappas:
                # dist1: mixture around +/- (1/sqrt(3)) * [-1, 0, 1, 1]
                base_mu1 = 1 / np.sqrt(3) * np.array([-1.0, 0.0, 1.0, 1.0])
                dist1_comp1 = VonMisesFisherDistribution(base_mu1, kappa1)
                dist1_comp2 = VonMisesFisherDistribution(-base_mu1, kappa1)
                dist1 = HypersphericalMixture(
                    [dist1_comp1, dist1_comp2], [0.5, 0.5]
                )

                # dist2: mixture around [0, -1, 0, 0] and [0, 1, 0, 0]
                mu21 = np.array([0.0, -1.0, 0.0, 0.0])
                mu22 = np.array([0.0, 1.0, 0.0, 0.0])
                dist2_comp1 = VonMisesFisherDistribution(mu21, kappa2)
                dist2_comp2 = VonMisesFisherDistribution(mu22, kappa2)
                dist2 = HypersphericalMixture(
                    [dist2_comp1, dist2_comp2], [0.5, 0.5]
                )

                # Hemisphere grids
                hhgd1 = HyperhemisphericalGridDistribution.from_distribution(
                    dist1, 1000, "eq_point_set"
                )
                hhgd2 = HyperhemisphericalGridDistribution.from_distribution(
                    dist2, 1000, "eq_point_set"
                )
                hhgd_filtered = hhgd1.multiply(hhgd2)

                # Full-sphere grids
                hgd1 = HypersphericalGridDistribution.from_distribution(
                    dist1, 2000, "eq_point_set_symm"
                )
                hgd2 = HypersphericalGridDistribution.from_distribution(
                    dist2, 2000, "eq_point_set_symm"
                )
                hgd_filtered = hgd1.multiply(hgd2)

                dim = hhgd_filtered.dim
                hemi_grid = _standardize_grid(hhgd_filtered.get_grid(), dim)
                full_grid = _standardize_grid(hgd_filtered.get_grid(), dim)

                n_hemi = hemi_grid.shape[0]

                np.testing.assert_allclose(
                    hemi_grid,
                    full_grid[:n_hemi, :],
                    rtol=0,
                    atol=1e-10,  # slightly looser than S2 test (matches MATLAB AbsTol=1e-4)
                )

                hemi_values = np.asarray(hhgd_filtered.grid_values)
                full_values = np.asarray(hgd_filtered.grid_values)

                np.testing.assert_allclose(
                    0.5 * hemi_values,
                    full_values[:n_hemi],
                    rtol=0,
                    atol=1e-4,
                )

    # ------------------------------------------------------------------ #
    # Multiply error (testMultiplyError) and to_full_sphere (testToFullSphere)
    # ------------------------------------------------------------------ #

    def test_multiply_error(self):
        """
        MATLAB: testMultiplyError

        Make two hemisphere grid distributions with incompatible grids and
        ensure multiply raises an error.
        """
        base_mu = 1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0])
        dist1_comp1 = VonMisesFisherDistribution(base_mu, 1.0)
        dist1_comp2 = VonMisesFisherDistribution(-base_mu, 1.0)
        dist1 = HypersphericalMixture([dist1_comp1, dist1_comp2], [0.5, 0.5])

        f1 = HyperhemisphericalGridDistribution.from_distribution(
            dist1, 84, "eq_point_set"
        )
        # Make an *independent* copy of f1 with truncated grid
        f2 = HyperhemisphericalGridDistribution(
            f1.grid.copy(), f1.grid_values.copy()
        )
        f2.grid_values = f2.grid_values[:-1]
        f2.grid = f2.grid[:, :-1]

        with self.assertRaises(ValueError) as cm:
            f1.multiply(f2)

        # If you implement multiply to raise a ValueError with message
        # "Multiply:IncompatibleGrid" (as in the MATLAB test), this will check it:
        self.assertIn("IncompatibleGrid", str(cm.exception))

    def test_to_full_sphere(self):
        """
        MATLAB: testToFullSphere

        Convert a hemisphere grid distribution back to a full-sphere one and
        compare with a direct HypersphericalGridDistribution approximation.
        """
        base_mu = 1 / np.sqrt(2) * np.array([-1.0, 0.0, 1.0])
        dist_comp1 = VonMisesFisherDistribution(base_mu, 1.0)
        dist_comp2 = VonMisesFisherDistribution(-base_mu, 1.0)
        dist = HypersphericalMixture([dist_comp1, dist_comp2], [0.5, 0.5])

        hgd = HypersphericalGridDistribution.from_distribution(
            dist, 84, "eq_point_set_symm"
        )
        hhgd = HyperhemisphericalGridDistribution.from_distribution(
            dist, 42, "eq_point_set_symm"
        )

        hhgd2hgd = hhgd.to_full_sphere()

        dim = hgd.dim
        grid_hgd = _standardize_grid(hgd.get_grid(), dim)
        grid_hhgd2hgd = _standardize_grid(hhgd2hgd.get_grid(), dim)

        np.testing.assert_allclose(
            grid_hhgd2hgd,
            grid_hgd,
            rtol=0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            hhgd2hgd.grid_values,
            hgd.grid_values,
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
