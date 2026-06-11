import unittest

import numpy.testing as npt
import pyrecest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, linalg, pi
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class TestStateSpaceSubdivisionGaussianDistribution(unittest.TestCase):
    @staticmethod
    def _make_grid(n):
        return HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), (n,)
        )

    @staticmethod
    def _make_gaussians(n, dim=1):
        return [GaussianDistribution(array([0.0] * dim), eye(dim)) for _ in range(n)]

    def test_constructor_rejects_non_gaussian_conditionals(self):
        gd = self._make_grid(2)

        with self.assertRaisesRegex(TypeError, "GaussianDistribution"):
            StateSpaceSubdivisionGaussianDistribution(
                gd, [GaussianDistribution(array([0.0]), eye(1)), object()]
            )

    def test_constructor_rejects_grid_length_mismatch(self):
        gd = self._make_grid(3)

        with self.assertRaisesRegex(ValueError, "Number of grid points"):
            StateSpaceSubdivisionGaussianDistribution(gd, self._make_gaussians(2))

    def test_constructor_rejects_mismatched_linear_dimensions(self):
        gd = self._make_grid(2)
        gaussians = [
            GaussianDistribution(array([0.0]), eye(1)),
            GaussianDistribution(array([0.0, 0.0]), eye(2)),
        ]

        with self.assertRaisesRegex(ValueError, "same dimension"):
            StateSpaceSubdivisionGaussianDistribution(gd, gaussians)

    def test_multiply_rejects_incompatible_operands(self):
        gd = self._make_grid(2)
        rbd = StateSpaceSubdivisionGaussianDistribution(gd, self._make_gaussians(2))

        with self.assertRaisesRegex(
            TypeError, "StateSpaceSubdivisionGaussianDistribution"
        ):
            rbd.multiply(object())

        other_count = StateSpaceSubdivisionGaussianDistribution(
            self._make_grid(3), self._make_gaussians(3)
        )
        with self.assertRaisesRegex(ValueError, "same number"):
            rbd.multiply(other_count)

        gd_shifted = HypertoroidalGridDistribution(
            array([1.0, 1.0]),
            grid_type="custom",
            grid=array([[0.1], [pi + 0.1]]),
        )
        other_grid = StateSpaceSubdivisionGaussianDistribution(
            gd_shifted, self._make_gaussians(2)
        )
        with self.assertRaisesRegex(ValueError, "equal grids"):
            rbd.multiply(other_grid)

    def test_multiply_s1_x_r1_identical_precise(self):
        """Multiply two S1xR1 distributions; linear uncertainty must decrease."""
        n = 100
        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), (n,)
        )
        gaussians = [
            GaussianDistribution(array([0.0]), array([[1.0]])) for _ in range(n)
        ]
        rbd1 = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)

        gaussians2 = [
            GaussianDistribution(array([2.0]), array([[1.0]])) for _ in range(n)
        ]
        rbd2 = StateSpaceSubdivisionGaussianDistribution(gd, gaussians2)

        rbd_up = rbd1.multiply(rbd2)

        for i in range(n):
            npt.assert_array_less(
                linalg.det(rbd_up.linear_distributions[i].C),
                linalg.det(rbd1.linear_distributions[i].C),
            )
            npt.assert_allclose(
                rbd_up.linear_distributions[i].mu, array([1.0]), atol=1e-14
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_multiply_s2_x_r3_rough(self):
        """Multiply two S2xR3 distributions; linear uncertainty must decrease."""
        n = 100
        gd = HyperhemisphericalGridDistribution.from_distribution(
            HyperhemisphericalUniformDistribution(2), n, "leopardi_symm"
        )
        gaussians = [
            GaussianDistribution(array([0.0, 0.0, 0.0]), 1000.0 * eye(3))
            for _ in range(n)
        ]
        rbd1 = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)

        gaussians2 = [
            GaussianDistribution(array([2.0, 2.0, 2.0]), 1000.0 * eye(3))
            for _ in range(n)
        ]
        rbd2 = StateSpaceSubdivisionGaussianDistribution(gd, gaussians2)

        rbd_up = rbd1.multiply(rbd2)

        for i in range(n):
            npt.assert_array_less(
                linalg.det(rbd_up.linear_distributions[i].C),
                linalg.det(rbd1.linear_distributions[i].C),
            )
            npt.assert_allclose(
                rbd_up.linear_distributions[i].mu,
                array([1.0, 1.0, 1.0]),
                atol=1e-10,
            )

    def test_hybrid_mean(self):
        """hybridMean returns concatenation of periodic and linear means."""
        n = 100
        mu_periodic = 4.0
        mu_linear = array([1.0, 2.0, 3.0])
        gd = HypertoroidalGridDistribution.from_distribution(
            VonMisesDistribution(mu_periodic, 1.0), (n,)
        )
        gaussians = [GaussianDistribution(mu_linear, 1000.0 * eye(3)) for _ in range(n)]
        rbd = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
        npt.assert_allclose(
            rbd.hybrid_mean(),
            array([mu_periodic, 1.0, 2.0, 3.0]),
            atol=1e-4,
        )

    def test_linear_mean(self):
        """linearMean returns the correct linear mean."""
        n = 100
        mu_linear = array([1.0, 2.0, 3.0])
        gd = HypertoroidalGridDistribution.from_distribution(
            VonMisesDistribution(4.0, 1.0), (n,)
        )
        gaussians = [GaussianDistribution(mu_linear, 1000.0 * eye(3)) for _ in range(n)]
        rbd = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
        npt.assert_allclose(rbd.linear_mean(), mu_linear, rtol=5e-7)

    def test_mode_warning_uniform(self):
        """mode() warns about potential multimodality for a uniform periodic part."""
        n = 100
        mu_linear = array([1.0, 2.0, 3.0])
        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), (n,)
        )
        gaussians = [GaussianDistribution(mu_linear, 1000.0 * eye(3)) for _ in range(n)]
        rbd = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
        with self.assertWarns(UserWarning):
            rbd.mode()

    def test_mode(self):
        """mode() returns the correct mode without warning for a unimodal density."""
        n = 100
        mu_periodic = 4.0
        mu_linear = array([1.0, 2.0, 3.0])
        gd = HypertoroidalGridDistribution.from_distribution(
            VonMisesDistribution(mu_periodic, 10.0), (n,)
        )
        gaussians = [GaussianDistribution(mu_linear, eye(3)) for _ in range(n)]
        rbd = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)

        # Should not warn
        with self.assertNoLogs(level="WARNING"):
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("error", UserWarning)
                m = rbd.mode()

        # Mode should be close to [mu_periodic, mu_linear]; tolerance is grid resolution
        npt.assert_allclose(
            m,
            array([mu_periodic, 1.0, 2.0, 3.0]),
            atol=pi / n,
        )


if __name__ == "__main__":
    unittest.main()
