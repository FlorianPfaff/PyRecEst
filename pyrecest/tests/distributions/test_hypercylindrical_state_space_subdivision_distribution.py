import unittest
from math import pi

import numpy.testing as npt
from pyrecest.backend import __backend_name__ as backend_name
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    array,
    squeeze,
)
from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import (
    CustomHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_distribution import (
    HypercylindricalStateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class HypercylindricalStateSpaceSubdivisionDistributionTest(unittest.TestCase):
    def setUp(self):
        self.vm = VonMisesDistribution(array(0.0), array(2.0))
        self.gauss = GaussianDistribution(array([1.0]), array([[0.25]]))
        self.n = 20

        def fun(x):
            return self.vm.pdf(x[:, 0]) * self.gauss.pdf(x[:, 1:])

        self.fun = fun
        self.chd = CustomHypercylindricalDistribution(fun, 1, 1)

    def test_from_distribution(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        self.assertIsInstance(hcrbd, HypercylindricalStateSpaceSubdivisionDistribution)
        self.assertEqual(hcrbd.bound_dim, 1)
        self.assertEqual(hcrbd.lin_dim, 1)

    def test_from_function(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            self.fun, self.n, 1, 1
        )
        self.assertIsInstance(hcrbd, HypercylindricalStateSpaceSubdivisionDistribution)

    def test_marginalize_linear(self):
        from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
            HypertoroidalGridDistribution,
        )

        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        gd_marginal = hcrbd.marginalize_linear()
        self.assertIsInstance(gd_marginal, HypertoroidalGridDistribution)

    def test_marginalize_periodic(self):
        from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture

        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        lin_marginal = hcrbd.marginalize_periodic()
        self.assertIsInstance(lin_marginal, LinearMixture)

    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_pdf_shape(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        test_points = array([[0.0, 1.0], [pi / 2, 0.5], [pi, 1.0]])
        p = hcrbd.pdf(test_points)
        self.assertEqual(array(p).shape, (3,))

    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_pdf_nonnegative(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        test_points = array([[0.0, 1.0], [pi / 2, 0.5], [pi, 1.0]])
        p = hcrbd.pdf(test_points)
        self.assertTrue(backend_all(p >= 0.0))

    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_pdf_approximation(self):
        """Test that the pdf approximation is reasonably close to the true pdf."""
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        test_points = array([[0.0, 1.0], [pi / 2, 0.5], [pi, 1.0], [3 * pi / 2, 1.5]])
        p_approx = hcrbd.pdf(test_points)
        p_true = self.chd.pdf(test_points)
        npt.assert_allclose(p_approx, p_true, rtol=0.3)

    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_mode_reasonable(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        m = hcrbd.mode()
        self.assertEqual(array(m).shape, (2,))
        # Mode of periodic part should be near 0 (von Mises mean)
        npt.assert_allclose(float(m[0]) % (2 * pi), 0.0, atol=2 * pi / self.n)
        # Mode of linear part should be near 1.0 (Gaussian mean)
        npt.assert_allclose(float(m[1]), 1.0, atol=0.1)

    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_sample_shape(self):
        from pyrecest.distributions.circle.circular_uniform_distribution import (
            CircularUniformDistribution,
        )
        from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
            HypertoroidalGridDistribution,
        )

        n_grid = 10
        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), (n_grid,)
        )
        lin_dists = [
            GaussianDistribution(array([0.0]), array([[1.0]])) for _ in range(n_grid)
        ]
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution(gd, lin_dists)
        samples = hcrbd.sample(50)
        self.assertEqual(array(samples).shape, (50, 2))

    @unittest.skipIf(
        backend_name != "numpy",
        reason="Not supported on this backend",
    )
    def test_pdf_integrates_to_one(self):
        """Coarse check that the pdf integrates close to 1."""
        from scipy.integrate import dblquad

        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            self.fun, self.n, 1, 1
        )
        integral, _ = dblquad(
            lambda y, x: float(squeeze(hcrbd.pdf(array([[x, y]])))),
            0,
            2 * pi,
            -4,
            6,
            epsabs=1e-2,
            epsrel=1e-2,
        )
        npt.assert_allclose(integral, 1.0, atol=0.05)


if __name__ == "__main__":
    unittest.main()
