import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, pi
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


def _make_s1_x_r1_state(n=20, mu_lin=None, cov_scale=1.0, periodic_dist=None):
    """Helper: create an S1×R1 StateSpaceSubdivisionGaussianDistribution."""
    if periodic_dist is None:
        periodic_dist = CircularUniformDistribution()
    gd = HypertoroidalGridDistribution.from_distribution(periodic_dist, (n,))
    if mu_lin is None:
        mu_lin = array([0.0])
    gaussians = [
        GaussianDistribution(mu_lin, cov_scale * eye(len(mu_lin))) for _ in range(n)
    ]
    return StateSpaceSubdivisionGaussianDistribution(gd, gaussians)


class TestStateSpaceSubdivisionFilterInit(unittest.TestCase):
    def test_init(self):
        state = _make_s1_x_r1_state()
        f = StateSpaceSubdivisionFilter(state)
        self.assertIsInstance(f.filter_state, StateSpaceSubdivisionGaussianDistribution)

    def test_set_state(self):
        state = _make_s1_x_r1_state()
        f = StateSpaceSubdivisionFilter(state)
        new_state = _make_s1_x_r1_state(mu_lin=array([3.0]))
        f.filter_state = new_state
        npt.assert_allclose(f.filter_state.linear_distributions[0].mu, array([3.0]))

    def test_set_state_different_size_warns(self):
        state = _make_s1_x_r1_state(n=20)
        f = StateSpaceSubdivisionFilter(state)
        new_state = _make_s1_x_r1_state(n=30)
        with self.assertWarns(UserWarning):
            f.filter_state = new_state


class TestStateSpaceSubdivisionFilterPredictLinear(unittest.TestCase):
    def test_predict_nothing_warns(self):
        state = _make_s1_x_r1_state()
        f = StateSpaceSubdivisionFilter(state)
        with self.assertWarns(UserWarning):
            f.predict_linear()

    def test_predict_case2_identity_system_no_change(self):
        """Case 2 with F=I, Q=0, u=0 should leave the state unchanged."""
        n = 20
        mu0 = array([1.0, 2.0])
        state = _make_s1_x_r1_state(n=n, mu_lin=mu0)
        f = StateSpaceSubdivisionFilter(state)

        F = eye(2)
        f.predict_linear(system_matrices=F)

        for ld in f.filter_state.linear_distributions:
            npt.assert_allclose(ld.mu, mu0, atol=1e-12)

    def test_predict_case2_add_noise_increases_covariance(self):
        """Case 2: adding noise covariance should increase each C."""
        n = 20
        mu0 = array([1.0])
        sigma0 = array([[1.0]])
        state = _make_s1_x_r1_state(n=n, mu_lin=mu0)
        f = StateSpaceSubdivisionFilter(state)
        Q = array([[0.5]])
        f.predict_linear(covariance_matrices=Q)
        for ld in f.filter_state.linear_distributions:
            npt.assert_allclose(ld.C, sigma0 + Q, atol=1e-12)

    def test_predict_case2_system_matrix(self):
        """Case 2: system matrix should transform the mean."""
        n = 20
        mu0 = array([1.0, 2.0])
        state = _make_s1_x_r1_state(n=n, mu_lin=mu0)
        f = StateSpaceSubdivisionFilter(state)

        F = array([[2.0, 0.0], [0.0, 3.0]])
        f.predict_linear(system_matrices=F)
        expected_mu = F @ mu0
        for ld in f.filter_state.linear_distributions:
            npt.assert_allclose(ld.mu, expected_mu, atol=1e-12)

    def test_predict_case2_input_vector(self):
        """Case 2: input vector should shift the mean."""
        n = 20
        mu0 = array([0.0])
        u = array([5.0])
        state = _make_s1_x_r1_state(n=n, mu_lin=mu0)
        f = StateSpaceSubdivisionFilter(state)
        f.predict_linear(linear_input_vectors=u)
        for ld in f.filter_state.linear_distributions:
            npt.assert_allclose(ld.mu, mu0 + u, atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_case3_identity_transition(self):
        """Case 3 with an identity-like transition should preserve the mean."""
        n = 20
        mu0 = array([1.0])

        gd = HypertoroidalGridDistribution.from_distribution(
            VonMisesDistribution(1.0, 10.0), (n,)
        )
        gaussians = [GaussianDistribution(mu0, eye(1)) for _ in range(n)]
        state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
        f = StateSpaceSubdivisionFilter(state)

        # Build an identity-like transition: f(next=i | current=j) = delta(i==j)
        # Approximate with a very narrow wrapped normal
        def identity_transition(xa, xb):
            # Very sharp wrapped normal centred on xa=xb
            diff = xa[:, 0] - xb[:, 0]
            from pyrecest.backend import (  # pylint: disable=import-outside-toplevel,no-name-in-module,no-member
                exp,
            )

            return exp(-(diff**2) / (2 * 0.001**2)) / (0.001 * (2 * pi) ** 0.5)

        td = TdCondTdGridDistribution.from_function(
            identity_transition,
            n,
            fun_does_cartesian_product=False,
            dim=2,
        )

        f.predict_linear(transition_density=td)

        # The periodic marginal should remain roughly the same (not tested in detail)
        # The linear means should still be approximately mu0
        for ld in f.filter_state.linear_distributions:
            npt.assert_allclose(ld.mu, mu0, atol=0.1)


class TestStateSpaceSubdivisionFilterUpdate(unittest.TestCase):
    def test_update_nothing_warns(self):
        state = _make_s1_x_r1_state()
        f = StateSpaceSubdivisionFilter(state)
        with self.assertWarns(UserWarning):
            f.update()

    def test_update_periodic_only(self):
        """Updating with a periodic likelihood only changes grid weights."""
        n = 20
        mu0 = array([0.0])
        state = _make_s1_x_r1_state(n=n, mu_lin=mu0)
        f = StateSpaceSubdivisionFilter(state)
        mu_periodic = 1.0
        f.update(likelihood_periodic_grid=VonMisesDistribution(mu_periodic, 5.0))
        # Linear means should be unchanged
        for ld in f.filter_state.linear_distributions:
            npt.assert_allclose(ld.mu, mu0, atol=1e-12)
        # Periodic mean should shift toward mu_periodic
        est = f.get_point_estimate()
        self.assertAlmostEqual(float(est[0]), mu_periodic, delta=0.2)

    def test_update_linear_only_reduces_uncertainty(self):
        """Updating with a linear likelihood should reduce covariance."""
        n = 20
        mu0 = array([0.0])
        C0 = eye(1)
        state = _make_s1_x_r1_state(n=n, mu_lin=mu0, cov_scale=1.0)
        f = StateSpaceSubdivisionFilter(state)

        likelihood = GaussianDistribution(array([2.0]), eye(1))
        f.update(likelihoods_linear=[likelihood])

        for ld in f.filter_state.linear_distributions:
            # Covariance should have decreased
            self.assertLess(float(ld.C[0, 0]), float(C0[0, 0]))
            # Mean should be halfway between 0 and 2
            npt.assert_allclose(ld.mu, array([1.0]), atol=1e-10)

    def test_update_joint(self):
        """Joint update: both periodic and linear likelihoods should narrow the state."""
        n = 40
        mu_lin = array([3.0])
        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), (n,)
        )
        gaussians = [GaussianDistribution(mu_lin, 10.0 * eye(1)) for _ in range(n)]
        state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
        f = StateSpaceSubdivisionFilter(state)

        mu_periodic = 2.0
        mu_lin_meas = array([5.0])
        periodic_like = VonMisesDistribution(mu_periodic, 10.0)
        linear_like = GaussianDistribution(mu_lin_meas, 10.0 * eye(1))

        f.update(
            likelihood_periodic_grid=periodic_like,
            likelihoods_linear=[linear_like],
        )

        est = f.get_point_estimate()
        # Periodic estimate should be close to mu_periodic
        npt.assert_allclose(float(est[0]), mu_periodic, atol=0.3)
        # Linear estimate should be halfway between prior mean and measurement
        npt.assert_allclose(
            float(est[1]), (float(mu_lin[0]) + float(mu_lin_meas[0])) / 2.0, atol=0.5
        )

    def test_get_estimate_returns_distribution(self):
        state = _make_s1_x_r1_state()
        f = StateSpaceSubdivisionFilter(state)
        est = f.get_estimate()
        self.assertIsInstance(est, StateSpaceSubdivisionGaussianDistribution)

    def test_get_point_estimate_shape(self):
        n = 20
        mu_lin = array([1.0, 2.0, 3.0])
        gd = HypertoroidalGridDistribution.from_distribution(
            VonMisesDistribution(1.0, 2.0), (n,)
        )
        gaussians = [GaussianDistribution(mu_lin, eye(3)) for _ in range(n)]
        state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
        f = StateSpaceSubdivisionFilter(state)
        pt = f.get_point_estimate()
        # S1 (1 angle) + R3 (3 linear) = 4-dimensional point estimate
        self.assertEqual(len(pt), 4)


if __name__ == "__main__":
    unittest.main()
