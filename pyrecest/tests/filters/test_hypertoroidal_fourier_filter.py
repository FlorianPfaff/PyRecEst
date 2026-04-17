import unittest
import warnings

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, column_stack, linspace, meshgrid, pi
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)
from pyrecest.filters.hypertoroidal_fourier_filter import HypertoroidalFourierFilter


def _integrate_1d(hfd, n=200):
    """Numerical integral of a 1-D HFD over [0, 2*pi)."""
    xs = linspace(0.0, 2.0 * pi, n, endpoint=False)
    return float(hfd.pdf(xs).sum()) * float(2.0 * pi / n)


def _integrate_2d(hfd, n=60):
    """Numerical integral of a 2-D HFD over [0, 2*pi)^2."""
    x = linspace(0.0, 2.0 * pi, n, endpoint=False)
    y = linspace(0.0, 2.0 * pi, n, endpoint=False)
    X, Y = meshgrid(x, y, indexing="ij")
    pts = column_stack((X.flatten(), Y.flatten()))
    return float(hfd.pdf(pts).sum()) * float((2.0 * pi / n) ** 2)


class TestHypertoroidalFourierFilter(unittest.TestCase):
    # -----------------------------------------------------------------
    # Constructor / initial state
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_init_1d_sqrt(self):
        """1-D filter with sqrt transformation starts with a uniform HFD."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        self.assertEqual(f.filter_state.dim, 1)
        self.assertEqual(f.filter_state.coeff_mat.shape, (11,))
        self.assertEqual(f.filter_state.transformation, "sqrt")
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_init_1d_identity(self):
        """1-D filter with identity transformation starts with a uniform HFD."""
        f = HypertoroidalFourierFilter((11,), "identity")
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        self.assertEqual(f.filter_state.transformation, "identity")
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_init_int_arg(self):
        """Passing an int instead of a tuple creates a 1-D filter."""
        f = HypertoroidalFourierFilter(11)
        self.assertEqual(f.filter_state.dim, 1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_init_2d(self):
        """2-D filter starts with a uniform HFD."""
        f = HypertoroidalFourierFilter((11, 13), "sqrt")
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        self.assertEqual(f.filter_state.dim, 2)
        self.assertEqual(f.filter_state.coeff_mat.shape, (11, 13))
        npt.assert_allclose(_integrate_2d(f.filter_state), 1.0, atol=1e-3)

    # -----------------------------------------------------------------
    # filter_state setter
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_set_state_hfd(self):
        """Setting state with a matching HFD works without warnings."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        hfd = HypertoroidalFourierDistribution.from_distribution(wn, (11,), "sqrt")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.filter_state = hfd
        self.assertFalse(
            any("setState" in str(wi.message) for wi in w),
            "No setState warning expected for matching HFD",
        )
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_set_state_non_hfd_converts(self):
        """Setting state with a non-HFD distribution triggers a warning and converts."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.filter_state = wn
        self.assertTrue(any("setState:nonFourier" in str(wi.message) for wi in w))
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_set_state_different_transformation_warns(self):
        """Setting state with a differently transformed HFD triggers a warning."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        hfd_id = HypertoroidalFourierDistribution.from_distribution(
            wn, (11,), "identity"
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.filter_state = hfd_id
        self.assertTrue(any("setState:transDiffer" in str(wi.message) for wi in w))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_set_state_different_n_coeffs_warns(self):
        """Setting state with a different number of coefficients triggers a warning."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        hfd_other = HypertoroidalFourierDistribution.from_distribution(wn, (9,), "sqrt")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.filter_state = hfd_other
        self.assertTrue(any("setState:noOfCoeffsDiffer" in str(wi.message) for wi in w))

    # -----------------------------------------------------------------
    # predict_identity - 1D
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_predict_identity_1d_sqrt(self):
        """1-D predict_identity (sqrt): posterior is still normalized."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        hfd_prior = HypertoroidalFourierDistribution.from_distribution(
            wn, (11,), "sqrt"
        )
        f.filter_state = hfd_prior
        noise = WrappedNormalDistribution(array(0.0), array(0.3))
        noise_hfd = HypertoroidalFourierDistribution.from_distribution(
            noise, (11,), "sqrt"
        )
        f.predict_identity(noise_hfd)
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_predict_identity_1d_identity(self):
        """1-D predict_identity (identity): posterior is still normalized."""
        f = HypertoroidalFourierFilter((11,), "identity")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        hfd_prior = HypertoroidalFourierDistribution.from_distribution(
            wn, (11,), "identity"
        )
        f.filter_state = hfd_prior
        noise = WrappedNormalDistribution(array(0.0), array(0.3))
        noise_hfd = HypertoroidalFourierDistribution.from_distribution(
            noise, (11,), "identity"
        )
        f.predict_identity(noise_hfd)
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_predict_identity_1d_auto_converts_noise(self):
        """1-D predict_identity: non-HFD noise is converted automatically."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        noise = WrappedNormalDistribution(array(0.0), array(0.3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.predict_identity(noise)
        self.assertTrue(
            any("predict_identity:automaticConversion" in str(wi.message) for wi in w)
        )
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    # -----------------------------------------------------------------
    # predict_identity - 2D
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_predict_identity_2d_sqrt(self):
        """2-D predict_identity (sqrt): posterior is normalized."""
        f = HypertoroidalFourierFilter((11, 11), "sqrt")
        mu = array([1.0, 2.0])
        C = 0.5 * array([[1.0, 0.3], [0.3, 1.0]])
        twn = ToroidalWrappedNormalDistribution(mu, C)
        hfd_prior = HypertoroidalFourierDistribution.from_distribution(
            twn, (11, 11), "sqrt"
        )
        f.filter_state = hfd_prior
        noise_hfd = HypertoroidalFourierDistribution.from_distribution(
            ToroidalWrappedNormalDistribution(
                array([0.0, 0.0]), 0.2 * array([[1.0, 0.0], [0.0, 1.0]])
            ),
            (11, 11),
            "sqrt",
        )
        f.predict_identity(noise_hfd)
        npt.assert_allclose(_integrate_2d(f.filter_state), 1.0, atol=5e-3)

    # -----------------------------------------------------------------
    # update_identity - 1D
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_update_identity_1d_sqrt(self):
        """1-D update_identity (sqrt): posterior integrates to 1."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn, (11,), "sqrt"
        )
        meas_noise = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(array(0.0), array(1.0)), (11,), "sqrt"
        )
        f.update_identity(meas_noise, array([1.5]))
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_update_identity_1d_identity(self):
        """1-D update_identity (identity): posterior integrates to 1."""
        f = HypertoroidalFourierFilter((11,), "identity")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn, (11,), "identity"
        )
        meas_noise = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(array(0.0), array(1.0)), (11,), "identity"
        )
        f.update_identity(meas_noise, array([1.5]))
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_update_identity_1d_auto_converts_noise(self):
        """1-D update_identity: non-HFD noise is converted automatically."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        meas_noise = WrappedNormalDistribution(array(0.0), array(1.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            f.update_identity(meas_noise, array([1.5]))
        self.assertTrue(
            any("update_identity:automaticConversion" in str(wi.message) for wi in w)
        )
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    # -----------------------------------------------------------------
    # update_identity - 2D
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("jax", "pytorch"),
        reason="Not supported on this backend",
    )
    def test_update_identity_2d_sqrt(self):
        """2-D update_identity (sqrt): posterior is normalized."""
        f = HypertoroidalFourierFilter((11, 11), "sqrt")
        mu = array([1.0, 2.0])
        C = 0.5 * array([[1.0, 0.3], [0.3, 1.0]])
        hfd_prior = HypertoroidalFourierDistribution.from_distribution(
            ToroidalWrappedNormalDistribution(mu, C), (11, 11), "sqrt"
        )
        f.filter_state = hfd_prior
        meas_noise = HypertoroidalFourierDistribution.from_distribution(
            ToroidalWrappedNormalDistribution(
                array([0.0, 0.0]), 0.5 * array([[1.0, 0.0], [0.0, 1.0]])
            ),
            (11, 11),
            "sqrt",
        )
        f.update_identity(meas_noise, array([1.0, 2.0]))
        npt.assert_allclose(_integrate_2d(f.filter_state), 1.0, atol=5e-3)

    # -----------------------------------------------------------------
    # predict_nonlinear (identity system function)
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_nonlinear_identity_transform(self):
        """predict_nonlinear with identity transform: result is normalized."""
        f = HypertoroidalFourierFilter((11,), "identity")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, (11,), "identity"
        )
        noise = WrappedNormalDistribution(array(0.0), array(0.3))

        # Identity system: x_{k+1} = x_k (plus noise)
        def sys_fn(x_k):
            return x_k

        f.predict_nonlinear(sys_fn, noise)
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_nonlinear_sqrt_transform(self):
        """predict_nonlinear with sqrt transform: result is normalized."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, (11,), "sqrt"
        )
        noise = WrappedNormalDistribution(array(0.0), array(0.3))

        def sys_fn(x_k):
            return x_k

        f.predict_nonlinear(sys_fn, noise)
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-3)

    # -----------------------------------------------------------------
    # predict_nonlinear_via_transition_density
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_via_transition_density_identity(self):
        """Prediction via explicit transition density (identity transform)."""
        n_coeffs = (11,)
        f = HypertoroidalFourierFilter(n_coeffs, "identity")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, n_coeffs, "identity"
        )
        noise = WrappedNormalDistribution(array(0.0), array(0.3))
        hfd_trans = f.get_f_trans_as_hfd(lambda x: x, noise)
        f.predict_nonlinear_via_transition_density(hfd_trans)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_via_transition_density_sqrt(self):
        """Prediction via explicit transition density (sqrt transform)."""
        n_coeffs = (11,)
        f = HypertoroidalFourierFilter(n_coeffs, "sqrt")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, n_coeffs, "sqrt"
        )
        noise = WrappedNormalDistribution(array(0.0), array(0.3))
        hfd_trans = f.get_f_trans_as_hfd(lambda x: x, noise)
        f.predict_nonlinear_via_transition_density(hfd_trans, truncate_joint_sqrt=True)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_via_transition_density_sqrt_no_truncation(self):
        """Prediction via transition density without truncation (sqrt transform)."""
        n_coeffs = (11,)
        f = HypertoroidalFourierFilter(n_coeffs, "sqrt")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, n_coeffs, "sqrt"
        )
        noise = WrappedNormalDistribution(array(0.0), array(0.3))
        hfd_trans = f.get_f_trans_as_hfd(lambda x: x, noise)
        f.predict_nonlinear_via_transition_density(hfd_trans, truncate_joint_sqrt=False)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-3)

    # -----------------------------------------------------------------
    # update_nonlinear
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_update_nonlinear_with_hfd(self):
        """update_nonlinear with a pre-computed HFD likelihood."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, (11,), "sqrt"
        )
        # Likelihood: uniform (should leave prior unchanged up to normalization)
        likelihood_hfd = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(array(0.0), array(2.0)), (11,), "sqrt"
        )
        f.update_nonlinear(likelihood_hfd)
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_nonlinear_with_function_and_measurement(self):
        """update_nonlinear with a likelihood function and measurement."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn_prior = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn_prior, (11,), "sqrt"
        )

        # Likelihood: wrapped normal noise with measurement z=1.5
        meas_noise = WrappedNormalDistribution(array(0.0), array(1.0))

        def likelihood_fn(z_mat, x_mat):
            # z_mat and x_mat are (dim=1, n_pts) arrays
            diff = (z_mat - x_mat).ravel()  # (n_pts,)
            return meas_noise.pdf(diff)

        z = array([1.5])
        f.update_nonlinear(likelihood_fn, z)
        self.assertIsInstance(f.filter_state, HypertoroidalFourierDistribution)
        npt.assert_allclose(_integrate_1d(f.filter_state), 1.0, atol=1e-4)

    # -----------------------------------------------------------------
    # get_point_estimate
    # -----------------------------------------------------------------

    @unittest.skipIf(
        pyrecest.backend.__backend_name__
        in ("jax", "pytorch"),  # pylint: disable=no-member
        reason="Not supported on this backend",
    )
    def test_get_point_estimate_1d(self):
        """get_point_estimate returns a value in [0, 2*pi)."""
        f = HypertoroidalFourierFilter((11,), "sqrt")
        wn = WrappedNormalDistribution(array(1.0), array(0.5))
        f.filter_state = HypertoroidalFourierDistribution.from_distribution(
            wn, (11,), "sqrt"
        )
        est = f.get_point_estimate()
        self.assertTrue(0.0 <= float(est.flat[0]) < 2.0 * float(pi))


if __name__ == "__main__":
    unittest.main()
