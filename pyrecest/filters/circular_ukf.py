"""
A modified unscented Kalman filter for circular distributions,
interprets circle as 1D interval [0, 2*pi).

References:
    Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
    Recursive Bayesian Filtering in Circular State Spaces
    arXiv preprint: Systems and Control (cs.SY), January 2015.
"""

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

from ._ukf import MerweScaledSigmaPoints, UnscentedKalmanFilter, _UKFModel

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, atleast_1d, mod, pi, sign
from pyrecest.distributions import GaussianDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin


def _make_ukf(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    fx, hx, dim_z, x0, P0, Q, R, alpha=1e-3, beta=2.0, kappa=0.0
):
    """Helper to build an UnscentedKalmanFilter.

    Parameters
    ----------
    R:
        Measurement noise covariance matrix of shape (dim_z, dim_z).
    alpha, beta, kappa:
        Sigma-point parameters for :class:`MerweScaledSigmaPoints`.
    """
    points = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=beta, kappa=kappa)
    ukf = UnscentedKalmanFilter(
        _UKFModel(dim_x=1, dim_z=dim_z, dt=1.0, hx=hx, fx=fx, points=points)
    )
    ukf.x = array([x0])
    ukf.P = array([[P0]])
    ukf.Q = array([[Q]])
    ukf.R = R  # R must already be a (dim_z, dim_z) array
    return ukf


class CircularUKF(AbstractFilter, CircularFilterMixin):
    """
    A modified unscented Kalman filter for circular distributions.

    The state is represented as a 1-D :class:`GaussianDistribution` on the
    circle [0, 2*pi).
    """

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """Initialise with a standard Gaussian at 0 with unit variance.

        Parameters
        ----------
        alpha:
            UKF sigma-point spread parameter (default 1e-3).
        beta:
            UKF prior distribution parameter (default 2.0, optimal for Gaussian).
        kappa:
            UKF secondary scaling parameter (default 0.0).
        """
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        initial_state = GaussianDistribution(array([0.0]), array([[1.0]]))
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property
    # ------------------------------------------------------------------

    @property
    def filter_state(self) -> GaussianDistribution:
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if not isinstance(new_state, GaussianDistribution):
            new_state = GaussianDistribution.from_distribution(new_state)
        assert new_state.dim == 1, "CircularUKF only supports 1-D state"
        self._filter_state = new_state

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, gauss_sys: GaussianDistribution):
        """
        Predict assuming identity system model:
            x(k+1) = x(k) + w(k)  mod 2*pi
        where *w(k)* is additive noise described by *gauss_sys*.

        Parameters
        ----------
        gauss_sys:
            Distribution of additive system noise (converted to
            :class:`GaussianDistribution` if necessary).
        """
        if not isinstance(gauss_sys, GaussianDistribution):
            gauss_sys = GaussianDistribution.from_distribution(gauss_sys)
        new_mu = mod(self._filter_state.mu + gauss_sys.mu, 2.0 * pi)
        new_C = self._filter_state.C + gauss_sys.C
        self._filter_state = GaussianDistribution(new_mu, new_C)

    def predict_nonlinear(self, f, gauss_sys: GaussianDistribution):
        """
        Predict assuming a nonlinear system model:
            x(k+1) = f(x(k)) + w(k)  mod 2*pi
        where *w(k)* is additive noise described by *gauss_sys*.

        Parameters
        ----------
        f:
            Function from [0, 2*pi) to [0, 2*pi).
        gauss_sys:
            Distribution of additive system noise.
        """
        if not isinstance(gauss_sys, GaussianDistribution):
            gauss_sys = GaussianDistribution.from_distribution(gauss_sys)

        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"

        mu0 = float(self._filter_state.mu[0])
        C0 = float(self._filter_state.C[0, 0])
        Q_val = float(gauss_sys.C[0, 0])
        noise_mean = float(gauss_sys.mu[0])

        def fx(x, dt):  # pylint: disable=unused-argument
            return array([f(x.flatten()[0]) + noise_mean])

        def hx(x):
            return x

        ukf = _make_ukf(
            fx,
            hx,
            dim_z=1,
            x0=mu0,
            P0=C0,
            Q=Q_val,
            R=array([[C0]]),
            alpha=self._alpha,
            beta=self._beta,
            kappa=self._kappa,
        )
        ukf.predict()

        new_mu = mod(array([ukf.x.flatten()[0]]), 2.0 * pi)
        new_C = array([[ukf.P[0, 0]]])
        self._filter_state = GaussianDistribution(new_mu, new_C)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, gauss_meas: GaussianDistribution, z):
        """
        Update assuming identity measurement model:
            z(k) = x(k) + v(k)  mod 2*pi
        where *v(k)* is additive noise described by *gauss_meas*.

        Parameters
        ----------
        gauss_meas:
            Distribution of additive measurement noise.
        z:
            Scalar measurement in [0, 2*pi).
        """
        if not isinstance(gauss_meas, GaussianDistribution):
            gauss_meas = GaussianDistribution.from_distribution(gauss_meas)

        z_val = float(array(z).flatten()[0])
        # Shift measurement by noise mean
        z_val = float(mod(array([z_val - float(gauss_meas.mu[0])]), 2.0 * pi)[0])

        mu = float(self._filter_state.mu[0])
        # Move measurement if necessary (wrap to be nearest to current mean)
        if abs(mu - z_val) > float(pi):
            z_val = z_val + 2.0 * float(pi) * sign(mu - z_val)

        C = float(self._filter_state.C[0, 0])
        R = float(gauss_meas.C[0, 0])

        # Kalman update
        K = C / (C + R)
        new_mu = mu + K * (z_val - mu)
        new_C = (1.0 - K) * C

        new_mu = float(mod(array([new_mu]), 2.0 * pi)[0])
        self._filter_state = GaussianDistribution(array([new_mu]), array([[new_C]]))

    def update_nonlinear(  # pylint: disable=too-many-locals
        self, f, gauss_meas: GaussianDistribution, z, measurement_periodic: bool = False
    ):
        """
        Update assuming a nonlinear measurement model:
            z(k) = f(x(k)) + v(k)           (if *measurement_periodic* is False)
            z(k) = f(x(k)) + v(k)  mod 2*pi (if *measurement_periodic* is True)
        where *v(k)* is additive noise described by *gauss_meas*.

        Parameters
        ----------
        f:
            Measurement function from [0, 2*pi) to R^d.
        gauss_meas:
            Distribution of additive measurement noise.
        z:
            Measurement vector of shape (d,).
        measurement_periodic:
            Whether the measurement is a periodic quantity.
        """
        if not isinstance(gauss_meas, GaussianDistribution):
            gauss_meas = GaussianDistribution.from_distribution(gauss_meas)

        assert pyrecest.backend.__backend_name__ not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"

        z = atleast_1d(array(z, dtype=float))
        dim_z = len(z.flatten())

        mu0 = float(self._filter_state.mu[0])
        C0 = float(self._filter_state.C[0, 0])
        pi_val = float(pi)

        if measurement_periodic:
            for i in range(dim_z):
                z_i = float(z[i])
                if abs(mu0 - z_i) > pi_val:
                    z[i] = z_i + 2.0 * pi_val * float(sign(mu0 - z_i))

        if dim_z == 1:
            R_mat = array([[float(gauss_meas.C.flatten()[0])]])
        else:
            R_mat = array(gauss_meas.C, dtype=float).reshape(dim_z, dim_z)

        def fx(x, dt):  # pylint: disable=unused-argument
            return x

        def hx(x):
            return atleast_1d(array([f(x.flatten()[0])], dtype=float))

        ukf = _make_ukf(
            fx,
            hx,
            dim_z=dim_z,
            x0=mu0,
            P0=C0,
            Q=0.0,
            R=R_mat,
            alpha=self._alpha,
            beta=self._beta,
            kappa=self._kappa,
        )
        # predict() with identity fx and Q=0 populates sigmas_f without
        # altering the mean or covariance, which is required before update().
        ukf.predict()
        ukf.update(z)

        new_mu = float(mod(array([ukf.x.flatten()[0]]), 2.0 * pi)[0])
        self._filter_state = GaussianDistribution(
            array([new_mu]), array([[ukf.P[0, 0]]])
        )

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the mean of the current state estimate."""
        return self._filter_state.mu
