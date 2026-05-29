# pylint: disable=no-name-in-module,no-member
import copy

# pylint: disable=redefined-builtin
from pyrecest.backend import abs, all, asarray, concatenate, dot, eye, isfinite, linalg
from pyrecest.distributions import GaussianDistribution

from .abstract_axial_filter import AbstractAxialFilter


class AxialKalmanFilter(AbstractAxialFilter):
    """Kalman Filter for directional estimation with antipodal symmetry.

    Works for antipodally symmetric complex numbers (2D unit vectors) and
    quaternions (4D unit vectors).

    References:
    - Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
      Recursive Bingham Filter for Directional Estimation Involving 180
      Degree Symmetry, Journal of Advances in Information Fusion,
      9(2):90-105, December 2014.
    """

    def __init__(self):
        from pyrecest.backend import array

        initial_state = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            eye(4),
        )
        AbstractAxialFilter.__init__(self, initial_state)
        self._set_composition_operator()

    @property
    def dim(self):
        """Manifold dimension (1 for complex/circle, 3 for quaternions)."""
        return self._filter_state.dim - 1

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        self._validate_axial_gaussian(new_state, "filter_state")
        self._filter_state = copy.deepcopy(new_state)
        self._set_composition_operator()

    @staticmethod
    def _validate_axial_gaussian(distribution, name):
        if not isinstance(distribution, GaussianDistribution):
            raise ValueError(f"{name} must be a GaussianDistribution.")
        if distribution.mu.shape[0] not in (2, 4):
            raise ValueError(f"{name} mean must have length 2 or 4.")
        if not bool(all(isfinite(distribution.mu))):
            raise ValueError(f"{name} mean must be finite.")
        if not bool(all(isfinite(distribution.C))):
            raise ValueError(f"{name} covariance must be finite.")
        if not bool(abs(linalg.norm(distribution.mu) - 1.0) < 1e-5):
            raise ValueError(f"{name} mean must be a unit vector.")

    def _validate_noise(self, noise, name):
        self._validate_axial_gaussian(noise, name)
        if noise.mu.shape != self._filter_state.mu.shape:
            raise ValueError(f"{name} mean shape must match the filter state.")

    def _as_measurement(self, z):
        measurement = asarray(z, dtype=float)
        if measurement.shape != self._filter_state.mu.shape:
            raise ValueError(
                "measurement must have the same shape as the filter state."
            )
        if not bool(all(isfinite(measurement))):
            raise ValueError("measurement must be finite.")
        if not bool(abs(linalg.norm(measurement) - 1.0) < 1e-5):
            raise ValueError("measurement must be a unit vector.")
        return measurement

    def predict_identity(self, gauss_w):
        """Predict assuming identity system model with noise gauss_w.

        Computes x(k+1) = x(k) ⊕ w(k), where ⊕ is complex or quaternion
        multiplication.

        Parameters:
            gauss_w (GaussianDistribution): system noise with unit vector mean
        """
        self._validate_noise(gauss_w, "system noise")
        mu_new = self.composition_operator(self._filter_state.mu, gauss_w.mu)
        C_new = self._filter_state.C + gauss_w.C
        self._filter_state = GaussianDistribution(mu_new, C_new, check_validity=False)

    def update_identity(self, gauss_v, z):
        """Update assuming identity measurement model with noise gauss_v.

        Computes z(k) = x(k) ⊕ v(k), where ⊕ is complex or quaternion
        multiplication.

        Parameters:
            gauss_v (GaussianDistribution): measurement noise with unit vector mean
            z (array): measurement as a unit vector of shape (2,) or (4,)
        """
        self._validate_noise(gauss_v, "measurement noise")
        z = self._as_measurement(z)

        # Conjugate of noise mean: negate all but the first component
        mu_v_conj = concatenate([gauss_v.mu[:1], -gauss_v.mu[1:]])
        z = self.composition_operator(mu_v_conj, z)

        if dot(z, self._filter_state.mu) < 0:
            z = -z

        d = self._filter_state.dim  # embedding dimension (2 or 4)
        IS = self._filter_state.C + gauss_v.C  # innovation covariance (H = I)
        K = linalg.solve(IS, self._filter_state.C).T  # Kalman gain: C @ inv(IS)
        mu_new = self._filter_state.mu + K @ (z - self._filter_state.mu)
        C_new = (eye(d) - K) @ self._filter_state.C

        mu_new = mu_new / linalg.norm(mu_new)  # enforce unit vector
        self._filter_state = GaussianDistribution(mu_new, C_new, check_validity=False)

    def get_point_estimate(self):
        """Return the mean of the current filter state."""
        return self._filter_state.mu
