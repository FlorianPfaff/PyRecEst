# pylint: disable=no-name-in-module,no-member
import copy

import pyrecest.backend
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    allclose,
    array,
    asarray,
    diag,
    eye,
    isclose,
    isfinite,
    linalg,
)
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)

from .abstract_filter import AbstractFilter


def _to_python_bool(value):
    """Convert scalar backend booleans to Python bools for validation."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _validate_bingham_distribution(distribution, role):
    if not isinstance(distribution, BinghamDistribution):
        raise ValueError(f"{role} must be a BinghamDistribution.")
    if distribution.dim not in (1, 3):
        raise ValueError(f"{role} must be a 2D or 4D Bingham distribution.")

    input_dim = distribution.dim + 1
    Z = asarray(distribution.Z)
    M = asarray(distribution.M)
    if Z.shape != (input_dim,):
        raise ValueError(f"{role} Z must have shape ({input_dim},).")
    if M.shape != (input_dim, input_dim):
        raise ValueError(f"{role} M must have shape ({input_dim}, {input_dim}).")
    if not _to_python_bool(backend_all(isfinite(Z))):
        raise ValueError(f"{role} Z must be finite.")
    if not _to_python_bool(backend_all(isfinite(M))):
        raise ValueError(f"{role} M must be finite.")
    if not _to_python_bool(isclose(Z[-1], 0.0)):
        raise ValueError(f"{role} last concentration must be zero.")
    if not _to_python_bool(backend_all(Z[:-1] <= Z[1:])):
        raise ValueError(f"{role} concentrations must be sorted ascending.")
    if not _to_python_bool(allclose(M @ M.T, eye(input_dim), atol=1e-3)):
        raise ValueError(f"{role} M must be orthogonal.")


def _validate_compatible_bingham(distribution, reference, role):
    _validate_bingham_distribution(distribution, role)
    if distribution.dim != reference.dim:
        raise ValueError(f"{role} dimension must match the filter state dimension.")


def _validate_bingham_measurement(z, input_dim):
    measurement = asarray(z).ravel()
    if measurement.shape != (input_dim,):
        raise ValueError(f"measurement z must have shape ({input_dim},).")
    if not _to_python_bool(backend_all(isfinite(measurement))):
        raise ValueError("measurement z must be finite.")
    if not _to_python_bool(isclose(linalg.norm(measurement), 1.0)):
        raise ValueError("measurement z must be a unit vector.")
    return measurement


class BinghamFilter(AbstractFilter):
    """Recursive filter based on the Bingham distribution.

    Supports antipodally symmetric complex numbers (2D) and quaternions (4D).

    References:
    - Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
      Recursive Bingham Filter for Directional Estimation Involving 180
      Degree Symmetry, Journal of Advances in Information Fusion,
      9(2):90-105, December 2014.
    - Igor Gilitschenski, Gerhard Kurz, Simon J. Julier, Uwe D. Hanebeck,
      Unscented Orientation Estimation Based on the Bingham Distribution,
      IEEE Transactions on Automatic Control, January 2016.
    """

    def __init__(self):
        if pyrecest.backend.__backend_name__ == "jax":
            raise NotImplementedError(
                "BinghamFilter is not supported on the JAX backend"
            )
        # Default 4-D identity initial state (uniform on S^3, suitable for quaternion orientation)
        initial_state = BinghamDistribution(
            array([-1.0, -1.0, -1.0, 0.0]),
            array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
            ),
        )
        AbstractFilter.__init__(self, initial_state)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        _validate_bingham_distribution(new_state, "filter_state")
        self._filter_state = copy.deepcopy(new_state)

    def predict_identity(self, bw):
        """Predict assuming identity system model with Bingham noise.

        Computes x(k+1) = x(k) (*) w(k) where (*) is complex or quaternion
        multiplication and w(k) ~ bw.

        Parameters:
            bw (BinghamDistribution): noise distribution
        """
        _validate_compatible_bingham(bw, self.filter_state, "system noise")
        self.filter_state = self.filter_state.compose(bw)

    def predict_nonlinear(self, a, bw):
        """Predict assuming nonlinear system model with Bingham noise.

        Computes x(k+1) = a(x(k)) (*) w(k) using a sigma-point approximation.

        Parameters:
            a (callable): nonlinear system function mapping R^n -> R^n
            bw (BinghamDistribution): noise distribution
        """
        if not callable(a):
            raise ValueError("system function must be callable.")
        _validate_compatible_bingham(bw, self.filter_state, "system noise")

        samples, weights = self.filter_state.sample_deterministic(0.5)

        # Propagate each sample through the system function
        for i in range(len(weights)):
            samples[:, i] = a(samples[:, i])

        # Compute scatter matrix of propagated samples
        S = samples @ diag(weights) @ samples.T
        S = (S + S.T) / 2

        predicted = BinghamDistribution.fit_to_moment(S)
        self.filter_state = predicted.compose(bw)

    def update_identity(self, bv, z):
        """Update assuming identity measurement model with Bingham noise.

        Applies the measurement z using likelihood based on Bingham noise bv.

        Parameters:
            bv (BinghamDistribution): measurement noise distribution
            z (numpy.ndarray): measurement as a unit vector of shape (dim+1,)
        """
        _validate_compatible_bingham(bv, self.filter_state, "measurement noise")
        z = _validate_bingham_measurement(z, self.filter_state.input_dim)

        bv = copy.deepcopy(bv)
        n = bv.input_dim
        for i in range(n):
            m_conj = self._conjugate(bv.M[:, i])
            bv.M[:, i] = self._compose(z, m_conj)

        self.filter_state = self.filter_state.multiply(bv)

    def get_point_estimate(self):
        """Return the mode of the current distribution as a point estimate."""
        return self.filter_state.mode()

    @staticmethod
    def _conjugate(q):
        """Return the conjugate of a unit complex number or quaternion.

        For q = [w, x, y, z], conjugate = [w, -x, -y, -z].
        For q = [a, b], conjugate = [a, -b].
        """
        result = copy.copy(q)
        result[1:] = -result[1:]
        return result

    @staticmethod
    def _compose(q1, q2):
        """Compose two unit complex numbers or quaternions via multiplication.

        Parameters:
            q1, q2: unit vectors of length 2 or 4

        Returns:
            product q1 * q2
        """
        if q1.shape[0] == 2:
            # Complex multiplication
            return array(
                [
                    q1[0] * q2[0] - q1[1] * q2[1],
                    q1[0] * q2[1] + q1[1] * q2[0],
                ]
            )
        # Hamilton quaternion product
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
