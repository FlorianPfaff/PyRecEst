"""Tangent-space Gaussian distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
import numpy as np
from pyrecest.backend import (
    abs,
    all,
    allclose,
    amax,
    array,
    diag,
    exp,
    isfinite,
    linalg,
    log,
    matmul,
    ndim,
    pi,
    random,
    sum,
    transpose,
    zeros,
)

from ._so3_helpers import (
    as_batch,
    exp_map_identity,
    geodesic_distance,
    log_map_identity,
    normalize_quaternions,
    quaternion_conjugate,
    quaternion_multiply,
    quaternions_to_rotation_matrices,
    so3_exp_map_volume_log_jacobian,
)
from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution


def _validate_positive_sample_count(n) -> int:
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")

    try:
        count_int = int(count)
        count_float = float(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    if not np.isfinite(count_float) or not count_float.is_integer():
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


def _to_python_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


class SO3TangentGaussianDistribution(AbstractBoundedDomainDistribution):
    """Log-Gaussian approximation on SO(3).

    Rotations are represented as scalar-last unit quaternions ``(x, y, z, w)``.
    The distribution is parameterized by a Gaussian in the principal tangent
    chart at ``mu``.  Density evaluation maps rotations into that chart and
    divides by the SO(3) exponential-map volume element, so ``pdf`` and
    ``ln_pdf`` are densities with respect to the upper-unit-quaternion/Haar
    volume measure rather than raw Euclidean log-coordinate densities.

    This remains a local, single-chart approximation: samples whose tangent
    draws leave the principal ball are mapped through the exponential map and
    folded by quaternion canonicalization, not represented as an exact globally
    wrapped Gaussian.
    """

    def __init__(self, mu, C, check_validity=True):
        super().__init__(dim=3)
        normalized_mu = self._normalize_quaternions(mu)
        if normalized_mu.shape[0] != 1:
            raise ValueError("mu must be a single SO(3) quaternion.")
        self.mu = normalized_mu[0]

        C = array(C, dtype=float)
        if ndim(C) != 2 or C.shape != (3, 3):
            raise ValueError("C must have shape (3, 3).")
        if check_validity:
            if not _to_python_bool(all(isfinite(C))):
                raise ValueError("C must contain only finite values.")
            if not _to_python_bool(allclose(C, transpose(C))):
                raise ValueError("C must be symmetric.")
            if not _to_python_bool(all(linalg.eigvalsh(C) > 0.0)):
                raise ValueError("C must be positive definite.")
        self.C = C

    @property
    def input_dim(self):
        return 4

    _normalize_quaternions = staticmethod(normalize_quaternions)

    @staticmethod
    def _as_tangent_batch(tangent_vectors):
        return as_batch(tangent_vectors, 3, "SO(3) tangent vectors")

    _quaternion_conjugate = staticmethod(quaternion_conjugate)
    _quaternion_multiply = staticmethod(quaternion_multiply)

    @staticmethod
    def exp_map(tangent_vectors, base=None):
        """Map tangent vectors to SO(3) quaternions.

        If ``base`` is given, the returned rotations are ``base * Exp(v)``.
        """
        tangent_vectors = SO3TangentGaussianDistribution._as_tangent_batch(
            tangent_vectors
        )
        delta_quaternions = exp_map_identity(tangent_vectors)

        if base is None:
            return delta_quaternions
        return SO3TangentGaussianDistribution._quaternion_multiply(
            base, delta_quaternions
        )

    @staticmethod
    def log_map(rotations, base=None):
        """Map SO(3) quaternions to tangent vectors.

        If ``base`` is given, this returns ``Log(base^{-1} * rotations)``.
        """
        rotations = SO3TangentGaussianDistribution._normalize_quaternions(rotations)
        if base is not None:
            rotations = SO3TangentGaussianDistribution._quaternion_multiply(
                SO3TangentGaussianDistribution._quaternion_conjugate(base), rotations
            )

        return log_map_identity(rotations)

    geodesic_distance = staticmethod(geodesic_distance)

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        return quaternions_to_rotation_matrices(quaternions)

    def pdf(self, xs):
        """Evaluate the SO(3) volume density at quaternions."""
        return exp(self.ln_pdf(xs))

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the SO(3) volume density."""
        tangent_vectors = self.log_map(xs, base=self.mu)
        precision = linalg.inv(self.C)
        quadratic = sum(tangent_vectors * matmul(tangent_vectors, precision), axis=-1)
        log_det = 2.0 * sum(log(diag(linalg.cholesky(self.C))))
        tangent_log_density = -0.5 * (3.0 * log(2.0 * pi) + log_det + quadratic)
        return tangent_log_density - so3_exp_map_volume_log_jacobian(tangent_vectors)

    def tangent_vectors(self, rotations):
        """Return log-map coordinates of rotations around the distribution mean."""
        return self.log_map(rotations, base=self.mu)

    def sample_tangent(self, n):
        """Draw tangent-space Gaussian samples with shape ``(n, 3)``."""
        n = _validate_positive_sample_count(n)
        return random.multivariate_normal(mean=zeros(3), cov=self.C, size=n)

    def sample(self, n):
        """Draw ``n`` SO(3) samples as scalar-last unit quaternions."""
        return self.exp_map(self.sample_tangent(n), base=self.mu)

    def mean(self):
        """Return the mean rotation as a scalar-last unit quaternion."""
        return self.mu

    def mode(self):
        """Return the modal rotation as a scalar-last unit quaternion."""
        return self.mu

    def mean_rotation_matrix(self):
        """Return the mean rotation matrix."""
        return self.as_rotation_matrices(self.mu)[0]

    def covariance(self):
        """Return the 3-by-3 tangent covariance matrix."""
        return self.C

    def set_mean(self, new_mean):
        """Return a copy with a replaced mean rotation."""
        return self.set_mode(new_mean)

    def set_mode(self, new_mode):
        """Return a copy with a replaced modal rotation."""
        new_dist = self.__class__(new_mode, self.C, check_validity=False)
        return new_dist

    def get_manifold_size(self):
        """Return the embedding half-sphere volume used for unit quaternions."""
        return pi**2

    def is_valid(self, tolerance=1e-6):
        """Return whether the mean and covariance have valid SO(3) dimensions."""
        covariance_is_symmetric = amax(abs(self.C - transpose(self.C))) <= tolerance
        return bool(
            abs(linalg.norm(self.mu) - 1.0) <= tolerance
            and self.mu[-1] >= -tolerance
            and covariance_is_symmetric
        )

    @staticmethod
    def from_covariance_diagonal(mu, covariance_diagonal):
        """Create a tangent Gaussian from a diagonal covariance vector."""
        return SO3TangentGaussianDistribution(mu, diag(covariance_diagonal))
