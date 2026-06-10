"""Tangent-space Gaussian distribution on Cartesian products of SO(3)."""

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
    reshape,
    stack,
    sum,
    take,
    transpose,
    zeros,
)

from ._so3_helpers import (
    exp_map_identity,
)
from ._so3_helpers import geodesic_distance as so3_geodesic_distance
from ._so3_helpers import (
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


class SO3ProductTangentGaussianDistribution(AbstractBoundedDomainDistribution):
    """Log-Gaussian approximation on Cartesian products of SO(3).

    Rotations are represented as scalar-last unit quaternions ``(x, y, z, w)``.
    The mean is stored as a product point with shape ``(K, 4)`` and covariance is
    a full matrix on the flattened tangent vector in ``R^(3K)``.  Density
    evaluation divides the tangent Gaussian by the product SO(3) exponential-map
    volume element, so values are densities with respect to the product
    upper-unit-quaternion/Haar volume measure.
    """

    def __init__(self, mu, C, num_rotations=None, check_validity=True):
        mean, inferred_num_rotations = self._as_product_point(
            mu, num_rotations=num_rotations
        )
        super().__init__(dim=3 * inferred_num_rotations)
        self.num_rotations = inferred_num_rotations
        self.mu = mean

        C = array(C, dtype=float)
        expected_shape = (self.dim, self.dim)
        if C.shape != expected_shape:
            raise ValueError(f"C must have shape {expected_shape}.")
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
        return 4 * self.num_rotations

    def get_manifold_size(self):
        return pi ** (2 * self.num_rotations)

    _normalize_quaternions = staticmethod(normalize_quaternions)

    @staticmethod
    def _as_product_point(rotations, num_rotations=None):
        rotations = array(rotations, dtype=float)

        if ndim(rotations) == 1:
            if rotations.shape[0] == 0 or rotations.shape[0] % 4 != 0:
                raise ValueError(
                    "Flattened SO(3)^K rotations need 4 entries per component."
                )
            inferred_num_rotations = rotations.shape[0] // 4
            rotations = reshape(rotations, (inferred_num_rotations, 4))
        elif ndim(rotations) == 2:
            if rotations.shape[-1] != 4:
                raise ValueError("SO(3) quaternions must have length 4.")
            if rotations.shape[0] == 0:
                raise ValueError(
                    "SO(3)^K rotations must contain at least one component."
                )
            inferred_num_rotations = rotations.shape[0]
        else:
            raise ValueError("A product point must have shape (K, 4) or (4 * K,).")

        if num_rotations is not None and inferred_num_rotations != num_rotations:
            raise ValueError("num_rotations does not match input shape.")

        return (
            SO3ProductTangentGaussianDistribution._normalize_quaternions(rotations),
            inferred_num_rotations,
        )

    @staticmethod
    def _as_product_batch(rotations, num_rotations=None):
        rotations = array(rotations, dtype=float)

        if ndim(rotations) == 1:
            if rotations.shape[0] == 0 or rotations.shape[0] % 4 != 0:
                raise ValueError(
                    "Flattened SO(3)^K rotations need 4 entries per component."
                )
            inferred_num_rotations = rotations.shape[0] // 4
            rotations = reshape(rotations, (1, inferred_num_rotations, 4))
        elif ndim(rotations) == 2:
            if rotations.shape[-1] == 4 and (
                num_rotations is None or rotations.shape[0] == num_rotations
            ):
                if rotations.shape[0] == 0:
                    raise ValueError(
                        "SO(3)^K rotations must contain at least one component."
                    )
                inferred_num_rotations = rotations.shape[0]
                rotations = reshape(rotations, (1, inferred_num_rotations, 4))
            else:
                if rotations.shape[-1] == 0 or rotations.shape[-1] % 4 != 0:
                    raise ValueError(
                        "Flattened SO(3)^K rotations need 4 entries per component."
                    )
                inferred_num_rotations = rotations.shape[-1] // 4
                rotations = reshape(
                    rotations, (rotations.shape[0], inferred_num_rotations, 4)
                )
        elif ndim(rotations) == 3:
            if rotations.shape[-1] != 4:
                raise ValueError("SO(3) quaternions must have length 4.")
            if rotations.shape[1] == 0:
                raise ValueError(
                    "SO(3)^K rotations must contain at least one component."
                )
            inferred_num_rotations = rotations.shape[1]
        else:
            raise ValueError(
                "SO(3)^K rotations must have shape (K, 4), (4 * K,), "
                "(n, K, 4), or (n, 4 * K)."
            )

        if num_rotations is not None and inferred_num_rotations != num_rotations:
            raise ValueError("num_rotations does not match input shape.")

        return (
            SO3ProductTangentGaussianDistribution._normalize_quaternions(rotations),
            inferred_num_rotations,
        )

    @staticmethod
    def _as_tangent_batch(tangent_vectors, num_rotations=None):
        tangent_vectors = array(tangent_vectors, dtype=float)

        if ndim(tangent_vectors) == 1:
            if tangent_vectors.shape[0] == 0 or tangent_vectors.shape[0] % 3 != 0:
                raise ValueError(
                    "Flattened SO(3)^K tangent vectors need 3 entries per component."
                )
            inferred_num_rotations = tangent_vectors.shape[0] // 3
            tangent_vectors = reshape(tangent_vectors, (1, inferred_num_rotations, 3))
        elif ndim(tangent_vectors) == 2:
            if tangent_vectors.shape[-1] == 3 and (
                num_rotations is None or tangent_vectors.shape[0] == num_rotations
            ):
                if tangent_vectors.shape[0] == 0:
                    raise ValueError(
                        "SO(3)^K tangent vectors must contain at least one component."
                    )
                inferred_num_rotations = tangent_vectors.shape[0]
                tangent_vectors = reshape(
                    tangent_vectors, (1, inferred_num_rotations, 3)
                )
            else:
                if tangent_vectors.shape[-1] == 0 or tangent_vectors.shape[-1] % 3 != 0:
                    raise ValueError(
                        "Flattened SO(3)^K tangent vectors need 3 entries per component."
                    )
                inferred_num_rotations = tangent_vectors.shape[-1] // 3
                tangent_vectors = reshape(
                    tangent_vectors,
                    (tangent_vectors.shape[0], inferred_num_rotations, 3),
                )
        elif ndim(tangent_vectors) == 3:
            if tangent_vectors.shape[-1] != 3:
                raise ValueError("SO(3) tangent vectors must have length 3.")
            if tangent_vectors.shape[1] == 0:
                raise ValueError(
                    "SO(3)^K tangent vectors must contain at least one component."
                )
            inferred_num_rotations = tangent_vectors.shape[1]
        else:
            raise ValueError(
                "SO(3)^K tangent vectors must have shape (K, 3), (3 * K,), "
                "(n, K, 3), or (n, 3 * K)."
            )

        if num_rotations is not None and inferred_num_rotations != num_rotations:
            raise ValueError("num_rotations does not match input shape.")

        return tangent_vectors, inferred_num_rotations

    _quaternion_conjugate = staticmethod(quaternion_conjugate)
    _quaternion_multiply = staticmethod(quaternion_multiply)
    _exp_map_so3_identity = staticmethod(exp_map_identity)
    _log_map_so3_identity = staticmethod(log_map_identity)

    @staticmethod
    def exp_map(tangent_vectors, base=None, num_rotations=None):
        """Map flattened tangent vectors to SO(3)^K product quaternions."""
        tangent_vectors, inferred_num_rotations = (
            SO3ProductTangentGaussianDistribution._as_tangent_batch(
                tangent_vectors, num_rotations=num_rotations
            )
        )

        if base is None:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                stack(
                    [
                        array([0.0, 0.0, 0.0, 1.0])
                        for _ in range(inferred_num_rotations)
                    ],
                    0,
                ),
                num_rotations=inferred_num_rotations,
            )
        else:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                base, num_rotations=inferred_num_rotations
            )

        components = []
        for i in range(inferred_num_rotations):
            delta = SO3ProductTangentGaussianDistribution._exp_map_so3_identity(
                tangent_vectors[:, i, :]
            )
            components.append(
                SO3ProductTangentGaussianDistribution._quaternion_multiply(
                    base[i, :], delta
                )
            )
        return stack(components, 1)

    @staticmethod
    def log_map(rotations, base=None, num_rotations=None):
        """Map SO(3)^K product quaternions to flattened tangent vectors."""
        rotations, inferred_num_rotations = (
            SO3ProductTangentGaussianDistribution._as_product_batch(
                rotations, num_rotations=num_rotations
            )
        )

        if base is None:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                stack(
                    [
                        array([0.0, 0.0, 0.0, 1.0])
                        for _ in range(inferred_num_rotations)
                    ],
                    0,
                ),
                num_rotations=inferred_num_rotations,
            )
        else:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                base, num_rotations=inferred_num_rotations
            )

        tangent_components = []
        for i in range(inferred_num_rotations):
            relative_rotation = (
                SO3ProductTangentGaussianDistribution._quaternion_multiply(
                    SO3ProductTangentGaussianDistribution._quaternion_conjugate(
                        base[i, :]
                    ),
                    rotations[:, i, :],
                )
            )
            tangent_components.append(
                SO3ProductTangentGaussianDistribution._log_map_so3_identity(
                    relative_rotation
                )
            )

        tangent_vectors = stack(tangent_components, 1)
        return reshape(
            tangent_vectors, (tangent_vectors.shape[0], 3 * inferred_num_rotations)
        )

    @staticmethod
    def geodesic_distance(rotation_a, rotation_b, reduce=True, num_rotations=None):
        """Return component-wise or summed SO(3)^K geodesic distances."""
        rotation_a, inferred_num_rotations = (
            SO3ProductTangentGaussianDistribution._as_product_batch(
                rotation_a, num_rotations=num_rotations
            )
        )
        rotation_b, _ = SO3ProductTangentGaussianDistribution._as_product_batch(
            rotation_b, num_rotations=inferred_num_rotations
        )

        distances = so3_geodesic_distance(rotation_a, rotation_b)
        if reduce:
            return sum(distances, axis=-1)
        return distances

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        return quaternions_to_rotation_matrices(quaternions)

    def pdf(self, xs):
        """Evaluate the SO(3)^K product volume density at rotations."""
        return exp(self.ln_pdf(xs))

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the SO(3)^K volume density."""
        residual = self.tangent_vectors(xs)
        precision = linalg.inv(self.C)
        quadratic = sum(residual * matmul(residual, precision), axis=-1)
        log_det = 2.0 * sum(log(diag(linalg.cholesky(self.C))))
        tangent_log_density = -0.5 * (self.dim * log(2.0 * pi) + log_det + quadratic)
        tangent_vectors_product = reshape(
            residual, (residual.shape[0], self.num_rotations, 3)
        )
        log_volume_jacobian = sum(
            so3_exp_map_volume_log_jacobian(tangent_vectors_product), axis=-1
        )
        return tangent_log_density - log_volume_jacobian

    def tangent_vectors(self, rotations):
        """Return flattened log-map coordinates around the distribution mean."""
        return self.log_map(rotations, base=self.mu, num_rotations=self.num_rotations)

    def tangent_vectors_product(self, rotations):
        """Return log-map coordinates with shape ``(n, K, 3)``."""
        tangent_vectors = self.tangent_vectors(rotations)
        return reshape(
            tangent_vectors, (tangent_vectors.shape[0], self.num_rotations, 3)
        )

    def sample_tangent(self, n):
        """Draw tangent-space Gaussian samples with shape ``(n, 3 * K)``."""
        n = _validate_positive_sample_count(n)
        samples = random.multivariate_normal(mean=zeros(self.dim), cov=self.C, size=n)
        if ndim(samples) == 1:
            return reshape(samples, (1, self.dim))
        return samples

    def sample(self, n):
        """Draw ``n`` SO(3)^K samples as scalar-last unit quaternions."""
        return self.exp_map(
            self.sample_tangent(n), base=self.mu, num_rotations=self.num_rotations
        )

    def mean(self):
        """Return the mean product rotation with shape ``(K, 4)``."""
        return self.mu

    def mode(self):
        """Return the modal product rotation with shape ``(K, 4)``."""
        return self.mu

    def mean_rotation_matrices(self):
        """Return rotation matrices of the mean product rotation."""
        return self.as_rotation_matrices(self.mu)

    def covariance(self):
        """Return the full ``(3K, 3K)`` tangent covariance matrix."""
        return self.C

    def set_mean(self, new_mean):
        """Return a copy with a replaced mean product rotation."""
        return self.set_mode(new_mean)

    def set_mode(self, new_mode):
        """Return a copy with a replaced modal product rotation."""
        return self.__class__(new_mode, self.C, check_validity=False)

    def marginalize_rotation(self, rotation_index):
        """Return the one-component SO(3) tangent Gaussian marginal."""
        return self.marginalize_rotations([rotation_index])

    def marginalize_rotations(self, rotation_indices):
        """Return the marginal over selected SO(3) components."""
        if isinstance(rotation_indices, int):
            rotation_indices = [rotation_indices]
        rotation_indices_array = array(rotation_indices)
        tangent_indices = [
            3 * rotation_index + offset
            for rotation_index in rotation_indices
            for offset in range(3)
        ]
        tangent_indices = array(tangent_indices)
        new_covariance = take(
            take(self.C, tangent_indices, axis=0), tangent_indices, axis=1
        )
        new_mean = reshape(
            take(self.mu, rotation_indices_array, axis=0),
            (len(rotation_indices), 4),
        )
        return SO3ProductTangentGaussianDistribution(
            new_mean,
            new_covariance,
            num_rotations=len(rotation_indices),
            check_validity=False,
        )

    def distance_to(self, rotations, reduce=True):
        """Return geodesic distances from the mean to ``rotations``."""
        return self.geodesic_distance(
            self.mu, rotations, reduce=reduce, num_rotations=self.num_rotations
        )

    def is_valid(self, tolerance=1e-6):
        """Return whether the mean and covariance have valid SO(3)^K dimensions."""
        mean_norms = linalg.norm(self.mu, axis=-1)
        covariance_is_symmetric = amax(abs(self.C - transpose(self.C))) <= tolerance
        return bool(
            amax(abs(mean_norms - 1.0)) <= tolerance
            and all(self.mu[:, -1] >= -tolerance)
            and covariance_is_symmetric
        )

    @staticmethod
    def from_covariance_diagonal(mu, covariance_diagonal):
        """Create a product tangent Gaussian from a diagonal covariance vector."""
        return SO3ProductTangentGaussianDistribution(mu, diag(covariance_diagonal))
