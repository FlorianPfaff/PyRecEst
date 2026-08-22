"""Bingham distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from math import isfinite as scalar_isfinite

from pyrecest.backend import (
    abs,
    all,
    amax,
    argsort,
    array,
    column_stack,
    diag,
    dot,
    eye,
    isfinite,
    linalg,
    log,
    ndim,
    one_hot,
    sum,
    transpose,
    zeros,
)

from ._so3_helpers import (
    geodesic_distance,
    normalize_quaternions,
    quaternions_to_rotation_matrices,
)
from .hypersphere_subset.bingham_distribution import BinghamDistribution
from .hypersphere_subset.hyperhemispherical_bingham_distribution import (
    HyperhemisphericalBinghamDistribution,
)


class SO3BinghamDistribution(HyperhemisphericalBinghamDistribution):
    """Bingham distribution over rotations represented by unit quaternions.

    The distribution uses scalar-last quaternion coordinates ``(x, y, z, w)``.
    Because ``q`` and ``-q`` encode the same rotation, this class models SO(3)
    through the upper quaternion hemisphere and evaluates twice the full
    antipodally symmetric S^3 Bingham density.
    """

    def __init__(self, Z, M):
        Z = array(Z, dtype=float)
        M = array(M, dtype=float)
        self._validate_parameters(Z, M)

        super().__init__(Z, M)
        self.distFullSphere.F = array(self.calculate_normalization_constant(Z))

    @staticmethod
    def _validate_parameters(Z, M):
        if ndim(Z) != 1 or Z.shape[0] != 4:
            raise ValueError("Z must have shape (4,).")
        if M.shape != (4, 4):
            raise ValueError("M must have shape (4, 4).")
        if not bool(all(isfinite(Z))):
            raise ValueError("Z must be finite.")
        if not bool(all(isfinite(M))):
            raise ValueError("M must be finite.")
        if not bool(abs(Z[-1]) <= 1e-12):
            raise ValueError("The last entry of Z must be zero.")
        if not bool(all(Z[:-1] <= Z[1:])):
            raise ValueError("Values in Z must be ascending.")
        if not bool(amax(abs(M @ transpose(M) - eye(4))) < 0.001):
            raise ValueError("M must be orthogonal.")

    @staticmethod
    def calculate_normalization_constant(Z):
        """Return the 4-D Bingham normalizing constant."""
        Z = array(Z, dtype=float)
        if Z.shape != (4,):
            raise ValueError("Z must have shape (4,).")
        if not bool(all(isfinite(Z))):
            raise ValueError("Z must be finite.")
        return BinghamDistribution.calculate_F(Z)

    @staticmethod
    def _normalize_quaternions(quaternions):
        return normalize_quaternions(quaternions)

    @staticmethod
    def _quaternion_right_multiply_matrix(quaternion):
        """Return the scalar-last matrix mapping ``p`` to ``p * quaternion``."""
        quaternion = SO3BinghamDistribution._normalize_quaternions(quaternion)[0]
        x, y, z, w = quaternion
        return array(
            [
                [w, z, -y, x],
                [-z, w, x, y],
                [y, -x, w, z],
                [-x, -y, -z, w],
            ]
        )

    @staticmethod
    def _canonicalize_quaternion(quaternion):
        return SO3BinghamDistribution._normalize_quaternions(quaternion)[0]

    @staticmethod
    def _orthogonal_completion(mode):
        mode = SO3BinghamDistribution._canonicalize_quaternion(mode)
        basis: list = []
        for column in eye(4):
            vector = column - dot(column, mode) * mode
            for existing in basis:
                vector = vector - dot(vector, existing) * existing

            norm = linalg.norm(vector)
            if norm > 1e-10:
                basis.append(vector / norm)
            if len(basis) == 3:
                break

        return column_stack((*basis, mode))

    @classmethod
    def from_mode_and_concentration(cls, mode, concentration):
        """Create an isotropic Bingham distribution around ``mode``."""
        concentration = float(concentration)
        if not scalar_isfinite(concentration) or concentration < 0.0:
            raise ValueError("concentration must be finite and nonnegative.")
        Z = array([-concentration, -concentration, -concentration, 0.0])
        M = cls._orthogonal_completion(mode)
        return cls(Z, M)

    @classmethod
    def from_concentration_matrix(cls, concentration_matrix):
        """Create from a symmetric 4-by-4 exponent matrix."""
        concentration_matrix = array(concentration_matrix, dtype=float)
        if concentration_matrix.shape != (4, 4):
            raise ValueError("concentration_matrix must have shape (4, 4).")
        if not bool(all(isfinite(concentration_matrix))):
            raise ValueError("concentration_matrix must be finite.")
        concentration_matrix = 0.5 * (
            concentration_matrix + transpose(concentration_matrix)
        )

        eigenvalues, eigenvectors = linalg.eigh(concentration_matrix)
        order = argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        Z = eigenvalues - eigenvalues[-1]
        return cls(array(Z), array(eigenvectors))

    @classmethod
    def from_bingham_distribution(cls, distribution):
        """Create from a 4-D hyperspherical Bingham distribution."""
        if not isinstance(distribution, BinghamDistribution):
            raise ValueError("distribution must be a BinghamDistribution.")
        if distribution.input_dim != 4:
            raise ValueError("distribution must be 4-D.")
        return cls(distribution.Z, distribution.M)

    def pdf(self, xs):
        """Evaluate the SO(3) density at scalar-last quaternions."""
        return super().pdf(self._normalize_quaternions(xs))

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the SO(3) density."""
        xs = self._normalize_quaternions(xs)
        exponent_matrix = self.concentration_matrix()
        return (
            log(array(2.0))
            - log(self.distFullSphere.F)
            + sum(xs * (xs @ exponent_matrix), axis=-1)
        )

    def mode(self):
        """Return the modal rotation as a canonical scalar-last quaternion."""
        return self._canonicalize_quaternion(self.distFullSphere.mode())

    def mean(self):
        """Return the principal quaternion axis as the SO(3) mean proxy."""
        return self.mean_axis()

    def mean_axis(self):
        """Return the principal axis as a canonical scalar-last quaternion."""
        return self.mode()

    def concentration_matrix(self):
        """Return the symmetric 4-by-4 exponent matrix."""
        return (
            self.distFullSphere.M
            @ diag(self.distFullSphere.Z)
            @ self.distFullSphere.M.T
        )

    def moment_weights(self):
        """Return normalized Bingham moment weights without backend mutation."""
        Z = array(self.distFullSphere.Z, dtype=float)
        normalizer = self.calculate_normalization_constant(Z)
        epsilon = 0.001
        derivatives = []
        for idx in range(4):
            delta = epsilon * one_hot(idx, 4)
            derivatives.append(
                (
                    self.calculate_normalization_constant(Z + delta)
                    - self.calculate_normalization_constant(Z - delta)
                )
                / (2.0 * epsilon)
            )

        weights = array(derivatives, dtype=float) / normalizer
        weights = weights / sum(weights)
        return array(weights)

    def moment(self):
        """Return the quaternion scatter matrix."""
        weights = self.moment_weights()
        moment_matrix = (
            self.distFullSphere.M @ diag(weights) @ transpose(self.distFullSphere.M)
        )
        return 0.5 * (moment_matrix + transpose(moment_matrix))

    def multiply(self, B2):
        """Multiply two SO(3) Bingham densities."""
        if not isinstance(B2, SO3BinghamDistribution):
            raise ValueError("B2 must be an SO3BinghamDistribution.")
        product = self.distFullSphere.multiply(B2.distFullSphere)
        return type(self)(product.Z, product.M)

    def compose(self, B2):
        """Approximate the distribution of the composed rotation ``self * other``."""
        if not isinstance(B2, SO3BinghamDistribution):
            raise ValueError("B2 must be an SO3BinghamDistribution.")

        weights = B2.moment_weights()
        first_moment = self.moment()

        composed_moment = zeros((4, 4))
        for idx in range(4):
            right_matrix = self._quaternion_right_multiply_matrix(
                B2.distFullSphere.M[:, idx]
            )
            composed_moment = composed_moment + weights[
                idx
            ] * right_matrix @ first_moment @ transpose(right_matrix)

        composed_moment = 0.5 * (composed_moment + transpose(composed_moment))
        return type(self).from_bingham_distribution(
            BinghamDistribution.fit_to_moment(composed_moment)
        )

    def sample(self, n):
        """Draw ``n`` canonical scalar-last unit quaternion samples."""
        return self._normalize_quaternions(super().sample(n))

    geodesic_distance = staticmethod(geodesic_distance)

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        return quaternions_to_rotation_matrices(quaternions)

    def mean_rotation_matrix(self):
        """Return the modal/principal rotation matrix."""
        return self.as_rotation_matrices(self.mean())[0]

    def is_valid(self, tolerance=1e-6):
        """Return whether the parameters define a valid SO(3) Bingham model."""
        Z = self.distFullSphere.Z
        M = self.distFullSphere.M
        orthogonal = amax(abs(M @ transpose(M) - eye(4))) <= tolerance
        sorted_z = all(Z[:-1] <= Z[1:] + tolerance)
        zero_last = abs(Z[-1]) <= tolerance
        symmetric = (
            amax(
                abs(
                    self.concentration_matrix() - transpose(self.concentration_matrix())
                )
            )
            <= tolerance
        )
        return bool(orthogonal and sorted_z and zero_last and symmetric)
