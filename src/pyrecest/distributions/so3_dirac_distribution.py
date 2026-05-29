"""Dirac distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from numbers import Integral

from pyrecest.backend import (
    abs,
    all,
    amax,
    array,
    asarray,
    clip,
    isfinite,
    linalg,
    ndim,
    spatial,
    sqrt,
    sum,
)

from ._so3_helpers import geodesic_distance, normalize_quaternions
from .hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)


class SO3DiracDistribution(HyperhemisphericalDiracDistribution):
    """Weighted Dirac distribution on SO(3).

    The distribution stores rotations as unit quaternions with scalar-last
    coordinates ``(x, y, z, w)``. Since ``q`` and ``-q`` represent the same
    rotation, quaternions are canonicalized to a nonnegative scalar component
    during construction.
    """

    def __init__(self, d, w=None):
        quaternions = self._normalize_quaternions(d)
        super().__init__(quaternions, w=w)

    _normalize_quaternions = staticmethod(normalize_quaternions)

    @staticmethod
    def _require_rotation_method(method_name):
        if not hasattr(spatial.Rotation, method_name):
            raise NotImplementedError(
                f"Rotation.{method_name} is not supported by the active backend."
            )

    @classmethod
    def from_rotation_matrices(cls, rotation_matrices, w=None):
        """Create an SO(3) Dirac distribution from rotation matrices."""
        cls._require_rotation_method("from_matrix")
        rotation_matrices = asarray(rotation_matrices)
        if ndim(rotation_matrices) == 2:
            if rotation_matrices.shape != (3, 3):
                raise ValueError("A single rotation matrix must have shape (3, 3).")
        elif ndim(rotation_matrices) >= 3:
            if rotation_matrices.shape[-2:] != (3, 3):
                raise ValueError("Rotation matrices must have shape (..., 3, 3).")
        else:
            raise ValueError("Rotation matrices must have shape (3, 3) or (..., 3, 3).")
        if not bool(all(isfinite(rotation_matrices))):
            raise ValueError("Rotation matrices must be finite.")

        quaternions = spatial.Rotation.from_matrix(rotation_matrices).as_quat()
        return cls(quaternions, w=w)

    def as_quaternions(self):
        """Return canonical scalar-last unit quaternions shaped ``(n, 4)``."""
        return self.d

    def as_rotation_matrices(self):
        """Return Dirac locations as rotation matrices shaped ``(n, 3, 3)``."""
        self._require_rotation_method("from_quat")
        return array(spatial.Rotation.from_quat(self.d).as_matrix())

    def mean_rotation_matrix(self):
        """Return the mean rotation as a 3-by-3 rotation matrix."""
        self._require_rotation_method("from_quat")
        return array(spatial.Rotation.from_quat(self.mean()).as_matrix())

    geodesic_distance = staticmethod(geodesic_distance)

    @staticmethod
    def chordal_distance(rotation_a, rotation_b):
        """Return the SO(3) Frobenius chordal distance between quaternions."""
        quat_a = SO3DiracDistribution._normalize_quaternions(rotation_a)
        quat_b = SO3DiracDistribution._normalize_quaternions(rotation_b)
        inner = abs(sum(quat_a * quat_b, axis=-1))
        return 2.0 * sqrt(2.0) * sqrt(clip(1.0 - inner**2, 0.0, 1.0))

    def distance_to(self, rotation):
        """Return geodesic distances from all Dirac locations to ``rotation``."""
        return self.geodesic_distance(self.d, rotation)

    def chordal_distance_to(self, rotation):
        """Return chordal distances from all Dirac locations to ``rotation``."""
        return self.chordal_distance(self.d, rotation)

    @classmethod
    def from_distribution(cls, distribution, n_particles=None):
        """Create an SO(3) Dirac distribution by sampling another distribution."""
        n_particles = cls._validate_particle_count(n_particles)
        return cls(distribution.sample(n_particles))

    @staticmethod
    def _validate_particle_count(n_particles):
        if (
            isinstance(n_particles, bool)
            or not isinstance(n_particles, Integral)
            or int(n_particles) <= 0
        ):
            raise ValueError("n_particles must be a positive integer")
        return int(n_particles)

    def angular_error_mean(self, rotation):
        """Return the weighted mean angular error to ``rotation`` in radians."""
        return sum(self.w * self.distance_to(rotation))

    def chordal_error_mean(self, rotation):
        """Return the weighted mean chordal error to ``rotation``."""
        return sum(self.w * self.chordal_distance_to(rotation))

    def is_valid(self, tolerance=1e-6):
        """Return whether all stored quaternions are normalized and canonical."""
        norms = linalg.norm(self.d, None, -1)
        return bool(
            amax(abs(norms - 1.0)) <= tolerance and all(self.d[:, -1] >= -tolerance)
        )
