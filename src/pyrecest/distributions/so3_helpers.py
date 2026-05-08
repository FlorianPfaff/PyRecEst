"""Public SO(3) geometry helpers.

Rotations are represented as scalar-last unit quaternions ``(x, y, z, w)``.
Quaternion-returning helpers canonicalize the antipodal representative by
normalizing the quaternion and choosing the representative with nonnegative
scalar component.
"""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import sum

from ._so3_helpers import (
    as_batch,
    exp_map_identity,
    geodesic_distance,
    log_map_identity,
    normalize_quaternions,
    quaternion_conjugate,
    quaternion_multiply,
    quaternions_to_rotation_matrices,
)

__all__ = [
    "as_batch",
    "canonicalize_quaternions",
    "exp_map",
    "exp_map_identity",
    "geodesic_distance",
    "log_map",
    "log_map_identity",
    "normalize_quaternions",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternions_to_rotation_matrices",
    "rotate_vectors",
    "rotate_vectors_by_quaternions",
    "rotation_matrices_from_quaternions",
]


def canonicalize_quaternions(quaternions):
    """Return canonical scalar-last unit quaternions.

    This is an explicit public alias for :func:`normalize_quaternions`. It
    normalizes the input and chooses the antipodal representative whose scalar
    component is nonnegative.
    """
    return normalize_quaternions(quaternions)


def exp_map(tangent_vectors, base=None):
    """Map SO(3) tangent vectors to scalar-last unit quaternions.

    If ``base`` is given, the returned rotations are ``base * Exp(v)``.
    """
    delta_quaternions = exp_map_identity(
        as_batch(tangent_vectors, 3, "SO(3) tangent vectors")
    )

    if base is None:
        return delta_quaternions
    return quaternion_multiply(base, delta_quaternions)


def log_map(rotations, base=None):
    """Map scalar-last unit quaternions to SO(3) tangent vectors.

    If ``base`` is given, this returns ``Log(base^{-1} * rotations)``.
    """
    rotations = normalize_quaternions(rotations)
    if base is not None:
        rotations = quaternion_multiply(quaternion_conjugate(base), rotations)

    return log_map_identity(rotations)


def rotation_matrices_from_quaternions(quaternions):
    """Convert scalar-last quaternions to rotation matrices."""
    return quaternions_to_rotation_matrices(quaternions)


def rotate_vectors(rotations, vectors):
    """Rotate 3-D vectors by scalar-last SO(3) quaternions.

    ``rotations`` and ``vectors`` may either have matching batch sizes or one of
    them may contain a single entry, in which case backend broadcasting applies.
    The returned array has shape ``(n, 3)``.
    """
    rotations = as_batch(rotations, 4, "SO(3) quaternions")
    vectors = as_batch(vectors, 3, "Euclidean vectors")
    rotation_matrices = quaternions_to_rotation_matrices(rotations)
    return sum(rotation_matrices * vectors[..., None, :], axis=-1)


def rotate_vectors_by_quaternions(rotations, vectors):
    """Rotate 3-D vectors by scalar-last SO(3) quaternions."""
    return rotate_vectors(rotations, vectors)
