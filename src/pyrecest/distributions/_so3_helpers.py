"""Shared helper functions for SO(3) distributions."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    arccos,
    arctan2,
    array,
    clip,
    concatenate,
    cos,
    linalg,
    ndim,
    reshape,
    sin,
    stack,
    sum,
    where,
)


def as_batch(values, width, name):
    """Return values with a fixed trailing width as a two-dimensional batch."""
    values = array(values, dtype=float)
    if ndim(values) == 1:
        assert values.shape[0] == width, f"{name} must have length {width}."
        values = reshape(values, (1, width))
    else:
        assert values.shape[-1] == width, f"{name} must have length {width}."
        values = reshape(values, (-1, width))
    return values


def normalize_quaternions(quaternions):
    """Return canonical scalar-last unit quaternions."""
    quaternions = array(quaternions, dtype=float)
    if ndim(quaternions) == 1:
        assert quaternions.shape[0] == 4, "SO(3) quaternions must have length 4."
        quaternions = reshape(quaternions, (1, 4))
    else:
        assert quaternions.shape[-1] == 4, "SO(3) quaternions must have length 4."

    norms = linalg.norm(quaternions, axis=-1)
    assert all(norms > 0.0), "SO(3) quaternions must be nonzero."

    normalized = quaternions / reshape(norms, tuple(norms.shape) + (1,))
    return where(normalized[..., -1:] < 0.0, -normalized, normalized)


def quaternion_conjugate(quaternions):
    """Return conjugates of scalar-last unit quaternions."""
    return normalize_quaternions(quaternions) * array([-1.0, -1.0, -1.0, 1.0])


def quaternion_multiply(left, right):
    """Return Hamilton products of scalar-last unit quaternions."""
    left = normalize_quaternions(left)
    right = normalize_quaternions(right)

    x1, y1, z1, w1 = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
    x2, y2, z2, w2 = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
    product = stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )
    return normalize_quaternions(product)


def exp_map_identity(tangent_vectors):
    """Map SO(3) tangent vectors at identity to scalar-last quaternions."""
    tangent_vectors = array(tangent_vectors, dtype=float)
    if ndim(tangent_vectors) == 1:
        assert (
            tangent_vectors.shape[0] == 3
        ), "SO(3) tangent vectors must have length 3."
        tangent_vectors = reshape(tangent_vectors, (1, 3))
    else:
        assert (
            tangent_vectors.shape[-1] == 3
        ), "SO(3) tangent vectors must have length 3."

    angles = linalg.norm(tangent_vectors, axis=-1)
    angles_col = reshape(angles, tuple(angles.shape) + (1,))
    safe_angles = where(angles_col > 1e-12, angles_col, 1.0)
    vector_scale = where(
        angles_col > 1e-12,
        sin(0.5 * angles_col) / safe_angles,
        0.5 - angles_col**2 / 48.0,
    )
    return normalize_quaternions(
        concatenate((tangent_vectors * vector_scale, cos(0.5 * angles_col)), axis=-1)
    )


def log_map_identity(rotations):
    """Map scalar-last SO(3) quaternions to tangent vectors at identity."""
    rotations = normalize_quaternions(rotations)
    vector_part = rotations[..., :3]
    scalar_part = clip(rotations[..., 3], -1.0, 1.0)
    vector_norm = linalg.norm(vector_part, axis=-1)
    angles = 2.0 * arctan2(vector_norm, scalar_part)
    vector_norm_col = reshape(vector_norm, tuple(vector_norm.shape) + (1,))
    safe_norm = where(vector_norm_col > 1e-12, vector_norm_col, 1.0)
    scale = where(
        vector_norm_col > 1e-12,
        reshape(angles, tuple(angles.shape) + (1,)) / safe_norm,
        2.0,
    )
    return vector_part * scale


def geodesic_distance(rotation_a, rotation_b):
    """Return the SO(3) geodesic distance between quaternions in radians."""
    scalar_input = ndim(array(rotation_a, dtype=float)) == 1 and ndim(array(rotation_b, dtype=float)) == 1
    quat_a = normalize_quaternions(rotation_a)
    quat_b = normalize_quaternions(rotation_b)
    inner = abs(sum(quat_a * quat_b, axis=-1))
    distances = 2.0 * arccos(clip(inner, 0.0, 1.0))
    return distances[0] if scalar_input else distances


def quaternions_to_rotation_matrices(quaternions):
    """Convert scalar-last quaternions to rotation matrices."""
    quaternions = normalize_quaternions(quaternions)
    x, y, z, w = (
        quaternions[..., 0],
        quaternions[..., 1],
        quaternions[..., 2],
        quaternions[..., 3],
    )
    row_0 = stack(
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        axis=-1,
    )
    row_1 = stack(
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        axis=-1,
    )
    row_2 = stack(
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        axis=-1,
    )
    return stack((row_0, row_1, row_2), axis=-2)
