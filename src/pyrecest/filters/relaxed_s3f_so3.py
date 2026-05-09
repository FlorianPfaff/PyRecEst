"""Relaxed S3F prediction helpers for S3+ x R3.

The helpers in this module mirror :mod:`pyrecest.filters.relaxed_s3f_circular`
for quaternion-grid orientation marginals coupled to a three-dimensional
linear conditional state.

The current SO(3) implementation uses deterministic samples in each grid
point's local tangent neighbourhood to approximate the unresolved cell
statistics. It is therefore a numerical local-cell approximation, not an exact
S3+ Voronoi-cell integration rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, pi, sqrt
from typing import Any

from pyrecest.backend import abs as backend_abs
from pyrecest.backend import (
    arccos,
    array,
    asarray,
    clip,
    einsum,
    linspace,
    mean,
    stack,
    transpose,
    zeros,
    zeros_like,
)
from pyrecest.distributions._so3_helpers import (
    exp_map_identity,
    geodesic_distance,
    normalize_quaternions,
    quaternion_multiply,
    quaternions_to_rotation_matrices,
)

from .state_space_subdivision_filter import StateSpaceSubdivisionFilter

SUPPORTED_RELAXED_S3F_SO3_VARIANTS = ("baseline", "r1", "r1_r2")
SUPPORTED_S3R3_CELL_METHODS = ("local_tangent_samples",)


@dataclass(frozen=True)
class S3R3CellStatistics:
    """Numerical R1/R2 statistics for local tangent cells around S3+ points.

    Attributes
    ----------
    grid
        Canonical scalar-last unit quaternions, one per grid cell.
    cell_radius_rad
        Radius used for the deterministic local tangent sample cloud.
    body_increment
        Body-frame R3 displacement used in the statistics.
    representative_displacements
        Baseline displacement obtained by rotating ``body_increment`` with the
        representative grid quaternion.
    mean_displacements
        R1 mean displacement estimated over local tangent samples.
    covariance_inflations
        R2 unresolved within-cell displacement covariance, one 3x3 matrix per
        grid cell.
    cell_sample_count
        Number of deterministic tangent samples per cell.
    method
        Cell-statistics approximation method. Currently only
        ``"local_tangent_samples"`` is implemented.
    """

    grid: Any
    cell_radius_rad: float
    body_increment: Any
    representative_displacements: Any
    mean_displacements: Any
    covariance_inflations: Any
    cell_sample_count: int
    method: str = "local_tangent_samples"


def rotate_quaternion_body_increment(quaternions: Any, body_increment: Any) -> Any:
    """Rotate a body-frame R3 increment by scalar-last SO(3) quaternions.

    Parameters
    ----------
    quaternions
        Scalar-last quaternions with shape ``(4,)`` or ``(n, 4)``.
    body_increment
        Vector ``(u_x, u_y, u_z)`` in the body frame.

    Returns
    -------
    array_like
        World-frame increments with shape ``(n, 3)``.
    """

    matrices = quaternions_to_rotation_matrices(quaternions)
    return einsum("nij,j->ni", matrices, _as_body_increment(body_increment))


def s3r3_cell_statistics(
    grid: Any,
    body_increment: Any,
    cell_sample_count: int = 27,
    *,
    method: str = "local_tangent_samples",
) -> S3R3CellStatistics:
    """Compute R1/R2 displacement statistics for an S3+ x R3 S3F grid.

    The implemented ``method="local_tangent_samples"`` estimates each cell's
    mean displacement and covariance inflation from deterministic tangent-space
    samples around the cell representative. This is an approximation to the
    unresolved cell statistics, not exact integration over the grid cell.
    """

    if method not in SUPPORTED_S3R3_CELL_METHODS:
        raise ValueError(
            f"Unknown S3R3 cell-statistics method {method!r}; expected one of "
            f"{SUPPORTED_S3R3_CELL_METHODS}."
        )
    if cell_sample_count <= 0:
        raise ValueError("cell_sample_count must be positive.")

    grid_array = _as_quaternion_grid(grid)
    increment = _as_body_increment(body_increment)
    return _cached_s3r3_cell_statistics(
        tuple(int(value) for value in grid_array.shape),
        _array_to_float_tuple(grid_array),
        _array_to_float_tuple(increment),
        int(cell_sample_count),
        method,
    )


@lru_cache(maxsize=128)
def _cached_s3r3_cell_statistics(
    grid_shape: tuple[int, ...],
    grid_values: tuple[float, ...],
    body_increment_values: tuple[float, ...],
    cell_sample_count: int,
    method: str,
) -> S3R3CellStatistics:
    grid = array(grid_values, dtype=float).reshape(grid_shape)
    body_increment = array(body_increment_values, dtype=float).reshape(3)
    stats = _compute_s3r3_cell_statistics(
        grid,
        body_increment,
        cell_sample_count,
        method,
    )
    return _freeze_cell_statistics(stats)


def predict_s3r3_relaxed(
    filter_: StateSpaceSubdivisionFilter,
    body_increment: Any,
    variant: str = "r1_r2",
    process_noise_cov: Any | None = None,
    cell_sample_count: int = 27,
    *,
    method: str = "local_tangent_samples",
) -> S3R3CellStatistics:
    """Predict an S3+ x R3 S3F with the selected relaxed displacement variant.

    Parameters
    ----------
    filter_
        State-space subdivision filter whose grid component is an S3+
        quaternion grid and whose linear conditional is 3-D.
    body_increment
        Body-frame translation increment applied during this prediction.
    variant
        One of ``"baseline"``, ``"r1"``, or ``"r1_r2"``.
    process_noise_cov
        Optional additive 3x3 process noise shared by all grid cells.
    cell_sample_count
        Number of deterministic local tangent samples used by the implemented
        cell-statistics approximation.
    method
        Cell-statistics approximation method. Currently only
        ``"local_tangent_samples"`` is implemented.
    """

    if variant not in SUPPORTED_RELAXED_S3F_SO3_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of "
            f"{SUPPORTED_RELAXED_S3F_SO3_VARIANTS}."
        )

    state = filter_.filter_state
    n_cells = len(state.linear_distributions)
    if state.lin_dim != 3:
        raise ValueError("predict_s3r3_relaxed requires a 3-D linear state.")

    grid = _as_quaternion_grid(state.gd.get_grid())
    if grid.shape[0] != n_cells:
        raise ValueError("grid size must match the number of linear distributions.")

    stats = s3r3_cell_statistics(
        grid,
        body_increment,
        cell_sample_count=cell_sample_count,
        method=method,
    )
    if variant == "baseline":
        displacements = stats.representative_displacements
        covariance_inflations = zeros_like(stats.covariance_inflations)
    elif variant == "r1":
        displacements = stats.mean_displacements
        covariance_inflations = zeros_like(stats.covariance_inflations)
    else:
        displacements = stats.mean_displacements
        covariance_inflations = stats.covariance_inflations

    q_base = (
        zeros((3, 3), dtype=float)
        if process_noise_cov is None
        else asarray(process_noise_cov, dtype=float)
    )
    if q_base.shape != (3, 3):
        raise ValueError("process_noise_cov must have shape (3, 3).")

    covariance_matrices = stack(
        [q_base + covariance_inflations[idx] for idx in range(n_cells)],
        axis=2,
    )
    filter_.predict_linear(
        covariance_matrices=asarray(covariance_matrices),
        linear_input_vectors=asarray(transpose(displacements)),
    )
    return stats


def s3r3_orientation_distance(rotation_a: Any, rotation_b: Any) -> float:
    """Return the antipodal-invariant SO(3) geodesic distance in radians."""

    return float(geodesic_distance(rotation_a, rotation_b))


def _compute_s3r3_cell_statistics(
    grid: Any,
    body_increment: Any,
    cell_sample_count: int,
    method: str,
) -> S3R3CellStatistics:
    if method != "local_tangent_samples":
        raise ValueError(f"Unknown S3R3 cell-statistics method {method!r}.")

    grid = _as_quaternion_grid(grid)
    body_increment = _as_body_increment(body_increment)
    cell_radius = _estimate_cell_radius(grid)
    tangent_offsets = _tangent_cell_offsets(cell_radius, cell_sample_count)
    local_quaternions = exp_map_identity(tangent_offsets)

    representative_displacements = rotate_quaternion_body_increment(
        grid, body_increment
    )
    mean_displacements = []
    covariance_inflations = []
    for idx in range(grid.shape[0]):
        center = grid[idx]
        center_batch = center.reshape(1, 4) + zeros(
            (local_quaternions.shape[0], 4), dtype=float
        )
        sample_quaternions = quaternion_multiply(center_batch, local_quaternions)
        displacements = rotate_quaternion_body_increment(
            sample_quaternions,
            body_increment,
        )
        mean_displacement = mean(displacements, axis=0)
        centered = displacements - mean_displacement
        covariance = transpose(centered) @ centered / displacements.shape[0]
        mean_displacements.append(mean_displacement)
        covariance_inflations.append(_symmetrize(covariance))

    return S3R3CellStatistics(
        grid=grid,
        cell_radius_rad=cell_radius,
        body_increment=body_increment,
        representative_displacements=representative_displacements,
        mean_displacements=stack(mean_displacements),
        covariance_inflations=stack(covariance_inflations),
        cell_sample_count=cell_sample_count,
        method=method,
    )


def _estimate_cell_radius(grid: Any) -> float:
    """Estimate a local tangent-cell radius from median nearest-neighbour spacing."""

    grid = _as_quaternion_grid(grid)
    if grid.shape[0] <= 1:
        return float(pi)

    inner = clip(backend_abs(grid @ transpose(grid)), 0.0, 1.0)
    distances = 2.0 * arccos(inner)
    nearest_distances = []
    for i in range(grid.shape[0]):
        row_distances = [float(distances[i, j]) for j in range(grid.shape[0]) if i != j]
        nearest_distances.append(min(row_distances))
    return float(max(0.5 * _median(nearest_distances), 1e-6))


def _tangent_cell_offsets(cell_radius: float, sample_count: int) -> Any:
    """Return deterministic tangent offsets ordered from the cell centre outward."""

    levels = int(ceil(sample_count ** (1.0 / 3.0)))
    if levels % 2 == 0:
        levels += 1
    while levels**3 < sample_count:
        levels += 2

    axis_values = (
        [0.0]
        if levels == 1
        else [float(value) for value in linspace(-cell_radius, cell_radius, levels)]
    )
    offsets = [(x, y, z) for x in axis_values for y in axis_values for z in axis_values]
    offsets.sort(
        key=lambda value: (
            sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2),
            value[0],
            value[1],
            value[2],
        )
    )
    return array(offsets[:sample_count], dtype=float)


def _as_quaternion_grid(grid: Any) -> Any:
    grid = normalize_quaternions(asarray(grid, dtype=float))
    if len(grid.shape) != 2 or grid.shape[1] != 4:
        raise ValueError("grid must have shape (n_cells, 4).")
    return grid


def _as_body_increment(body_increment: Any) -> Any:
    increment = asarray(body_increment, dtype=float).reshape(-1)
    if increment.shape != (3,):
        raise ValueError("body_increment must be a 3-D vector.")
    return increment


def _array_to_float_tuple(values: Any) -> tuple[float, ...]:
    flattened = asarray(values, dtype=float).reshape(-1)
    return tuple(float(value) for value in flattened)


def _freeze_cell_statistics(stats: S3R3CellStatistics) -> S3R3CellStatistics:
    for value in (
        stats.grid,
        stats.body_increment,
        stats.representative_displacements,
        stats.mean_displacements,
        stats.covariance_inflations,
    ):
        setflags = getattr(value, "setflags", None)
        if setflags is not None:
            setflags(write=False)
    return stats


def _symmetrize(matrix: Any) -> Any:
    return 0.5 * (matrix + transpose(matrix))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])
