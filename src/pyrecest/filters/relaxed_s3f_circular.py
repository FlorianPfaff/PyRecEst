"""Relaxed S3F prediction helpers for S1 x R2.

The helpers in this module implement the WP1 pilot relaxations for uniform
circular cells:

* R1: replace representative-orientation displacement by a cell average.
* R2: add the unresolved within-cell displacement covariance.

They intentionally call the existing :class:`StateSpaceSubdivisionFilter`
``predict_linear`` method, so the baseline prediction implementation remains
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyrecest.backend import abs as backend_abs
from pyrecest.backend import angle as backend_angle
from pyrecest.backend import (
    array,
    asarray,
    column_stack,
    cos,
    empty,
    exp,
    linspace,
    mod,
    outer,
    pi,
    sin,
    stack,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    zeros,
    zeros_like,
)

from .state_space_subdivision_filter import StateSpaceSubdivisionFilter

SUPPORTED_RELAXED_S3F_VARIANTS = ("baseline", "r1", "r1_r2")


@dataclass(frozen=True)
class CircularCellStatistics:
    """Closed-form statistics for uniform circular cells."""

    grid: Any
    cell_width: float
    body_increment: Any
    representative_displacements: Any
    mean_displacements: Any
    covariance_inflations: Any


def rotation_matrix(angle: float) -> Any:
    """Return the 2-D rotation matrix for ``angle``."""

    c = cos(angle)
    s = sin(angle)
    return array([[c, -s], [s, c]], dtype=float)


def rotate_body_increment(angles: Any, body_increment: Any) -> Any:
    """Rotate a body-frame increment for one or more angles.

    Parameters
    ----------
    angles
        Array of angles in radians.
    body_increment
        Vector ``(u_x, u_y)`` in the body frame.
    """

    angles = asarray(angles, dtype=float).reshape(-1)
    u = _as_body_increment(body_increment)
    c = cos(angles)
    s = sin(angles)
    return column_stack((c * u[0] - s * u[1], s * u[0] + c * u[1]))


def uniform_circular_cell_statistics(  # pylint: disable=too-many-locals
    n_cells: int,
    body_increment: Any,
    grid: Any | None = None,
) -> CircularCellStatistics:
    """Compute R1/R2 statistics for equal-width cells on S1.

    Each cell is centered at the supplied grid point, with width ``2*pi/n``.
    The within-cell orientation is treated as uniform. This matches the
    piecewise-constant marginal used by the S3F pilot.
    """

    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")

    u = _as_body_increment(body_increment)
    centers = (
        linspace(0.0, 2.0 * pi, n_cells, endpoint=False)
        if grid is None
        else asarray(grid, dtype=float).reshape(-1)
    )
    if centers.shape[0] != n_cells:
        raise ValueError("grid must contain exactly n_cells entries.")

    width = 2.0 * pi / n_cells
    half_width = 0.5 * width
    first_factor = _safe_sinc(half_width)
    second_factor = _safe_sinc(2.0 * half_width)

    representative_displacements = rotate_body_increment(centers, u)
    mean_displacements = first_factor * representative_displacements

    covariance_inflation_list = []
    for idx, center in enumerate(centers):
        cos_2 = cos(2.0 * center)
        sin_2 = sin(2.0 * center)

        e_cos2 = 0.5 * (1.0 + second_factor * cos_2)
        e_sin2 = 0.5 * (1.0 - second_factor * cos_2)
        e_cossin = 0.5 * second_factor * sin_2

        ux, uy = u
        second_moment = array(
            [
                [
                    ux * ux * e_cos2 - 2.0 * ux * uy * e_cossin + uy * uy * e_sin2,
                    (ux * ux - uy * uy) * e_cossin + ux * uy * second_factor * cos_2,
                ],
                [
                    (ux * ux - uy * uy) * e_cossin + ux * uy * second_factor * cos_2,
                    ux * ux * e_sin2 + 2.0 * ux * uy * e_cossin + uy * uy * e_cos2,
                ],
            ],
            dtype=float,
        )
        mean = mean_displacements[idx]
        cov = second_moment - outer(mean, mean)
        covariance_inflation_list.append(0.5 * (cov + cov.T))

    covariance_inflations = (
        stack(covariance_inflation_list, axis=0)
        if covariance_inflation_list
        else empty((0, 2, 2), dtype=float)
    )

    return CircularCellStatistics(
        grid=centers,
        cell_width=width,
        body_increment=u,
        representative_displacements=representative_displacements,
        mean_displacements=mean_displacements,
        covariance_inflations=covariance_inflations,
    )


def predict_circular_relaxed(
    filter_: StateSpaceSubdivisionFilter,
    body_increment: Any,
    variant: str = "r1_r2",
    process_noise_cov: Any | None = None,
) -> CircularCellStatistics:
    """Predict an ``S1 x R2`` S3F with the selected relaxed variant.

    Parameters
    ----------
    filter_
        State-space subdivision filter whose grid component is an equal-width
        circular grid and whose linear conditional is 2-D.
    body_increment
        Body-frame translation increment applied during this prediction.
    variant
        One of ``"baseline"``, ``"r1"``, or ``"r1_r2"``.
    process_noise_cov
        Optional additive 2x2 process noise shared by all cells.
    """

    if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of "
            f"{SUPPORTED_RELAXED_S3F_VARIANTS}."
        )

    state = filter_.filter_state
    n_cells = len(state.linear_distributions)
    if state.lin_dim != 2:
        raise ValueError("predict_circular_relaxed requires a 2-D linear state.")

    grid = asarray(state.gd.get_grid(), dtype=float).reshape(-1)
    stats = uniform_circular_cell_statistics(n_cells, body_increment, grid=grid)

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
        zeros((2, 2), dtype=float)
        if process_noise_cov is None
        else asarray(process_noise_cov, dtype=float)
    )
    if q_base.shape != (2, 2):
        raise ValueError("process_noise_cov must have shape (2, 2).")

    covariance_matrices = stack(
        [q_base + covariance_inflations[idx] for idx in range(n_cells)],
        axis=2,
    )

    filter_.predict_linear(
        covariance_matrices=asarray(covariance_matrices),
        linear_input_vectors=asarray(displacements.T),
    )
    return stats


def grid_probability_masses(grid_values: Any) -> Any:
    """Convert equal-cell grid density values to normalized cell masses."""

    values = asarray(grid_values, dtype=float).reshape(-1)
    total = backend_sum(values)
    if float(total) <= 0.0:
        raise ValueError("grid values must have positive total mass.")
    return values / total


def circular_error(angle_a: float, angle_b: float) -> float:
    """Return the wrapped absolute angular error in radians."""

    return float(backend_abs(mod(angle_a - angle_b + pi, 2.0 * pi) - pi))


def circular_weighted_mean(angles: Any, weights: Any) -> float:
    """Return the circular mean angle for weighted grid values."""

    angles = asarray(angles, dtype=float).reshape(-1)
    weights = grid_probability_masses(weights)
    moment = backend_sum(weights * exp(1j * angles))
    return float(mod(backend_angle(moment), 2.0 * pi))


def _safe_sinc(x: float) -> float:
    if abs(x) < 1e-14:
        return 1.0
    return float(sin(x) / x)


def _as_body_increment(body_increment: Any) -> Any:
    u = asarray(body_increment, dtype=float).reshape(-1)
    if u.shape != (2,):
        raise ValueError("body_increment must be a 2-D vector.")
    return u
