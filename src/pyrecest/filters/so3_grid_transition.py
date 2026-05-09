"""Transition-density helpers for grid filters on SO(3)."""

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    abs,
    all,
    amax,
    array,
    clip,
    exp,
    linalg,
    mean,
    ndim,
    transpose,
)
from pyrecest.distributions._so3_helpers import (
    exp_map_identity,
    normalize_quaternions,
    quaternion_multiply,
)
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import (
    SdHalfCondSdHalfGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


def so3_right_multiplication_grid_transition(
    grid,
    orientation_increment,
    kappa,
) -> SdHalfCondSdHalfGridDistribution:
    """Build a soft grid transition for right-multiplicative SO(3) dynamics.

    The returned conditional density represents

    ``q_next = q_current * delta_q``

    on a scalar-last unit-quaternion grid. Columns condition on the current
    grid point and rows correspond to the next grid point, i.e.
    ``grid_values[i, j] = f(grid[i] | grid[j])``. This matches
    :meth:`HyperhemisphericalGridFilter.predict_nonlinear_via_transition_density`.

    Parameters
    ----------
    grid : array_like or object with ``get_grid()``
        Quaternion grid of shape ``(n_grid, 4)``. Quaternions are interpreted
        as scalar-last SO(3) representatives and canonicalized to the upper
        S3 hemisphere.
    orientation_increment : array_like
        Either a tangent-vector increment of shape ``(3,)`` at the identity or
        a scalar-last quaternion increment of shape ``(4,)``.
    kappa : float
        Positive concentration parameter. Larger values place more mass on the
        grid point nearest to ``q_current * delta_q``.

    Returns
    -------
    SdHalfCondSdHalfGridDistribution
        Normalized conditional density on the same canonicalized quaternion
        grid.

    Notes
    -----
    The unnormalized score is proportional to

    ``exp(kappa * |<q_next, q_current * delta_q>|**2)``.

    The columns are normalized by the S3 upper-hemisphere grid quadrature rule,
    so ``mean(grid_values[:, j]) * surface(S3+) == 1`` for every column.
    """

    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")

    quaternion_grid = _as_quaternion_grid(grid)
    delta_quaternion = _as_so3_increment(orientation_increment)

    targets = quaternion_multiply(quaternion_grid, delta_quaternion)
    inner_products = clip(abs(quaternion_grid @ transpose(targets)), 0.0, 1.0)

    # Subtracting the per-column maximum keeps the normalization stable for
    # large kappa without changing the normalized conditional density.
    exponents = kappa * inner_products**2
    scores = exp(exponents - amax(exponents, axis=0, keepdims=True))

    manifold_size = (
        0.5
        * AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
            quaternion_grid.shape[1]
        )
    )
    column_integrals = mean(scores, axis=0, keepdims=True) * manifold_size
    density_values = scores / column_integrals

    return SdHalfCondSdHalfGridDistribution(
        quaternion_grid,
        density_values,
        enforce_pdf_nonnegative=True,
    )


def quaternion_grid_transition_density(
    grid,
    orientation_increment,
    kappa,
) -> SdHalfCondSdHalfGridDistribution:
    """Alias for :func:`so3_right_multiplication_grid_transition`."""

    return so3_right_multiplication_grid_transition(
        grid,
        orientation_increment,
        kappa,
    )


def _as_quaternion_grid(grid):
    if hasattr(grid, "get_grid"):
        grid = grid.get_grid()

    quaternion_grid = array(grid, dtype=float)
    if ndim(quaternion_grid) != 2 or quaternion_grid.shape[1] != 4:
        raise ValueError("grid must have shape (n_grid, 4) with scalar-last quaternions.")
    if quaternion_grid.shape[0] == 0:
        raise ValueError("grid must contain at least one quaternion.")
    if not all(linalg.norm(quaternion_grid, axis=1) > 0.0):
        raise ValueError("grid quaternions must be nonzero.")

    return normalize_quaternions(quaternion_grid)


def _as_so3_increment(orientation_increment):
    values = array(orientation_increment, dtype=float)
    if ndim(values) == 1:
        if values.shape[0] == 3:
            return exp_map_identity(values)[0]
        if values.shape[0] == 4:
            return normalize_quaternions(values)[0]
    elif ndim(values) == 2 and values.shape[0] == 1:
        if values.shape[1] == 3:
            return exp_map_identity(values[0])[0]
        if values.shape[1] == 4:
            return normalize_quaternions(values)[0]

    raise ValueError(
        "orientation_increment must have shape (3,) tangent or (4,) quaternion."
    )


__all__ = [
    "quaternion_grid_transition_density",
    "so3_right_multiplication_grid_transition",
]
