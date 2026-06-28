# pylint: disable=no-name-in-module,no-member
from math import isfinite as math_isfinite

from pyrecest.backend import all as backend_all
from pyrecest.backend import array, diff, isfinite, prod, reshape, to_numpy
from scipy.integrate import nquad

from ..abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)

_ERROR_SCALAR_PDF_VALUE = (
    "pdf must return one finite scalar value per integration point"
)


def _require_finite_bounds(bounds, name: str) -> None:
    try:
        finite = bool(backend_all(isfinite(bounds)))
    except TypeError as exc:
        raise ValueError(f"{name} must contain only finite values") from exc
    if not finite:
        raise ValueError(f"{name} must contain only finite values")


def _require_increasing_bounds(bounds, name: str) -> None:
    widths = diff(bounds, axis=1)
    if not bool(backend_all(widths > 0.0)):
        raise ValueError(f"{name} must be strictly increasing in every dimension")


def _require_scalar_pdf_value(values) -> float:
    try:
        flat_values = to_numpy(array(values)).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(_ERROR_SCALAR_PDF_VALUE) from exc
    if flat_values.size != 1:
        raise ValueError(_ERROR_SCALAR_PDF_VALUE)
    try:
        value = float(flat_values[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_ERROR_SCALAR_PDF_VALUE) from exc
    if not math_isfinite(value):
        raise ValueError(_ERROR_SCALAR_PDF_VALUE)
    return value


class AbstractHyperrectangularDistribution(AbstractBoundedNonPeriodicDistribution):
    def __init__(self, bounds):
        bounds = array(bounds)
        if bounds.ndim == 1:
            if bounds.shape[0] != 2:
                raise ValueError("one-dimensional bounds must have length 2")
            bounds = reshape(bounds, (1, 2))
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2)")
        _require_finite_bounds(bounds, "bounds")
        _require_increasing_bounds(bounds, "bounds")
        AbstractBoundedNonPeriodicDistribution.__init__(self, int(bounds.shape[0]))
        self.bounds = bounds

    def get_manifold_size(self):
        s = prod(diff(self.bounds, axis=1))
        return s

    @property
    def input_dim(self):
        return self.dim

    def integrate(self, integration_boundaries=None) -> float:
        """
        Integrate the probability density function over given boundaries.
        If no boundaries are provided, default to `self.bounds`.

        Args:
            integration_boundaries (tuple): A tuple of two elements, each of which can be either a scalar or an array.
                If a scalar, it represents a single boundary value.
                If an array, it represents multiple boundary values.

        Returns:
            float: The result of the integration.
        """
        if integration_boundaries is None:
            integration_boundaries = self.bounds

        try:
            integration_boundaries = reshape(array(integration_boundaries), (-1, 2))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"integration_boundaries must have shape ({self.dim}, 2)"
            ) from exc
        if integration_boundaries.shape[0] != self.dim:
            raise ValueError(f"integration_boundaries must have shape ({self.dim}, 2)")
        _require_finite_bounds(integration_boundaries, "integration_boundaries")
        _require_increasing_bounds(integration_boundaries, "integration_boundaries")
        left = integration_boundaries[:, 0]
        right = integration_boundaries[:, 1]
        ranges = [
            (float(lower), float(upper))
            for lower, upper in zip(to_numpy(left), to_numpy(right))
        ]

        def integrand(*args):
            values = self.pdf(reshape(array(args), (1, self.dim)))
            return _require_scalar_pdf_value(values)

        return nquad(integrand, ranges)[0]
