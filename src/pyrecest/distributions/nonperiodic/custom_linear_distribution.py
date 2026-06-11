import copy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ndim, reshape, zeros

from ..abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from .abstract_linear_distribution import AbstractLinearDistribution


def _as_shift_vector(shift_by, dim: int, *, name: str = "shift_by"):
    shift_by = array(shift_by)
    if shift_by.ndim == 0:
        if dim != 1:
            raise ValueError(f"{name} must have shape ({dim},), got scalar.")
        return reshape(shift_by, (1,))
    if shift_by.ndim == 1 and shift_by.shape[0] == dim:
        return shift_by
    raise ValueError(f"{name} must have shape ({dim},), got {shift_by.shape}.")


class CustomLinearDistribution(
    AbstractLinearDistribution, AbstractCustomNonPeriodicDistribution
):
    """
    Linear distribution with custom pdf.
    """

    def __init__(self, f, dim, scale_by=1, shift_by=None):
        """
        Constructor, it is the user's responsibility to ensure that f is a valid
        linear density.

        Parameters:
        f_ (function handle)
            pdf of the distribution
        dim_ (scalar)
            dimension of the Euclidean space
        """
        AbstractLinearDistribution.__init__(self, dim=dim)
        AbstractCustomNonPeriodicDistribution.__init__(self, f, scale_by=scale_by)
        if shift_by is not None:
            self.shift_by = _as_shift_vector(shift_by, dim)
        else:
            self.shift_by = zeros(dim)

    def shift(self, shift_by):
        shift_by = _as_shift_vector(shift_by, self.dim)
        cd = copy.deepcopy(self)
        cd.shift_by = self.shift_by + shift_by
        return cd

    def set_mean(self, new_mean):
        new_mean = _as_shift_vector(new_mean, self.dim, name="new_mean")
        mean_offset = new_mean - self.mean()
        self.shift_by = self.shift_by + mean_offset
        if self._mean_numerical is not None:
            self._mean_numerical = new_mean

    def pdf(self, xs):
        xs = array(xs)
        if not (
            (self.dim == 1 and xs.ndim <= 1)
            or (xs.ndim > 0 and xs.shape[-1] == self.dim)
        ):
            raise ValueError(
                f"xs must be scalar, one-dimensional for dim=1, or have last "
                f"dimension {self.dim}; got shape {xs.shape}."
            )
        p = self.scale_by * self.f(
            # To ensure 2-d for broadcasting
            reshape(xs, (-1, self.dim))
            - reshape(self.shift_by, (1, -1))
        )
        if ndim(p) > 1:
            raise ValueError("The custom pdf function must return at most 1-D values.")
        return p

    @staticmethod
    def from_distribution(distribution):
        """
        Creates a CustomLinearDistribution from some other distribution

        Parameters:
        dist (AbstractLinearDistribution)
            distribution to convert

        Returns:
        chd (CustomLinearDistribution)
            CustomLinearDistribution with identical pdf
        """
        chd = CustomLinearDistribution(distribution.pdf, distribution.dim)
        return chd

    def integrate(self, left=None, right=None):
        return AbstractLinearDistribution.integrate(self, left, right)
