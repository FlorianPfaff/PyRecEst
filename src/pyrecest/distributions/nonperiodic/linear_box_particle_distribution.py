"""Weighted box-particle distributions on Euclidean spaces."""

import copy
import warnings
from typing import Union

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    all,
    any,
    arange,
    argmax,
    array,
    diag,
    expand_dims,
    int32,
    int64,
    isclose,
    logical_and,
    maximum,
    minimum,
    ones,
    ones_like,
    prod,
    random,
    reshape,
    sum,
    where,
    zeros_like,
)

from .abstract_linear_distribution import AbstractLinearDistribution


class LinearBoxParticleDistribution(AbstractLinearDistribution):
    r"""Mixture of uniform densities supported on axis-aligned boxes.

    The represented density is

    .. math:: p(x) = \sum_i w_i \mathcal{U}(x; [\ell_i, u_i]),

    where each box particle is encoded by a lower and an upper corner.  The
    ``d`` property returns the box centers so that the class can interoperate
    with APIs that expect particle locations, while the full box support is kept
    in ``lower`` and ``upper``.
    """

    def __init__(self, lower, upper=None, w=None):
        lower, upper = self._coerce_box_arrays(lower, upper)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if lower.ndim != 2:
            raise ValueError("lower and upper must be arrays with shape (n_boxes, dim)")
        if not bool(all(upper >= lower)):
            raise ValueError("Each box must satisfy upper >= lower component-wise")

        AbstractLinearDistribution.__init__(self, int(lower.shape[1]))
        self.lower = copy.copy(lower)
        self.upper = copy.copy(upper)

        n_boxes = int(lower.shape[0])
        if w is None:
            self.w = ones(n_boxes) / n_boxes
        else:
            weights = array(w)
            if weights.ndim != 1:
                weights = reshape(weights, (-1,))
            if weights.shape[0] != n_boxes:
                raise ValueError("Number of weights and boxes must match")
            if bool(any(weights < 0)):
                raise ValueError("Weights must be nonnegative")
            self.w = copy.copy(weights)
        self.normalize_in_place()

    @staticmethod
    def _coerce_box_arrays(lower, upper=None):
        lower = array(lower)
        if upper is None:
            if lower.ndim != 3 or lower.shape[1] != 2:
                raise ValueError(
                    "When upper is omitted, lower must have shape (n_boxes, 2, dim)"
                )
            boxes = lower
            lower = boxes[:, 0, :]
            upper = boxes[:, 1, :]
        else:
            upper = array(upper)

        if lower.ndim == 1:
            lower = reshape(lower, (1, -1))
        if upper.ndim == 1:
            upper = reshape(upper, (1, -1))
        return lower, upper

    @property
    def d(self):
        """Return the box centers for compatibility with particle APIs."""
        return self.centers()

    @d.setter
    def d(self, new_centers):
        """Move boxes to new centers while preserving their half widths."""
        new_centers = array(new_centers)
        if new_centers.ndim == 1:
            new_centers = reshape(new_centers, (1, -1))
        if new_centers.shape != self.lower.shape:
            raise ValueError(
                f"new centers have shape {new_centers.shape}, expected {self.lower.shape}"
            )
        half_widths = self.half_widths()
        self.lower = new_centers - half_widths
        self.upper = new_centers + half_widths

    def normalize_in_place(self):
        """Normalize weights in-place."""
        total_weight = sum(self.w)
        if bool(total_weight <= 0):
            raise ValueError("At least one box particle must have positive weight")
        if not bool(isclose(total_weight, 1.0, atol=1e-10)):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
            self.w = self.w / total_weight

    def normalize(self):
        dist = copy.deepcopy(self)
        dist.normalize_in_place()
        return dist

    def centers(self):
        """Return all box centers with shape ``(n_boxes, dim)``."""
        return 0.5 * (self.lower + self.upper)

    def widths(self):
        """Return side lengths of all boxes."""
        return self.upper - self.lower

    def half_widths(self):
        """Return side half-lengths of all boxes."""
        return 0.5 * self.widths()

    def volumes(self):
        """Return the hypervolume of every box particle."""
        return prod(maximum(self.widths(), 0.0), axis=1)

    def mean(self):
        """Return the mixture mean."""
        return self.w @ self.centers()

    def covariance(self):
        """Return the covariance of the uniform box mixture.

        This includes both the covariance of the box centers and the within-box
        covariance of each uniform hyperrectangle, ``diag(width**2 / 12)``.
        """
        centers = self.centers()
        mean = self.mean()
        deviations = centers - mean
        between_box_cov = (deviations.T * self.w) @ deviations
        within_box_var = self.w @ (self.widths() ** 2 / 12.0)
        return between_box_cov + diag(within_box_var)

    def set_mean(self, new_mean):
        """Shift all boxes so the mixture has the supplied mean."""
        offset = array(new_mean) - self.mean()
        offset = reshape(offset, (1, -1))
        self.lower = self.lower + offset
        self.upper = self.upper + offset
        return self

    def sample(self, n: Union[int, int32, int64]):
        """Draw point samples from the represented box mixture."""
        n = int(n)
        indices = random.choice(arange(self.w.shape[0]), n, p=self.w)
        lower = self.lower[indices]
        upper = self.upper[indices]
        unit_samples = random.uniform(size=(n, self.dim))
        return lower + unit_samples * (upper - lower)

    def pdf(self, xs):
        """Evaluate the mixture density at ``xs``.

        For one-dimensional distributions, a vector of shape ``(n,)`` is treated
        as ``n`` scalar evaluation points.  For higher dimensions, a vector of
        shape ``(dim,)`` is treated as a single point.
        """
        xs = self._coerce_points(xs)
        xs_expanded = expand_dims(xs, 1)
        lower_expanded = expand_dims(self.lower, 0)
        upper_expanded = expand_dims(self.upper, 0)
        inside = all(
            logical_and(xs_expanded >= lower_expanded, xs_expanded <= upper_expanded),
            axis=2,
        )
        volumes = self.volumes()
        safe_volumes = where(volumes > 0, volumes, ones_like(volumes))
        box_densities = where(volumes > 0, self.w / safe_volumes, zeros_like(self.w))
        return where(inside, 1.0, 0.0) @ box_densities

    def _coerce_points(self, xs):
        xs = array(xs)
        if xs.ndim == 0:
            if self.dim != 1:
                raise ValueError("Scalar points are only valid for dim == 1")
            return reshape(xs, (1, 1))
        if xs.ndim == 1:
            if self.dim == 1:
                return reshape(xs, (-1, 1))
            if xs.shape[0] == self.dim:
                return reshape(xs, (1, self.dim))
            raise ValueError(
                f"Point dimension {xs.shape[0]} does not match dim {self.dim}"
            )
        if xs.ndim != 2 or xs.shape[1] != self.dim:
            raise ValueError(f"xs must have shape (n, {self.dim})")
        return xs

    def reweigh(self, f):
        """Return a copy with weights multiplied by ``f`` evaluated at centers."""
        dist = copy.deepcopy(self)
        weights_update = f(dist.centers())
        if weights_update.shape != dist.w.shape:
            raise ValueError("Function returned wrong output dimensions")
        if bool(any(weights_update < 0)):
            raise ValueError("All weight updates must be nonnegative")
        new_weights = dist.w * weights_update
        if bool(sum(new_weights) <= 0):
            raise ValueError("The sum of all weights must be positive")
        dist.w = new_weights / sum(new_weights)
        return dist

    def integrate(self, left=None, right=None):
        """Integrate the mixture over an axis-aligned query box."""
        if left is None and right is None:
            return sum(self.w)
        if left is None or right is None:
            raise ValueError("left and right must either both be given or both be None")

        left = self._coerce_vector(left, "left")
        right = self._coerce_vector(right, "right")
        intersection_lower = maximum(self.lower, reshape(left, (1, -1)))
        intersection_upper = minimum(self.upper, reshape(right, (1, -1)))
        intersection_volumes = prod(
            maximum(intersection_upper - intersection_lower, 0.0), axis=1
        )
        volumes = self.volumes()
        safe_volumes = where(volumes > 0, volumes, ones_like(volumes))
        ratios = where(
            volumes > 0, intersection_volumes / safe_volumes, zeros_like(volumes)
        )
        return sum(self.w * ratios)

    def _coerce_vector(self, value, name):
        value = array(value)
        if value.ndim == 0:
            value = reshape(value, (1,))
        else:
            value = reshape(value, (-1,))
        if value.shape[0] != self.dim:
            raise ValueError(f"{name} must have dimension {self.dim}")
        return value

    def mode(self, starting_point=None):  # pylint: disable=unused-argument
        """Return the center of the box with highest mixture density."""
        volumes = self.volumes()
        safe_volumes = where(volumes > 0, volumes, ones_like(volumes))
        box_densities = where(volumes > 0, self.w / safe_volumes, zeros_like(self.w))
        return self.centers()[int(argmax(box_densities))]

    @staticmethod
    def from_distribution(
        distribution, n_particles=None, n_samples=None, n=None, box_half_width=0.5
    ):
        """Create equally sized boxes around samples from another distribution."""
        particle_count = LinearBoxParticleDistribution._resolve_particle_count(
            n_particles=n_particles,
            n_samples=n_samples,
            n=n,
        )
        samples = distribution.sample(particle_count)
        half_width = LinearBoxParticleDistribution._coerce_half_width(
            box_half_width, distribution.dim
        )
        return LinearBoxParticleDistribution(
            samples - reshape(half_width, (1, -1)),
            samples + reshape(half_width, (1, -1)),
            ones(particle_count) / particle_count,
        )

    @staticmethod
    def _resolve_particle_count(n_particles=None, n_samples=None, n=None):
        specified_counts = [
            value for value in (n_particles, n_samples, n) if value is not None
        ]
        if not specified_counts:
            raise ValueError(
                "LinearBoxParticleDistribution.from_distribution requires "
                "n_particles, n_samples, or n."
            )
        particle_counts = [int(value) for value in specified_counts]
        if len(set(particle_counts)) != 1:
            raise ValueError(
                "n_particles, n_samples, and n must agree when more than one "
                "particle-count alias is supplied."
            )
        particle_count = particle_counts[0]
        if particle_count <= 0:
            raise ValueError("Number of particles must be positive.")
        return particle_count

    @staticmethod
    def _coerce_half_width(box_half_width, dim):
        half_width = array(box_half_width)
        if half_width.ndim == 0:
            half_width = ones(dim) * half_width
        else:
            half_width = reshape(half_width, (-1,))
        if half_width.shape[0] != dim:
            raise ValueError(f"box_half_width must have dimension {dim}")
        if bool(any(half_width < 0)):
            raise ValueError("box_half_width must be nonnegative")
        return half_width
