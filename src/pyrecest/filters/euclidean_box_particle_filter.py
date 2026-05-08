"""Euclidean box particle filter."""

import copy
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import product
from typing import Union

import numpy as _np

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    all,
    amax,
    amin,
    arange,
    array,
    diag,
    expand_dims,
    int32,
    int64,
    logical_and,
    maximum,
    minimum,
    ones,
    ones_like,
    prod,
    random,
    reshape,
    sqrt,
    sum,
    to_numpy,
    vstack,
    where,
    zeros_like,
)
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from pyrecest.distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import EuclideanFilterMixin


class EuclideanBoxParticleFilter(AbstractParticleFilter, EuclideanFilterMixin):
    """Box particle filter for Euclidean state spaces.

    The state is represented by a weighted mixture of uniform hyperrectangles.
    Prediction propagates boxes through an inclusion function.  Correction
    contracts predicted boxes and multiplies the weights by the contracted
    volume divided by the predicted volume.
    """

    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        dim: Union[int, int32, int64],
        box_half_width=0.5,
        resampling_criterion: Callable | None = None,
        split_resampled_boxes: bool = True,
    ):
        n_particles = int(n_particles)
        dim = int(dim)
        if n_particles <= 0:
            raise ValueError("n_particles must be a positive integer")
        if dim <= 0:
            raise ValueError("dim must be a positive integer")

        self.default_box_half_width = LinearBoxParticleDistribution._coerce_half_width(
            box_half_width, dim
        )
        self.split_resampled_boxes = split_resampled_boxes

        lower = -ones((n_particles, dim)) * reshape(
            self.default_box_half_width, (1, -1)
        )
        upper = ones((n_particles, dim)) * reshape(
            self.default_box_half_width, (1, -1)
        )
        initial_distribution = LinearBoxParticleDistribution(lower, upper)

        EuclideanFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(
            self,
            initial_distribution,
            resampling_criterion=resampling_criterion,
        )

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, LinearBoxParticleDistribution):
            dist_box = copy.deepcopy(new_state)
        elif isinstance(new_state, AbstractLinearDistribution):
            dist_box = LinearBoxParticleDistribution.from_distribution(
                new_state,
                n_particles=self._filter_state.w.shape[0],
                box_half_width=self.default_box_half_width,
            )
        else:
            raise TypeError(
                "EuclideanBoxParticleFilter state must be a LinearBoxParticleDistribution "
                "or another AbstractLinearDistribution that can be sampled."
            )

        if dist_box.dim != self._filter_state.dim:
            raise ValueError("New filter state has the wrong dimension")
        if dist_box.w.shape[0] != self._filter_state.w.shape[0]:
            raise ValueError("New filter state has the wrong number of boxes")
        self._filter_state = dist_box

    def predict_identity(
        self,
        noise_distribution=None,
        process_noise_bounds=None,
        scaling_factor: float = 3.0,
    ):
        """Predict with identity dynamics and bounded additive process noise."""
        if process_noise_bounds is None and noise_distribution is not None:
            process_noise_bounds = self._noise_distribution_to_bounds(
                noise_distribution, scaling_factor
            )
        if process_noise_bounds is None:
            return self
        return self.predict_interval(lambda lower, upper: (lower, upper), process_noise_bounds)

    def predict_interval(
        self,
        inclusion_function: Callable,
        process_noise_bounds=None,
        function_is_vectorized: bool = True,
    ):
        """Predict boxes using an interval inclusion function.

        ``inclusion_function`` must return ``(lower, upper)`` for the image of
        the input boxes.  In vectorized mode it receives arrays with shape
        ``(n_boxes, dim)``; otherwise it is called once per box with vectors of
        shape ``(dim,)``.
        """
        prior = self.filter_state
        if function_is_vectorized:
            predicted = inclusion_function(prior.lower, prior.upper)
            pred_lower, pred_upper = self._coerce_box_result(predicted)
        else:
            lower_parts = []
            upper_parts = []
            for i in range(prior.w.shape[0]):
                pred_lower_i, pred_upper_i = self._coerce_box_result(
                    inclusion_function(prior.lower[i], prior.upper[i])
                )
                lower_parts.append(pred_lower_i)
                upper_parts.append(pred_upper_i)
            pred_lower = vstack(lower_parts)
            pred_upper = vstack(upper_parts)

        if process_noise_bounds is not None:
            noise_lower, noise_upper = self._coerce_interval(
                process_noise_bounds, None, self.dim
            )
            pred_lower = pred_lower + reshape(noise_lower, (1, -1))
            pred_upper = pred_upper + reshape(noise_upper, (1, -1))

        self._filter_state = LinearBoxParticleDistribution(
            pred_lower,
            pred_upper,
            prior.w,
        )
        return self

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = False,  # kept for API compatibility
        process_noise_bounds=None,
        inclusion_function: Callable | None = None,
        scaling_factor: float = 3.0,
    ):
        """Predict through nonlinear dynamics.

        For guaranteed interval propagation, pass ``inclusion_function``.  If no
        inclusion function is supplied, the method encloses the images of the box
        corners.  Corner enclosure is exact for affine maps and often useful for
        monotone maps, but it is not a mathematically guaranteed inclusion for an
        arbitrary nonlinear function.
        """
        del shift_instead_of_add
        if process_noise_bounds is None and noise_distribution is not None:
            process_noise_bounds = self._noise_distribution_to_bounds(
                noise_distribution, scaling_factor
            )
        if inclusion_function is not None:
            return self.predict_interval(
                inclusion_function,
                process_noise_bounds=process_noise_bounds,
                function_is_vectorized=function_is_vectorized,
            )

        prior = self.filter_state
        pred_lower, pred_upper = self._corner_enclosure(
            f,
            prior.lower,
            prior.upper,
            function_is_vectorized=function_is_vectorized,
        )
        if process_noise_bounds is not None:
            noise_lower, noise_upper = self._coerce_interval(
                process_noise_bounds, None, self.dim
            )
            pred_lower = pred_lower + reshape(noise_lower, (1, -1))
            pred_upper = pred_upper + reshape(noise_upper, (1, -1))
        self._filter_state = LinearBoxParticleDistribution(pred_lower, pred_upper, prior.w)
        return self

    def update_identity_box(
        self,
        measurement_lower,
        measurement_upper=None,
        measurement_noise_bounds=None,
    ):
        """Update by intersecting state boxes with an identity measurement box.

        If the measurement model is ``z = x + v`` and ``measurement_noise_bounds``
        is supplied as ``(v_lower, v_upper)``, the state constraint is
        ``x in [z_lower - v_upper, z_upper - v_lower]``.
        """
        z_lower, z_upper = self._coerce_interval(
            measurement_lower, measurement_upper, self.dim
        )
        if measurement_noise_bounds is not None:
            noise_lower, noise_upper = self._coerce_interval(
                measurement_noise_bounds, None, self.dim
            )
            z_lower = z_lower - noise_upper
            z_upper = z_upper - noise_lower

        z_lower = reshape(z_lower, (1, -1))
        z_upper = reshape(z_upper, (1, -1))

        def contractor(lower, upper):
            return maximum(lower, z_lower), minimum(upper, z_upper)

        return self.update_contracted(contractor)

    def update_contracted(
        self,
        contractor: Callable,
        measurement=None,
        likelihood: Callable | None = None,
    ):
        """Update by contracting predicted boxes.

        ``contractor`` receives ``(lower, upper)`` or ``(lower, upper,
        measurement)`` and returns contracted ``(lower, upper)`` arrays.  The Box
        PF likelihood is the contracted volume divided by the predicted volume.
        An optional additional likelihood can be multiplied in, evaluated at the
        contracted box centers.
        """
        predicted = self.filter_state
        previous_volumes = predicted.volumes()

        if measurement is None:
            contracted = contractor(predicted.lower, predicted.upper)
        else:
            contracted = contractor(predicted.lower, predicted.upper, measurement)
        contracted_lower, contracted_upper = self._coerce_box_result(contracted)

        valid = all(contracted_upper >= contracted_lower, axis=1)
        contracted_widths = maximum(contracted_upper - contracted_lower, 0.0)
        contracted_volumes = prod(contracted_widths, axis=1)
        safe_previous_volumes = where(
            previous_volumes > 0, previous_volumes, ones_like(previous_volumes)
        )
        ratios = where(
            logical_and(valid, previous_volumes > 0),
            contracted_volumes / safe_previous_volumes,
            zeros_like(previous_volumes),
        )

        if likelihood is not None:
            centers = 0.5 * (contracted_lower + contracted_upper)
            likelihood_values = likelihood(centers)
            ratios = ratios * likelihood_values

        new_weights = predicted.w * ratios
        if bool(sum(new_weights) <= 0):
            raise ValueError(
                "All contracted boxes have zero likelihood; check the contractor, "
                "measurement, or noise bounds."
            )
        new_weights = new_weights / sum(new_weights)

        valid_column = expand_dims(valid, 1)
        safe_lower = where(valid_column, contracted_lower, predicted.lower)
        safe_upper = where(valid_column, contracted_upper, predicted.upper)
        self._filter_state = LinearBoxParticleDistribution(
            safe_lower,
            safe_upper,
            new_weights,
        )
        self.resample_if_needed()
        return self

    def update_nonlinear_using_likelihood(self, likelihood, measurement=None):
        """Fallback update using a point likelihood at box centers."""
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            self._filter_state = self.filter_state.reweigh(likelihood)
        else:
            self._filter_state = self.filter_state.reweigh(
                lambda centers: likelihood(measurement, centers)
            )
        self.resample_if_needed()
        return self

    def association_likelihood(self, likelihood: AbstractManifoldSpecificDistribution):
        values = likelihood.pdf(self.filter_state.centers())
        return sum(values * self.filter_state.w)

    def effective_sample_size(self):
        """Return the usual particle-filter effective sample size."""
        return 1.0 / sum(self.filter_state.w**2)

    def resample(self, split_resampled_boxes: bool | None = None):
        """Multinomially resample boxes and split duplicates along their widest side."""
        if split_resampled_boxes is None:
            split_resampled_boxes = self.split_resampled_boxes

        state = self.filter_state
        n_boxes = int(state.w.shape[0])
        indices = random.choice(arange(n_boxes), n_boxes, p=state.w)
        indices_np = _np.asarray(to_numpy(indices), dtype=int).reshape(-1)
        lower_np = _np.asarray(to_numpy(state.lower), dtype=float)
        upper_np = _np.asarray(to_numpy(state.upper), dtype=float)

        counts = Counter(indices_np.tolist())
        seen = defaultdict(int)
        new_lower = _np.empty_like(lower_np)
        new_upper = _np.empty_like(upper_np)

        for out_idx, source_idx in enumerate(indices_np):
            source_idx = int(source_idx)
            lower = lower_np[source_idx].copy()
            upper = upper_np[source_idx].copy()
            multiplicity = counts[source_idx]

            if split_resampled_boxes and multiplicity > 1:
                occurrence = seen[source_idx]
                split_dim = int(_np.argmax(upper - lower))
                edges = _np.linspace(lower[split_dim], upper[split_dim], multiplicity + 1)
                lower[split_dim] = edges[occurrence]
                upper[split_dim] = edges[occurrence + 1]
                seen[source_idx] += 1

            new_lower[out_idx] = lower
            new_upper[out_idx] = upper

        self._filter_state = LinearBoxParticleDistribution(
            array(new_lower),
            array(new_upper),
            ones(n_boxes) / n_boxes,
        )
        return self

    def _corner_enclosure(self, f, lower, upper, function_is_vectorized: bool):
        n_boxes = int(lower.shape[0])
        dim = int(lower.shape[1])
        patterns = _np.asarray(list(product([False, True], repeat=dim)), dtype=bool)
        n_corners = int(patterns.shape[0])

        lower_np = _np.asarray(to_numpy(lower), dtype=float)
        upper_np = _np.asarray(to_numpy(upper), dtype=float)
        corners_np = _np.where(patterns[None, :, :], upper_np[:, None, :], lower_np[:, None, :])
        corners = array(corners_np.reshape(n_boxes * n_corners, dim))

        if function_is_vectorized:
            values = f(corners)
        else:
            values = vstack([f(corners[i]) for i in range(corners.shape[0])])
        values = array(values)
        if values.ndim == 1:
            values = reshape(values, (-1, 1))

        values = reshape(values, (n_boxes, n_corners, -1))
        return amin(values, axis=1), amax(values, axis=1)

    def _noise_distribution_to_bounds(self, noise_distribution, scaling_factor: float):
        if isinstance(noise_distribution, (tuple, list)):
            return self._coerce_interval(noise_distribution, None, self.dim)
        if hasattr(noise_distribution, "bounds"):
            return self._coerce_interval(noise_distribution.bounds, None, self.dim)
        if hasattr(noise_distribution, "mean") and hasattr(noise_distribution, "covariance"):
            mean = noise_distribution.mean()
            std = sqrt(diag(noise_distribution.covariance()))
            return mean - scaling_factor * std, mean + scaling_factor * std
        raise TypeError(
            "noise_distribution must expose bounds, or mean/covariance, or pass "
            "process_noise_bounds explicitly."
        )

    @staticmethod
    def _coerce_box_result(result):
        if isinstance(result, LinearBoxParticleDistribution):
            return result.lower, result.upper
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Box functions must return (lower, upper)")
        lower, upper = result
        lower = array(lower)
        upper = array(upper)
        if lower.ndim == 1:
            lower = reshape(lower, (1, -1))
        if upper.ndim == 1:
            upper = reshape(upper, (1, -1))
        if lower.shape != upper.shape:
            raise ValueError("Returned lower and upper boxes have different shapes")
        return lower, upper

    @staticmethod
    def _coerce_interval(lower, upper, dim: int):
        if upper is None:
            if isinstance(lower, (tuple, list)) and len(lower) == 2:
                return EuclideanBoxParticleFilter._coerce_interval(lower[0], lower[1], dim)
            bounds = array(lower)
            if bounds.ndim == 1 and dim == 1 and bounds.shape[0] == 2:
                return reshape(bounds[0], (1,)), reshape(bounds[1], (1,))
            if bounds.ndim == 2 and bounds.shape == (2, dim):
                return bounds[0], bounds[1]
            if bounds.ndim == 2 and bounds.shape == (dim, 2):
                return bounds[:, 0], bounds[:, 1]
            raise ValueError(
                "Bounds must be supplied as (lower, upper), shape (2, dim), "
                "or shape (dim, 2)."
            )
        return (
            EuclideanBoxParticleFilter._coerce_vector(lower, dim, "lower"),
            EuclideanBoxParticleFilter._coerce_vector(upper, dim, "upper"),
        )

    @staticmethod
    def _coerce_vector(value, dim: int, name: str):
        value = array(value)
        if value.ndim == 0:
            value = reshape(value, (1,))
        else:
            value = reshape(value, (-1,))
        if value.shape[0] != dim:
            raise ValueError(f"{name} must have dimension {dim}")
        return value


BoxParticleFilter = EuclideanBoxParticleFilter
