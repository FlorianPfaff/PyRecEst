"""Euclidean boxed particle filter.

This is a point-particle filter with box-constrained particle generation. It is
not a box particle filter: the filter state remains a weighted Dirac mixture;
boxes are used as predicted support/proposal regions.
"""

import copy
from collections.abc import Callable
from typing import Union

import numpy as _np

# pylint: disable=no-name-in-module,no-member,redefined-builtin,too-many-positional-arguments
from pyrecest.backend import (
    all,
    any,
    array,
    int32,
    int64,
    logical_and,
    ones,
    random,
    reshape,
    sum,
    to_numpy,
    where,
    zeros_like,
)
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)

from .euclidean_particle_filter import EuclideanParticleFilter


class EuclideanBoxedParticleFilter(EuclideanParticleFilter):
    """Point-particle filter with boxed particle-generation support.

    This class implements the ``boxed PF`` interpretation in which particles
    remain ordinary Dirac point particles, but a box is used as a proposal or
    support constraint. This is deliberately distinct from
    :class:`EuclideanBoxParticleFilter`, whose particles themselves are boxes.
    """

    _VALID_GENERATION_METHODS = {
        "accept",
        "inscribed_gaussian",
        "resample",
        "reweight",
        "uniform",
        "upsample",
    }

    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        dim: Union[int, int32, int64],
        box_lower=None,
        box_upper=None,
        boxed_generation_method: str = "uniform",
        resampling_criterion: Callable | None = None,
        gaussian_sigma_scale: float = 3.0,
        max_sampling_iterations: Union[int, int32, int64] = 100,
    ):
        super().__init__(int(n_particles), int(dim))
        self.resampling_criterion = resampling_criterion
        self.default_boxed_generation_method = self._validate_generation_method(
            boxed_generation_method
        )
        self.gaussian_sigma_scale = self._validate_positive_float(
            gaussian_sigma_scale, "gaussian_sigma_scale"
        )
        self.max_sampling_iterations = self._validate_positive_int(
            max_sampling_iterations, "max_sampling_iterations"
        )
        self.box_lower = None
        self.box_upper = None
        if box_lower is None and box_upper is not None:
            raise ValueError("box_lower must be supplied when box_upper is supplied")
        if box_lower is not None:
            self.set_box_bounds(box_lower, box_upper)

    def set_box_bounds(self, box_lower, box_upper=None):
        """Store an axis-aligned box used by boxed particle generation."""
        lower, upper = self._coerce_interval(box_lower, box_upper, self.dim)
        self._validate_box(lower, upper, require_positive_volume=True)
        self.box_lower = lower
        self.box_upper = upper
        return self

    def clear_box_bounds(self):
        """Remove the stored boxed proposal/support constraint."""
        self.box_lower = None
        self.box_upper = None
        return self

    def get_box_bounds(self):
        """Return the stored box bounds."""
        if self.box_lower is None or self.box_upper is None:
            raise ValueError("No box bounds are stored; pass box_lower and box_upper")
        return self.box_lower, self.box_upper

    def particles_inside_box(self, box_lower=None, box_upper=None):
        """Return a Boolean mask selecting particles inside the given box."""
        lower, upper = self._resolve_box(box_lower, box_upper)
        return self._points_inside_box(self.filter_state.d, lower, upper)

    def generate_boxed_particles(
        self,
        box_lower=None,
        box_upper=None,
        boxed_generation_method: str | None = None,
        proposal_distribution=None,
        batch_size: Union[int, int32, int64] | None = None,
        max_sampling_iterations: Union[int, int32, int64] | None = None,
        max_tries_per_particle: Union[int, int32, int64] | None = None,
        gaussian_sigma_scale: float | None = None,
    ):
        """Force the point-particle set to respect a boxed support constraint.

        Parameters
        ----------
        box_lower, box_upper : array-like, optional
            Lower and upper corners of the support box. If omitted, the stored
            box bounds set with :meth:`set_box_bounds` are used.
        boxed_generation_method : str, optional
            Supported values are ``"uniform"``, ``"accept"``, ``"resample"``,
            ``"upsample"``, ``"inscribed_gaussian"``, and ``"reweight"``.
            ``"accept"`` and ``"resample"`` are synonyms that resample from
            existing in-box particles. ``"upsample"`` repeatedly draws from a
            proposal distribution. ``"inscribed_gaussian"`` uses a Gaussian
            centered in the box and rejects samples outside the box.
        proposal_distribution : distribution or callable, optional
            Distribution with a ``sample(n)`` method, or a callable returning
            samples, used by ``"upsample"``.
        batch_size : int, optional
            Proposal batch size for rejection-based methods.
        max_sampling_iterations : int, optional
            Maximum number of proposal batches.
        max_tries_per_particle : int, optional
            Backward-compatible shorthand for ``max_sampling_iterations`` in
            ``"upsample"`` mode.
        gaussian_sigma_scale : float, optional
            Number of marginal standard deviations corresponding to the box
            half-width for ``"inscribed_gaussian"``.
        """
        lower, upper = self._resolve_box(box_lower, box_upper)
        method = (
            self.default_boxed_generation_method
            if boxed_generation_method is None
            else self._validate_generation_method(boxed_generation_method)
        )

        if method == "reweight":
            return self.reweight_by_box(lower, upper)

        self._validate_box(lower, upper, require_positive_volume=True)
        n_particles = int(self.filter_state.w.shape[0])
        if method == "uniform":
            new_particles = self._sample_uniform_in_box(lower, upper, n_particles)
        elif method in {"accept", "resample"}:
            new_particles = self._resample_from_inside_box(lower, upper, n_particles)
        elif method == "upsample":
            new_particles = self._sample_from_proposal_inside_box(
                proposal_distribution,
                lower,
                upper,
                n_particles,
                batch_size=batch_size,
                max_sampling_iterations=max_sampling_iterations,
                max_tries_per_particle=max_tries_per_particle,
            )
        elif method == "inscribed_gaussian":
            new_particles = self._sample_from_inscribed_gaussian(
                lower,
                upper,
                n_particles,
                batch_size=batch_size,
                max_sampling_iterations=max_sampling_iterations,
                gaussian_sigma_scale=gaussian_sigma_scale,
            )
        else:  # pragma: no cover - guarded by _validate_generation_method
            raise RuntimeError(f"Unsupported boxed generation method {method!r}")

        self._filter_state = LinearDiracDistribution(
            new_particles,
            ones(n_particles) / n_particles,
        )
        return self

    def predict_boxed(self, *args, **kwargs):
        """Alias for :meth:`generate_boxed_particles`."""
        return self.generate_boxed_particles(*args, **kwargs)

    def apply_box_constraint(self, *args, **kwargs):
        """Alias for :meth:`generate_boxed_particles`."""
        return self.generate_boxed_particles(*args, **kwargs)

    def reweight_by_box(self, box_lower=None, box_upper=None, likelihood=None):
        """Zero out-of-box particle weights and normalize remaining weights."""
        lower, upper = self._resolve_box(box_lower, box_upper)
        self._validate_box(lower, upper, require_positive_volume=False)
        inside = self._points_inside_box(self.filter_state.d, lower, upper)
        weight_update = where(inside, self.filter_state.w, zeros_like(self.filter_state.w))

        if likelihood is not None:
            if isinstance(likelihood, AbstractManifoldSpecificDistribution):
                likelihood = likelihood.pdf
            likelihood_values = array(likelihood(self.filter_state.d))
            if likelihood_values.shape != self.filter_state.w.shape:
                raise ValueError("likelihood must return one value per particle")
            weight_update = weight_update * likelihood_values

        if bool(sum(weight_update) <= 0):
            raise ValueError("No particle lies inside the requested box")
        new_state = copy.deepcopy(self.filter_state)
        new_state.w = weight_update / sum(weight_update)
        self._filter_state = new_state
        self.resample_if_needed()
        return self

    def update_box_constraint(self, box_lower=None, box_upper=None, likelihood=None):
        """Alias for :meth:`reweight_by_box`."""
        return self.reweight_by_box(box_lower, box_upper, likelihood=likelihood)

    def update_identity_box(
        self,
        measurement_lower,
        measurement_upper=None,
        measurement_noise_bounds=None,
        *,
        likelihood=None,
    ):
        """Update an identity measurement by a bounded measurement box.

        For ``z = x + v`` and noise bounds ``[v_lower, v_upper]``, the implied
        state constraint is ``x in [z_lower - v_upper, z_upper - v_lower]``.
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
        return self.reweight_by_box(z_lower, z_upper, likelihood=likelihood)

    def predict_nonlinear_boxed(
        self,
        f: Callable,
        box_lower=None,
        box_upper=None,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = False,
        box_generator: Callable | None = None,
        boxed_generation_method: str | None = None,
        proposal_distribution=None,
        batch_size: Union[int, int32, int64] | None = None,
        max_sampling_iterations: Union[int, int32, int64] | None = None,
        max_tries_per_particle: Union[int, int32, int64] | None = None,
        gaussian_sigma_scale: float | None = None,
    ):
        """Predict with ordinary particles and then apply boxed generation.

        ``box_generator`` may be supplied to compute the box from the predicted
        point-particle set. It receives the predicted particles and must return
        ``(box_lower, box_upper)``.
        """
        super().predict_nonlinear(
            f,
            noise_distribution=noise_distribution,
            function_is_vectorized=function_is_vectorized,
            shift_instead_of_add=shift_instead_of_add,
        )

        if box_generator is not None:
            box_lower, box_upper = self._coerce_box_result(
                box_generator(self.filter_state.d)
            )

        return self.generate_boxed_particles(
            box_lower,
            box_upper,
            boxed_generation_method=boxed_generation_method,
            proposal_distribution=proposal_distribution,
            batch_size=batch_size,
            max_sampling_iterations=max_sampling_iterations,
            max_tries_per_particle=max_tries_per_particle,
            gaussian_sigma_scale=gaussian_sigma_scale,
        )

    def predict_boxed_nonlinear(self, *args, **kwargs):
        """Alias for :meth:`predict_nonlinear_boxed`."""
        return self.predict_nonlinear_boxed(*args, **kwargs)

    def _resolve_box(self, box_lower, box_upper):
        if box_lower is None and box_upper is None:
            return self.get_box_bounds()
        if box_lower is None and box_upper is not None:
            raise ValueError("box_lower must be supplied when box_upper is supplied")
        lower, upper = self._coerce_interval(box_lower, box_upper, self.dim)
        self._validate_box(lower, upper, require_positive_volume=False)
        return lower, upper

    def _points_inside_box(self, points, lower, upper):
        points = self._coerce_points(points)
        lower = reshape(lower, (1, -1))
        upper = reshape(upper, (1, -1))
        return all(logical_and(points >= lower, points <= upper), axis=1)

    def _sample_uniform_in_box(self, lower, upper, n_particles: int):
        lower = reshape(lower, (1, -1))
        upper = reshape(upper, (1, -1))
        unit_samples = random.uniform(size=(n_particles, self.dim))
        return lower + unit_samples * (upper - lower)

    def _resample_from_inside_box(self, lower, upper, n_particles: int):
        inside = self._points_inside_box(self.filter_state.d, lower, upper)
        inside_weights = where(
            inside,
            self.filter_state.w,
            zeros_like(self.filter_state.w),
        )
        if bool(sum(inside_weights) <= 0):
            raise ValueError("No particle lies inside the requested box")
        proposal = LinearDiracDistribution(
            self.filter_state.d,
            inside_weights / sum(inside_weights),
        )
        return self._coerce_points(proposal.sample(n_particles))

    def _sample_from_proposal_inside_box(
        self,
        proposal_distribution,
        lower,
        upper,
        n_particles: int,
        *,
        batch_size,
        max_sampling_iterations,
        max_tries_per_particle,
    ):
        batch_size = self._resolve_batch_size(batch_size, n_particles)
        max_sampling_iterations = self._resolve_max_sampling_iterations(
            max_sampling_iterations,
            max_tries_per_particle,
        )
        accepted_parts = []
        n_accepted = 0
        for _ in range(max_sampling_iterations):
            candidates = self._draw_proposal_samples(proposal_distribution, batch_size)
            inside = self._points_inside_box(candidates, lower, upper)
            inside_np = _np.asarray(to_numpy(inside), dtype=bool).reshape(-1)
            if inside_np.any():
                candidates_np = _np.asarray(to_numpy(candidates), dtype=float)
                accepted_np = candidates_np[inside_np]
                accepted_parts.append(accepted_np)
                n_accepted += int(accepted_np.shape[0])
                if n_accepted >= n_particles:
                    break

        if n_accepted < n_particles:
            raise ValueError(
                "Could not generate enough in-box particles from the proposal; "
                "increase max_sampling_iterations or use a better proposal."
            )

        return array(_np.vstack(accepted_parts)[:n_particles])

    def _sample_from_inscribed_gaussian(
        self,
        lower,
        upper,
        n_particles: int,
        *,
        batch_size,
        max_sampling_iterations,
        gaussian_sigma_scale,
    ):
        batch_size = self._resolve_batch_size(batch_size, n_particles)
        max_sampling_iterations = self._resolve_max_sampling_iterations(
            max_sampling_iterations,
            None,
        )
        gaussian_sigma_scale = self._validate_positive_float(
            self.gaussian_sigma_scale
            if gaussian_sigma_scale is None
            else gaussian_sigma_scale,
            "gaussian_sigma_scale",
        )
        lower_np = _np.asarray(to_numpy(lower), dtype=float).reshape(-1)
        upper_np = _np.asarray(to_numpy(upper), dtype=float).reshape(-1)
        center = 0.5 * (lower_np + upper_np)
        std = 0.5 * (upper_np - lower_np) / gaussian_sigma_scale

        accepted_parts = []
        n_accepted = 0
        for _ in range(max_sampling_iterations):
            candidates = _np.random.normal(
                loc=center.reshape(1, -1),
                scale=std.reshape(1, -1),
                size=(batch_size, self.dim),
            )
            inside = self._points_inside_box(array(candidates), lower, upper)
            inside_np = _np.asarray(to_numpy(inside), dtype=bool).reshape(-1)
            if inside_np.any():
                accepted_parts.append(candidates[inside_np])
                n_accepted += int(inside_np.sum())
                if n_accepted >= n_particles:
                    break

        if n_accepted < n_particles:
            raise ValueError(
                "Could not generate enough particles from the inscribed Gaussian; "
                "increase max_sampling_iterations or gaussian_sigma_scale."
            )
        return array(_np.vstack(accepted_parts)[:n_particles])

    def _draw_proposal_samples(self, proposal_distribution, n_particles: int):
        if proposal_distribution is None:
            proposal_distribution = self.filter_state
        if callable(proposal_distribution):
            samples = proposal_distribution(n_particles)
        elif hasattr(proposal_distribution, "sample"):
            samples = proposal_distribution.sample(n_particles)
        else:
            raise TypeError(
                "proposal_distribution must be callable or expose a sample(n) method"
            )
        return self._coerce_points(samples)

    def _coerce_points(self, points):
        points = array(points)
        if points.ndim == 0:
            if self.dim != 1:
                raise ValueError("Scalar points are only valid for dim == 1")
            return reshape(points, (1, 1))
        if points.ndim == 1:
            if self.dim == 1:
                return reshape(points, (-1, 1))
            if points.shape[0] == self.dim:
                return reshape(points, (1, self.dim))
            raise ValueError(
                f"Point dimension {points.shape[0]} does not match dim {self.dim}"
            )
        if points.ndim != 2 or points.shape[1] != self.dim:
            raise ValueError(f"points must have shape (n, {self.dim})")
        return points

    @staticmethod
    def _validate_generation_method(method: str):
        method = method.lower()
        if method not in EuclideanBoxedParticleFilter._VALID_GENERATION_METHODS:
            raise ValueError(
                "boxed_generation_method must be one of "
                f"{sorted(EuclideanBoxedParticleFilter._VALID_GENERATION_METHODS)}"
            )
        return method

    @staticmethod
    def _coerce_box_result(result):
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("box_generator must return (box_lower, box_upper)")
        return result

    @staticmethod
    def _validate_positive_int(value, name: str):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _validate_positive_float(value, name: str):
        value = float(value)
        if not _np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def _resolve_batch_size(self, batch_size, n_particles: int):
        if batch_size is None:
            return n_particles
        return self._validate_positive_int(batch_size, "batch_size")

    def _resolve_max_sampling_iterations(
        self,
        max_sampling_iterations,
        max_tries_per_particle,
    ):
        if max_sampling_iterations is not None:
            return self._validate_positive_int(
                max_sampling_iterations, "max_sampling_iterations"
            )
        if max_tries_per_particle is not None:
            max_tries_per_particle = self._validate_positive_int(
                max_tries_per_particle, "max_tries_per_particle"
            )
            return max(1, int(max_tries_per_particle))
        return self.max_sampling_iterations

    @staticmethod
    def _coerce_interval(lower, upper, dim: int):
        if upper is None:
            if isinstance(lower, (tuple, list)) and len(lower) == 2:
                return EuclideanBoxedParticleFilter._coerce_interval(
                    lower[0], lower[1], dim
                )
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
            EuclideanBoxedParticleFilter._coerce_vector(lower, dim, "lower"),
            EuclideanBoxedParticleFilter._coerce_vector(upper, dim, "upper"),
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

    @staticmethod
    def _validate_box(lower, upper, *, require_positive_volume: bool):
        if bool(any(upper < lower)):
            raise ValueError("box_upper must be greater than or equal to box_lower")
        if require_positive_volume and bool(any(upper <= lower)):
            raise ValueError("box_upper must be greater than box_lower")


BoxedParticleFilter = EuclideanBoxedParticleFilter
