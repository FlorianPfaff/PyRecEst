import copy
import inspect
from collections.abc import Callable

import numpy as np
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    hstack,
    ndim,
    ones_like,
    random,
    reshape,
    sum,
    vmap,
    vstack,
)
from pyrecest.diagnostics import ParticleDiagnostics
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

from .abstract_filter import AbstractFilter


def _call_vectorized_sample_next(sample_next, particles, n_particles):
    """Call vectorized sample_next, passing the batch size when supported."""
    owner = getattr(sample_next, "__self__", None)
    if (
        getattr(owner, "function_is_vectorized", False)
        and hasattr(owner, "_sample_next_count_call_mode")
        and getattr(owner, "_sample_next_count_call_mode", None) is None
    ):
        return sample_next(particles)

    try:
        signature = inspect.signature(sample_next)
    except (TypeError, ValueError):
        return sample_next(particles)

    parameters = signature.parameters
    n_parameter = parameters.get("n")
    if n_parameter is not None:
        if n_parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            return sample_next(particles, n_particles)
        return sample_next(particles, n=n_particles)

    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return sample_next(particles, n=n_particles)

    return sample_next(particles)


def _stack_particle_updates(updates, reference_particles):
    """Stack scalar/one-dimensional particle updates without transposing matrices."""
    if ndim(reference_particles) == 1:
        return hstack(updates)
    return vstack(updates)


def _normalize_nonadditive_noise_weights(weights):
    """Validate and stably normalize nonadditive-noise probabilities."""
    host_weights = np.asarray(pyrecest.backend.to_numpy(weights), dtype=float)
    if not np.all(np.isfinite(host_weights)):
        raise ValueError("Noise weights must be finite.")
    if np.any(host_weights < 0.0):
        raise ValueError("Noise weights must be nonnegative.")

    try:
        weight_sum = sum(weights)
        host_weight_sum = float(np.asarray(pyrecest.backend.to_numpy(weight_sum)))
    except FloatingPointError:
        weight_sum = None
        host_weight_sum = float("nan")
    if (
        weight_sum is not None
        and np.isfinite(host_weight_sum)
        and host_weight_sum > 0.0
    ):
        return weights / weight_sum

    # Backend reductions can overflow for large finite weights, while XLA can
    # flush representable float64 subnormals to zero during arithmetic. The raw
    # values are already finite/nonnegative, so recover their ratios after a
    # bounded host-side max scaling and move only ordinary probabilities back.
    if host_weights.size == 0:
        raise ValueError("Noise weights must have positive finite total mass.")
    weight_scale = float(np.max(host_weights))
    if not np.isfinite(weight_scale) or weight_scale <= 0.0:
        raise ValueError("Noise weights must have positive finite total mass.")

    scaled_weights = host_weights / weight_scale
    scaled_weight_sum = float(np.sum(scaled_weights))
    if not np.isfinite(scaled_weight_sum) or scaled_weight_sum <= 0.0:
        raise ValueError("Noise weights must have positive finite total mass.")
    return array(scaled_weights / scaled_weight_sum, dtype=float)


class AbstractParticleFilter(AbstractFilter):
    def __init__(
        self,
        initial_filter_state=None,
        resampling_criterion: Callable | None = None,
    ):
        AbstractFilter.__init__(self, initial_filter_state)
        self.resampling_criterion = resampling_criterion
        self._resampling_count = 0

    @property
    def resampling_criterion(self):
        """Criterion deciding whether to resample after an update.

        ``None`` preserves the historical behavior and always resamples.
        Otherwise, the callable receives the current weighted filter state and
        must return a truthy value if the particle set should be resampled.
        """
        return self._resampling_criterion

    @resampling_criterion.setter
    def resampling_criterion(self, criterion: Callable | None):
        if criterion is not None and not callable(criterion):
            raise TypeError("resampling_criterion must be callable or None")
        self._resampling_criterion = criterion

    def set_resampling_criterion(self, criterion: Callable | None):
        """Set the post-update resampling criterion and return the filter."""
        self.resampling_criterion = criterion
        return self

    def particle_diagnostics(self, *, resampled: bool | None = None):
        """Return standard health diagnostics for the current particle weights."""
        return ParticleDiagnostics.from_weights(
            self.filter_state.w,
            resampled=resampled,
            resampling_count=self._resampling_count,
        )

    def should_resample(self) -> bool:
        """Return whether the current weighted particle set should resample.

        The default criterion, ``None``, always returns ``True`` to retain the
        previous update behavior.
        """
        if self.resampling_criterion is None:
            return True
        return bool(self.resampling_criterion(self.filter_state))

    def resample(self):
        """Manually resample particles according to their current weights.

        The particle locations are sampled with replacement from the current
        weighted particle set, and the resulting weights are reset to uniform.
        """
        self._filter_state.d = self.filter_state.sample(self.filter_state.w.shape[0])
        self._filter_state.w = (
            ones_like(self.filter_state.w) / self.filter_state.w.shape[0]
        )
        self._resampling_count += 1
        return self

    def resample_if_needed(self) -> bool:
        """Resample if the configured criterion requests it.

        Returns
        -------
        bool
            ``True`` if resampling was performed, otherwise ``False``.
        """
        if self.should_resample():
            self.resample()
            return True
        return False

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(
            f=lambda x: x,
            noise_distribution=noise_distribution,
            function_is_vectorized=True,
        )

    def predict_model(self, transition_model):
        """Predict using a reusable particle transition model."""
        if not hasattr(transition_model, "sample_next"):
            raise TypeError(
                "Particle-filter transition models must expose a sample_next callable."
            )

        sample_next = transition_model.sample_next
        function_is_vectorized = getattr(
            transition_model,
            "function_is_vectorized",
            getattr(transition_model, "vectorized", True),
        )
        n_particles = self.filter_state.w.shape[0]

        if function_is_vectorized:
            updated_particles = _call_vectorized_sample_next(
                sample_next, self.filter_state.d, n_particles
            )
        else:
            updated_particles = [
                sample_next(particle) for particle in self.filter_state.d
            ]
            updated_particles = _stack_particle_updates(
                updated_particles, self.filter_state.d
            )

        if updated_particles.shape != self.filter_state.d.shape:
            raise ValueError(
                "sample_next returned particles with shape "
                f"{updated_particles.shape}, expected {self.filter_state.d.shape}."
            )

        self._filter_state.d = updated_particles

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        if (
            noise_distribution is not None
            and self.filter_state.dim != noise_distribution.dim
        ):
            raise ValueError(
                f"Noise distribution dimension {noise_distribution.dim} does not match filter-state dimension {self.filter_state.dim}."
            )

        expected_shape = self.filter_state.d.shape
        if function_is_vectorized:
            d_f_applied = f(self.filter_state.d)
        else:
            transformed_state = self.filter_state.apply_function(
                f, function_is_vectorized=False
            )
            d_f_applied = transformed_state.d

        transformed_shape = getattr(d_f_applied, "shape", None)
        if transformed_shape != expected_shape:
            raise ValueError(
                "Nonlinear transition returned particles with shape "
                f"{transformed_shape}, expected {expected_shape}."
            )

        n_particles = self.filter_state.w.shape[0]
        if noise_distribution is None:
            updated_particles = d_f_applied
        else:
            updated_particles = []
            for i in range(n_particles):
                if not shift_instead_of_add:
                    noise = noise_distribution.sample(1)
                    updated_particles.append(d_f_applied[i] + noise)
                else:
                    noise_curr = copy.deepcopy(noise_distribution)
                    shifted_noise = noise_curr.set_mean(d_f_applied[i])
                    if shifted_noise is not None:
                        noise_curr = shifted_noise
                    updated_particles.append(noise_curr.sample(1))

            updated_particles = _stack_particle_updates(
                updated_particles, self.filter_state.d
            )

        updated_shape = getattr(updated_particles, "shape", None)
        if updated_shape != expected_shape:
            raise ValueError(
                "Nonlinear prediction produced particles with shape "
                f"{updated_shape}, expected {expected_shape}."
            )

        self._filter_state.d = updated_particles

    def predict_nonlinear_nonadditive(self, f, samples, weights):
        samples = array(samples)
        weights = array(weights, dtype=float)
        if samples.ndim == 0:
            raise ValueError("samples must contain a leading sample dimension")
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if samples.shape[0] != weights.shape[0]:
            raise ValueError("samples and weights must match in size")

        weights = _normalize_nonadditive_noise_weights(weights)
        n_particles = self.filter_state.w.shape[0]
        noise_samples = random.choice(samples, n_particles, p=weights)

        batched_apply_f = vmap(f)

        expected_shape = self.filter_state.d.shape
        d = batched_apply_f(self.filter_state.d, noise_samples)
        transformed_shape = getattr(d, "shape", None)
        if transformed_shape != expected_shape:
            raise ValueError(
                "Nonadditive transition returned particles with shape "
                f"{transformed_shape}, expected {expected_shape}."
            )

        self._filter_state.d = d

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if self._filter_state is None:
            self._filter_state = copy.deepcopy(new_state)
        elif isinstance(new_state, type(self.filter_state)):
            if self.filter_state.d.shape != new_state.d.shape:
                raise ValueError(
                    "The shape of new state does not match with the existing state."
                )
            self._filter_state = copy.deepcopy(new_state)
        else:
            samples = new_state.sample(self.filter_state.w.shape[0])
            if samples.shape != self.filter_state.d.shape:
                raise ValueError(
                    "Samples from new state have shape "
                    f"{samples.shape}, expected {self.filter_state.d.shape}."
                )
            self._filter_state.d = samples
            self._filter_state.w = (
                ones_like(self.filter_state.w) / self.filter_state.w.shape[0]
            )

    def update_model(
        self, measurement_model, measurement=None, *, return_diagnostics=False
    ):
        """Update using a reusable particle measurement model."""
        if not hasattr(measurement_model, "likelihood"):
            raise TypeError(
                "Particle-filter measurement models must expose a likelihood callable."
            )

        return self.update_nonlinear_using_likelihood(
            measurement_model.likelihood,
            measurement=measurement,
            return_diagnostics=return_diagnostics,
        )

    def update_identity(
        self,
        meas_noise,
        measurement,
        shift_instead_of_add: bool = True,
        *,
        return_diagnostics=False,
    ):
        if measurement is None:
            raise ValueError("measurement must not be None.")

        measurement = array(measurement)
        if meas_noise.dim == 1 and ndim(measurement) == 0:
            measurement = reshape(measurement, (1,))
        if ndim(measurement) != 1 or measurement.shape[0] != meas_noise.dim:
            raise ValueError(
                f"measurement must have shape ({meas_noise.dim},), got {measurement.shape}."
            )
        if not shift_instead_of_add:
            raise NotImplementedError()

        likelihood = meas_noise.set_mode(measurement).pdf
        return self.update_nonlinear_using_likelihood(
            likelihood, return_diagnostics=return_diagnostics
        )

    def update_nonlinear_using_likelihood(
        self, likelihood, measurement=None, *, return_diagnostics=False
    ):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            self._filter_state = self.filter_state.reweigh(likelihood)
        else:
            self._filter_state = self.filter_state.reweigh(
                lambda x: likelihood(measurement, x)
            )

        diagnostics = self.particle_diagnostics(resampled=False)
        resampled = self.resample_if_needed()
        diagnostics.resampled = resampled
        diagnostics.resampling_count = self._resampling_count
        if return_diagnostics:
            return diagnostics
        return None

    def association_likelihood(self, likelihood: AbstractManifoldSpecificDistribution):
        likelihood_val = sum(likelihood.pdf(self.filter_state.d) * self.filter_state.w)
        return likelihood_val
