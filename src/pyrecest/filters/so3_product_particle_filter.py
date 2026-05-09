"""Particle filter for Cartesian products of SO(3)."""

import math
from collections.abc import Callable

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all,
    any,
    array,
    diag,
    exp,
    isfinite,
    isnan,
    log,
    max,
    maximum,
    ndim,
    ones,
    random,
    reshape,
    sqrt,
    stack,
    sum,
    to_numpy,
    where,
    zeros,
)
from pyrecest.distributions import SO3DiracDistribution
from pyrecest.distributions._so3_helpers import geodesic_distance, normalize_quaternions
from pyrecest.distributions.so3_tangent_gaussian_distribution import (
    SO3TangentGaussianDistribution,
)

from .hyperhemisphere_cart_prod_particle_filter import (
    HyperhemisphereCartProdParticleFilter,
)


class SO3ProductParticleFilter(HyperhemisphereCartProdParticleFilter):
    """Particle filter for states on ``SO(3)^K``.

    Particles are exposed as scalar-last unit quaternions with shape
    ``(n_particles, num_rotations, 4)``. Internally, the filter stores them in
    the generic hyperhemisphere Cartesian-product particle filter with
    ``dim_hemisphere=3``.
    """

    def __init__(
        self,
        n_particles: int,
        num_rotations: int,
        initial_particles=None,
        weights=None,
    ) -> None:
        if n_particles <= 0:
            raise ValueError("n_particles must be positive.")
        if num_rotations <= 0:
            raise ValueError("num_rotations must be positive.")

        self.num_rotations = int(num_rotations)
        super().__init__(
            n_particles=n_particles,
            dim_hemisphere=3,
            n_hemispheres=self.num_rotations,
        )

        particles = self._identity_particles(n_particles, self.num_rotations)
        if initial_particles is not None:
            particles = self._as_particle_array(initial_particles, self.num_rotations)
            if particles.shape[0] != n_particles:
                raise ValueError("initial_particles must contain n_particles entries.")

        if weights is not None:
            weights = self._normalize_weights(weights)
        self.set_particles(particles, weights=weights)

    @staticmethod
    def _identity_particles(n_particles, num_rotations):
        return array(
            [
                [[0.0, 0.0, 0.0, 1.0] for _ in range(num_rotations)]
                for _ in range(n_particles)
            ],
            dtype=float,
        )

    @staticmethod
    def _as_particle_array(particles, num_rotations):
        particles = array(particles, dtype=float)

        if ndim(particles) == 1:
            if particles.shape[0] != 4 * num_rotations:
                raise ValueError("A flat SO(3)^K particle must have length 4K.")
            particles = reshape(particles, (1, num_rotations, 4))
        elif ndim(particles) == 2:
            if particles.shape == (num_rotations, 4):
                particles = reshape(particles, (1, num_rotations, 4))
            elif particles.shape[-1] == 4 * num_rotations:
                particles = reshape(particles, (particles.shape[0], num_rotations, 4))
            else:
                raise ValueError(
                    "SO(3)^K particles must have shape (K, 4), (n, 4K), or "
                    "(n, K, 4)."
                )
        elif ndim(particles) == 3:
            if particles.shape[1:] != (num_rotations, 4):
                raise ValueError("SO(3)^K particles must have shape (n, K, 4).")
        else:
            raise ValueError(
                "SO(3)^K particles must have shape (4K,), (K, 4), "
                "(n, 4K), or (n, K, 4)."
            )

        normalized = normalize_quaternions(reshape(particles, (-1, 4)))
        return reshape(normalized, particles.shape)

    @staticmethod
    def _as_product_point(rotation, num_rotations):
        particles = SO3ProductParticleFilter._as_particle_array(rotation, num_rotations)
        if particles.shape[0] != 1:
            raise ValueError("Expected a single SO(3)^K product point.")
        return particles[0]

    @staticmethod
    def _flatten_particles(particles):
        return reshape(particles, (particles.shape[0], 4 * particles.shape[1]))

    @staticmethod
    def _normalize_weights(weights):
        weights = array(weights, dtype=float)
        if ndim(weights) != 1:
            weights = reshape(weights, (-1,))
        if not all(weights >= 0.0):
            raise ValueError("Particle weights must be nonnegative.")
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("At least one particle weight must be positive.")
        return weights / weight_sum

    @staticmethod
    def _normalize_log_weights(log_weights):
        """Normalize possibly unscaled log weights with log-sum-exp stability."""
        log_weights = array(log_weights, dtype=float)
        if ndim(log_weights) != 1:
            log_weights = reshape(log_weights, (-1,))
        if any(isnan(log_weights)):
            raise ValueError("log weights must not contain NaN.")
        max_log_weight = max(log_weights)
        if not isfinite(max_log_weight):
            raise ValueError(
                "At least one log weight must be finite and no log weight may be +inf."
            )
        return SO3ProductParticleFilter._normalize_weights(
            exp(log_weights - max_log_weight)
        )

    @staticmethod
    def _validate_probability(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be a finite probability in [0, 1].")
        return value

    @staticmethod
    def _as_component_mask(mask, num_rotations):
        if mask is None:
            return ones(num_rotations)
        mask = array(mask, dtype=float)
        if mask.shape != (num_rotations,):
            raise ValueError("mask must have shape (num_rotations,).")
        if not all(isfinite(mask)):
            raise ValueError("mask values must be finite.")
        if not all(mask >= 0.0):
            raise ValueError("mask values must be nonnegative.")
        return mask

    @staticmethod
    def _as_component_confidence(confidence, num_rotations):
        if confidence is None:
            return ones(num_rotations)
        confidence = array(confidence, dtype=float)
        if confidence.shape != (num_rotations,):
            raise ValueError("confidence must have shape (num_rotations,).")
        if not all(isfinite(confidence)):
            raise ValueError("confidence values must be finite.")
        if not all(confidence >= 0.0) or not all(confidence <= 1.0):
            raise ValueError("confidence values must be in [0, 1].")
        return confidence

    @staticmethod
    def _as_component_noise_std(noise_std, component_noise_std, num_rotations):
        if component_noise_std is None:
            if noise_std is None:
                raise ValueError("noise_std or component_noise_std must be provided.")
            sigma = array(noise_std, dtype=float)
        else:
            sigma = array(component_noise_std, dtype=float)

        if ndim(sigma) == 0:
            if not isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    "noise standard deviations must be positive and finite."
                )
            return ones(num_rotations) * sigma
        if sigma.shape != (num_rotations,):
            raise ValueError("component_noise_std must have shape (num_rotations,).")
        if not all(isfinite(sigma)):
            raise ValueError("noise standard deviations must be finite.")
        if not all(sigma > 0.0):
            raise ValueError("noise standard deviations must be positive.")
        return sigma

    @staticmethod
    def confidence_to_noise_std(
        confidence,
        noise_std,
        max_noise_std,
        *,
        confidence_exponent: float = 1.0,
        mask=None,
    ):
        """Map detector confidence values in ``[0, 1]`` to SO(3) noise scales.

        The mapping is
        ``sigma(c)^2 = noise_std^2 + (1 - c)^confidence_exponent *``
        ``(max_noise_std^2 - noise_std^2)``.
        """
        confidence = array(confidence, dtype=float)
        if ndim(confidence) != 1:
            raise ValueError("confidence must be a one-dimensional array.")
        num_rotations = confidence.shape[0]
        confidence = SO3ProductParticleFilter._as_component_confidence(
            confidence, num_rotations
        )
        if noise_std is None or max_noise_std is None:
            raise ValueError("noise_std and max_noise_std must be provided.")
        min_sigma = float(noise_std)
        max_sigma = float(max_noise_std)
        exponent = float(confidence_exponent)
        if min_sigma <= 0.0 or not math.isfinite(min_sigma):
            raise ValueError("noise_std must be positive and finite.")
        if max_sigma <= 0.0 or not math.isfinite(max_sigma):
            raise ValueError("max_noise_std must be positive and finite.")
        if max_sigma < min_sigma:
            raise ValueError(
                "max_noise_std must be greater than or equal to noise_std."
            )
        if exponent <= 0.0 or not math.isfinite(exponent):
            raise ValueError("confidence_exponent must be positive and finite.")

        variance = min_sigma * min_sigma + (1.0 - confidence) ** exponent * (
            max_sigma * max_sigma - min_sigma * min_sigma
        )
        sigma = sqrt(maximum(variance, 1e-16))
        if mask is not None:
            mask = SO3ProductParticleFilter._as_component_mask(mask, num_rotations)
            sigma = where(mask > 0.0, sigma, max_sigma)
        return sigma

    @staticmethod
    def _as_tangent_array(tangent_vectors, n_particles, num_rotations):
        tangent_vectors = array(tangent_vectors, dtype=float)

        if ndim(tangent_vectors) == 1:
            if tangent_vectors.shape[0] != 3 * num_rotations:
                raise ValueError("A flat SO(3)^K tangent vector must have length 3K.")
            tangent_vectors = reshape(tangent_vectors, (1, num_rotations, 3))
        elif ndim(tangent_vectors) == 2:
            if tangent_vectors.shape == (num_rotations, 3):
                tangent_vectors = reshape(tangent_vectors, (1, num_rotations, 3))
            elif tangent_vectors.shape[-1] == 3 * num_rotations:
                tangent_vectors = reshape(
                    tangent_vectors, (tangent_vectors.shape[0], num_rotations, 3)
                )
            else:
                raise ValueError(
                    "SO(3)^K tangent vectors must have shape (K, 3), "
                    "(n, 3K), or (n, K, 3)."
                )
        elif ndim(tangent_vectors) == 3:
            if tangent_vectors.shape[1:] != (num_rotations, 3):
                raise ValueError("SO(3)^K tangent vectors must have shape (n, K, 3).")
        else:
            raise ValueError(
                "SO(3)^K tangent vectors must have shape (3K,), (K, 3), "
                "(n, 3K), or (n, K, 3)."
            )

        if tangent_vectors.shape[0] == 1 and n_particles != 1:
            tangent_vectors = stack(
                [tangent_vectors[0] for _ in range(n_particles)], axis=0
            )
        if tangent_vectors.shape[0] != n_particles:
            raise ValueError("Tangent vector batch size must match particle count.")

        return tangent_vectors

    @staticmethod
    def _sample_tangent_noise(covariance, n_particles, num_rotations):
        if covariance is None:
            return zeros((n_particles, num_rotations, 3))

        covariance = array(covariance, dtype=float)
        if ndim(covariance) == 2:
            if covariance.shape == (3, 3):
                components = [
                    random.multivariate_normal(
                        mean=zeros(3), cov=covariance, size=n_particles
                    )
                    for _ in range(num_rotations)
                ]
                return stack(components, axis=1)
            if covariance.shape == (3 * num_rotations, 3 * num_rotations):
                samples = random.multivariate_normal(
                    mean=zeros(3 * num_rotations), cov=covariance, size=n_particles
                )
                return reshape(samples, (n_particles, num_rotations, 3))
        elif ndim(covariance) == 3 and covariance.shape == (num_rotations, 3, 3):
            components = [
                random.multivariate_normal(
                    mean=zeros(3), cov=covariance[i], size=n_particles
                )
                for i in range(num_rotations)
            ]
            return stack(components, axis=1)

        raise ValueError(
            "tangent_noise_covariance must have shape (3, 3), (K, 3, 3), "
            "or (3K, 3K)."
        )

    @property
    def n_particles(self):
        return self.filter_state.d.shape[0]

    @property
    def particles(self):
        """Return particles with shape ``(n_particles, num_rotations, 4)``."""
        return reshape(self.filter_state.d, (self.n_particles, self.num_rotations, 4))

    @property
    def weights(self):
        """Return normalized particle weights."""
        return self.filter_state.w

    def set_particles(self, particles, weights=None):
        """Replace particles and optionally weights."""
        particles = self._as_particle_array(particles, self.num_rotations)
        if particles.shape[0] != self.n_particles:
            raise ValueError("New particles must match the existing particle count.")
        self.filter_state.d = self._flatten_particles(particles)
        if weights is not None:
            weights = self._normalize_weights(weights)
            if weights.shape[0] != self.n_particles:
                raise ValueError("weights must match the particle count.")
            self.filter_state.w = weights

    def mean(self):
        """Return the component-wise chordal mean product rotation."""
        means = [
            SO3DiracDistribution(self.particles[:, i, :], self.weights).mean()
            for i in range(self.num_rotations)
        ]
        return stack(means, axis=0)

    def mode(self):
        """Return the highest-weight product particle."""
        return reshape(self.filter_state.mode(), (self.num_rotations, 4))

    def get_point_estimate(self):
        """Return the component-wise SO(3) mean."""
        return self.mean()

    def effective_sample_size(self):
        """Return the particle effective sample size."""
        return 1.0 / sum(self.weights**2)

    def resample_systematic(self):
        """Systematically resample particles and reset weights to uniform."""
        weights = self._normalize_weights(self.weights)
        weights_list = [float(weight) for weight in to_numpy(weights).reshape(-1)]
        n_particles = len(weights_list)
        start = float(to_numpy(random.rand(1)).reshape(-1)[0]) / n_particles
        positions = [start + i / n_particles for i in range(n_particles)]

        indices = []
        cumulative_weight = weights_list[0]
        source_index = 0
        for position in positions:
            while position > cumulative_weight and source_index < n_particles - 1:
                source_index += 1
                cumulative_weight += weights_list[source_index]
            indices.append(source_index)

        index_array = array(indices)
        self.filter_state.d = self.filter_state.d[index_array]
        self.filter_state.w = ones(n_particles) / n_particles
        return index_array

    def _apply_tangent_delta(self, particles, tangent_delta):
        tangent_delta = self._as_tangent_array(
            tangent_delta, particles.shape[0], self.num_rotations
        )
        updated_components = [
            SO3TangentGaussianDistribution.exp_map(
                tangent_delta[:, i, :], base=particles[:, i, :]
            )
            for i in range(self.num_rotations)
        ]
        return stack(updated_components, axis=1)

    def predict_with_tangent_delta(self, tangent_delta, tangent_noise_covariance=None):
        """Apply tangent-space deltas and optional tangent Gaussian noise."""
        tangent_delta = self._as_tangent_array(
            tangent_delta, self.n_particles, self.num_rotations
        )
        tangent_noise = self._sample_tangent_noise(
            tangent_noise_covariance, self.n_particles, self.num_rotations
        )
        self.set_particles(
            self._apply_tangent_delta(self.particles, tangent_delta + tangent_noise)
        )

    def predict_identity(self, noise_distribution=None):
        """Predict with identity dynamics and optional tangent Gaussian noise."""
        self.predict_with_tangent_delta(
            zeros((self.num_rotations, 3)),
            tangent_noise_covariance=noise_distribution,
        )

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        """Apply a nonlinear transition on product particles."""
        if not shift_instead_of_add:
            raise NotImplementedError("Only tangent-space noise is supported.")
        if function_is_vectorized:
            predicted_particles = f(self.particles)
        else:
            predicted_particles = stack(
                [f(self.particles[i]) for i in range(self.n_particles)], axis=0
            )
        predicted_particles = self._as_particle_array(
            predicted_particles, self.num_rotations
        )
        tangent_noise = self._sample_tangent_noise(
            noise_distribution, self.n_particles, self.num_rotations
        )
        self.set_particles(
            self._apply_tangent_delta(predicted_particles, tangent_noise)
        )

    def update_with_likelihood(
        self,
        likelihood: Callable,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update weights from a likelihood evaluated on product particles."""
        if measurement is None:
            likelihood_values = likelihood(self.particles)
        else:
            likelihood_values = likelihood(measurement, self.particles)
        likelihood_values = array(likelihood_values, dtype=float)
        if likelihood_values.shape != self.weights.shape:
            raise ValueError("likelihood must return one value per particle.")
        if not all(likelihood_values >= 0.0):
            raise ValueError("likelihood values must be nonnegative.")

        return self.update_with_log_likelihood(
            log(likelihood_values),
            resample=resample,
            ess_threshold=ess_threshold,
        )

    def update_with_log_likelihood(
        self,
        log_likelihood: Callable,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update weights from log-likelihoods evaluated on product particles."""
        if callable(log_likelihood):
            if measurement is None:
                log_likelihood_values = log_likelihood(self.particles)
            else:
                log_likelihood_values = log_likelihood(measurement, self.particles)
        else:
            log_likelihood_values = log_likelihood
        log_likelihood_values = array(log_likelihood_values, dtype=float)
        if log_likelihood_values.shape != self.weights.shape:
            raise ValueError("log_likelihood must return one value per particle.")

        self.filter_state.w = self._normalize_log_weights(
            log(self.weights) + log_likelihood_values
        )
        ess = self.effective_sample_size()
        threshold = self.n_particles / 2.0 if ess_threshold is None else ess_threshold
        if resample and ess < threshold:
            self.resample_systematic()
        return ess

    def component_geodesic_log_likelihood(
        self,
        measurement,
        noise_std=None,
        *,
        component_noise_std=None,
        mask=None,
        confidence=None,
        max_noise_std=None,
        confidence_exponent: float = 1.0,
        outlier_prob: float = 0.0,
    ):
        """Return per-component masked geodesic log-likelihoods.

        The returned array has shape ``(n_particles, num_rotations)``.  Masks
        deactivate components, confidence values in ``[0, 1]`` scale each
        component's log-likelihood contribution, ``component_noise_std`` supplies
        heteroskedastic per-component noise, and ``outlier_prob`` adds an
        unnormalized constant likelihood floor for robustness.
        """
        if max_noise_std is not None and component_noise_std is not None:
            raise ValueError(
                "component_noise_std and max_noise_std are mutually exclusive."
            )

        measurement = self._as_product_point(measurement, self.num_rotations)
        mask = self._as_component_mask(mask, self.num_rotations)
        confidence_weights = self._as_component_confidence(
            confidence, self.num_rotations
        )
        component_weights = mask * confidence_weights

        if max_noise_std is not None:
            if confidence is None:
                raise ValueError(
                    "confidence must be provided when max_noise_std is used."
                )
            component_noise_std = self.confidence_to_noise_std(
                confidence,
                noise_std,
                max_noise_std,
                confidence_exponent=confidence_exponent,
                mask=mask,
            )
        sigma = self._as_component_noise_std(
            noise_std, component_noise_std, self.num_rotations
        )
        eps = self._validate_probability("outlier_prob", outlier_prob)

        distances = stack(
            [
                geodesic_distance(self.particles[:, i, :], measurement[i, :])
                for i in range(self.num_rotations)
            ],
            axis=1,
        )
        clean_log_likelihood = -0.5 * (distances / sigma) ** 2
        if eps > 0.0:
            clean_probability = eps if eps < 1.0 - 1e-12 else 1.0 - 1e-12
            outlier_probability = eps if eps > 1e-12 else 1e-12
            clean_term = log(1.0 - clean_probability) + clean_log_likelihood
            outlier_term = log(outlier_probability)
            normalizer = maximum(clean_term, outlier_term)
            clean_log_likelihood = normalizer + log(
                exp(clean_term - normalizer) + exp(outlier_term - normalizer)
            )
        return component_weights * clean_log_likelihood

    def geodesic_log_likelihood(
        self,
        measurement,
        noise_std=None,
        *,
        component_noise_std=None,
        mask=None,
        confidence=None,
        max_noise_std=None,
        confidence_exponent: float = 1.0,
        outlier_prob: float = 0.0,
    ):
        """Return one masked geodesic log-likelihood per product particle."""
        return sum(
            self.component_geodesic_log_likelihood(
                measurement,
                noise_std,
                component_noise_std=component_noise_std,
                mask=mask,
                confidence=confidence,
                max_noise_std=max_noise_std,
                confidence_exponent=confidence_exponent,
                outlier_prob=outlier_prob,
            ),
            axis=1,
        )

    def update_with_geodesic_log_likelihood(
        self,
        measurement,
        noise_std=None,
        *,
        component_noise_std=None,
        mask=None,
        confidence=None,
        max_noise_std=None,
        confidence_exponent: float = 1.0,
        outlier_prob: float = 0.0,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update with a masked, confidence-aware geodesic log-likelihood."""
        return self.update_with_log_likelihood(
            self.geodesic_log_likelihood(
                measurement,
                noise_std,
                component_noise_std=component_noise_std,
                mask=mask,
                confidence=confidence,
                max_noise_std=max_noise_std,
                confidence_exponent=confidence_exponent,
                outlier_prob=outlier_prob,
            ),
            resample=resample,
            ess_threshold=ess_threshold,
        )

    def update_with_geodesic_likelihood(
        self,
        measurement,
        noise_std,
        *,
        component_noise_std=None,
        mask=None,
        confidence=None,
        max_noise_std=None,
        confidence_exponent: float = 1.0,
        outlier_prob: float = 0.0,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update with a masked geodesic likelihood on SO(3)^K.

        This method preserves the existing likelihood-space API and delegates to
        the log-likelihood implementation for numerical stability.
        """
        return self.update_with_geodesic_log_likelihood(
            measurement,
            noise_std,
            component_noise_std=component_noise_std,
            mask=mask,
            confidence=confidence,
            max_noise_std=max_noise_std,
            confidence_exponent=confidence_exponent,
            outlier_prob=outlier_prob,
            resample=resample,
            ess_threshold=ess_threshold,
        )

    @staticmethod
    def from_covariance_diagonal(
        n_particles,
        mean,
        covariance_diagonal,
    ):
        """Create a filter by sampling tangent noise around ``mean``."""
        mean = SO3ProductParticleFilter._as_product_point(
            mean, num_rotations=len(covariance_diagonal) // 3
        )
        num_rotations = mean.shape[0]
        covariance = diag(array(covariance_diagonal, dtype=float))
        tangent_noise = SO3ProductParticleFilter._sample_tangent_noise(
            covariance, n_particles, num_rotations
        )
        initial_filter = SO3ProductParticleFilter(n_particles, num_rotations)
        initial_filter.set_particles(stack([mean for _ in range(n_particles)], axis=0))
        initial_filter.predict_with_tangent_delta(tangent_noise)
        return initial_filter
