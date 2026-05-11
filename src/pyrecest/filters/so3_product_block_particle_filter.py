"""Block particle filter for Cartesian products of SO(3)."""

# pylint: disable=no-name-in-module,no-member,too-many-positional-arguments
from pyrecest.backend import stack
from pyrecest.distributions import SO3DiracDistribution

from .block_particle_filter import BlockParticleFilter
from .so3_product_particle_filter import SO3ProductParticleFilter


class SO3ProductBlockParticleFilter(BlockParticleFilter, SO3ProductParticleFilter):
    """Particle filter for ``SO(3)^K`` with independent block weights.

    The filter approximates the product-state posterior by a product over a
    user-supplied partition of the ``K`` SO(3) components. Each block keeps its
    own particle weights and may be resampled independently. This preserves
    correlations inside each block while avoiding the degeneracy of a single
    global weight vector in high-dimensional product spaces.

    The inherited ``weights`` property remains available as a normalized average
    of the block weights for API compatibility. Point estimates, block ESS, and
    block resampling use the block-specific weights.
    """

    def __init__(
        self,
        n_particles: int,
        num_rotations: int,
        partition=None,
        initial_particles=None,
        weights=None,
        block_weights=None,
    ) -> None:
        SO3ProductParticleFilter.__init__(
            self,
            n_particles=n_particles,
            num_rotations=num_rotations,
            initial_particles=initial_particles,
            weights=weights,
        )
        self._initialize_block_particle_filter(
            n_components=self.num_rotations,
            partition=partition,
            block_weights=block_weights,
            weights=weights,
        )

    def mean(self):
        """Return the component-wise chordal mean using block-specific weights."""
        means = [
            SO3DiracDistribution(
                self.particles[:, component_idx, :],
                self.component_weights(component_idx),
            ).mean()
            for component_idx in range(self.num_rotations)
        ]
        return stack(means, axis=0)

    def get_point_estimate(self):
        """Return the component-wise SO(3) mean."""
        return self.mean()

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
        """Update block weights with masked component geodesic log-likelihoods."""
        return self.update_with_component_log_likelihoods(
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
        """Update with masked geodesic likelihoods per partition block.

        This preserves the existing likelihood-space API while delegating to the
        log-likelihood implementation for numerical stability.
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
