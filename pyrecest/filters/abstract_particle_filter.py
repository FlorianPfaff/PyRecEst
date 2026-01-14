import copy
from collections.abc import Callable

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import hstack, ndim, ones_like, random, sum, vmap, vstack
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

from .abstract_filter import AbstractFilter


class AbstractParticleFilter(AbstractFilter):
    def __init__(self, initial_filter_state=None):
        AbstractFilter.__init__(self, initial_filter_state)

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(
            f=lambda x: x,
            noise_distribution=noise_distribution,
            function_is_vectorized=True,
        )

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        assert (
            noise_distribution is None
            or self.filter_state.dim == noise_distribution.dim
        )

        if function_is_vectorized:
            d_f_applied = f(self.filter_state.d)
        else:
            self.filter_state = self.filter_state.apply_function(f, function_is_vectorized=function_is_vectorized)
            d_f_applied = self.filter_state.d

        n_particles = self.filter_state.w.shape[0]
        if noise_distribution is None:
            updated_particles = d_f_applied
        else:
            # Cannot easily use vmap because control flow depends on data for some lower level function
            updated_particles = []
            for i in range(n_particles):
                if not shift_instead_of_add:
                    noise = noise_distribution.sample(
                        1
                    )  # Assuming sample(1) returns a single row of values
                    updated_particles.append(d_f_applied[i] + noise)
                else:
                    noise_curr = noise_distribution.set_mean(d_f_applied[i])
                    updated_particles.append(noise_curr.sample(1))

        if self.filter_state.dim == 1:
            updated_particles = hstack(updated_particles)
        else:
            updated_particles = vstack(updated_particles)

        # Do not use constructor directly because some manifolds,
        # e.g., hypercylinders, need special handling
        self._filter_state.d = updated_particles

    def predict_nonlinear_nonadditive(self, f, samples, weights):
        assert (
            samples.shape[0] == weights.shape[0]
        ), "samples and weights must match in size"

        weights = weights / sum(weights)
        n_particles = self.filter_state.w.shape[0]
        noise_samples = random.choice(samples, n_particles, p=weights)

        batched_apply_f = vmap(f)

        d = batched_apply_f(self.filter_state.d, noise_samples)

        self._filter_state.d = d

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if self._filter_state is None:
            self._filter_state = copy.deepcopy(new_state)
        elif isinstance(new_state, type(self.filter_state)):
            assert (
                self.filter_state.d.shape == new_state.d.shape
            )  # This also ensures the dimension and type stays the same
            self._filter_state = copy.deepcopy(new_state)
        else:
            # Sample if it does not inherit from the previous distribution
            samples = new_state.sample(self.filter_state.w.shape[0])
            assert (
                samples.shape == self.filter_state.d.shape
            )  # This also ensures the dimension and type stays the same
            self._filter_state.d = samples
            self._filter_state.w = (
                ones_like(self.filter_state.w) / self.filter_state.w.shape[0]
            )

    def update_identity(
        self, meas_noise, measurement, shift_instead_of_add: bool = True
    ):
        assert (
            measurement is None
            or measurement.shape == (meas_noise.dim,)
            or meas_noise.dim == 1
            and measurement.shape == ()
        )
        assert ndim(measurement) == 1 or ndim(measurement) == 0 and meas_noise.dim == 1
        if not shift_instead_of_add:
            raise NotImplementedError()

        likelihood = meas_noise.set_mode(measurement).pdf
        self.update_nonlinear_using_likelihood(likelihood)

    def update_nonlinear_using_likelihood(self, likelihood, measurement=None):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            self._filter_state = self.filter_state.reweigh(likelihood)
        else:
            self._filter_state = self.filter_state.reweigh(
                lambda x: likelihood(measurement, x)
            )

        self._filter_state.d = self.filter_state.sample(self.filter_state.w.shape[0])
        self._filter_state.w = (
            1 / self.filter_state.w.shape[0] * ones_like(self.filter_state.w)
        )

    def association_likelihood(self, likelihood: AbstractManifoldSpecificDistribution):
        likelihood_val = sum(likelihood.pdf(self.filter_state.d) * self.filter_state.w)
        return likelihood_val
