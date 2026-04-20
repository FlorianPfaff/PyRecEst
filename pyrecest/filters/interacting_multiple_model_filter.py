# pylint: disable=no-name-in-module,no-member
"""Interacting Multiple Model (IMM) filter."""

import copy
import warnings
from collections.abc import Sequence

import pyrecest.backend
from pyrecest.backend import (
    allclose,
    argmax,
    array,
    asarray,
    diag,
    empty,
    exp,
    eye,
    full,
    isclose,
    isfinite,
    linalg,
    log,
    ones,
    outer,
    pi,
    stack,
    zeros,
    zeros_like,
)
from pyrecest.distributions import GaussianDistribution, GaussianMixture
from scipy.special import logsumexp

from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin


class InteractingMultipleModelFilter(AbstractFilter, EuclideanFilterMixin):
    """Linear-Gaussian interacting multiple model (IMM) filter.

    The filter is built from a bank of single-model filters whose states are assumed
    to be representable by :class:`~pyrecest.distributions.GaussianDistribution`.
    Each subfilter is expected to expose ``filter_state`` together with the methods
    ``predict_identity`` / ``predict_linear`` / ``update_identity`` /
    ``update_linear``. Nonlinear prediction and update are also supported when the
    corresponding subfilter methods exist, but nonlinear updates require externally
    supplied model likelihoods because PyRecEst's current nonlinear filters do not
    yet expose a uniform predictive-likelihood interface.

    The transition matrix is interpreted row-wise,
    ``transition_matrix[i, j] = p(m_k=j | m_{k-1}=i)``.
    """

    def __init__(self, filter_bank, transition_matrix, mode_probabilities=None):
        if pyrecest.backend.__backend_name__ != "numpy":
            warnings.warn(
                "InteractingMultipleModelFilter is currently only tested on the "
                "numpy backend.",
                UserWarning,
            )

        AbstractFilter.__init__(self, None)
        self.filter_bank = [copy.deepcopy(curr_filter) for curr_filter in filter_bank]
        self._validate_filter_bank()

        self.transition_matrix = self._prepare_transition_matrix(
            transition_matrix, self.n_models
        )
        self.mode_probabilities = self._prepare_mode_probabilities(
            mode_probabilities, self.n_models
        )

        self.latest_mixing_probabilities = None
        self.latest_model_likelihoods = None
        self.latest_log_model_likelihoods = None

    @property
    def n_models(self) -> int:
        """Number of interacting models."""
        return len(self.filter_bank)

    @property
    def dim(self) -> int:
        """State dimension."""
        return self._as_gaussian(self.filter_bank[0].filter_state).dim

    @property
    def model_probabilities(self):
        """Alias for :attr:`mode_probabilities`."""
        return self.mode_probabilities

    @model_probabilities.setter
    def model_probabilities(self, new_probabilities):
        self.mode_probabilities = self._prepare_mode_probabilities(
            new_probabilities, self.n_models
        )

    @property
    def filter_state(self):
        """Current Gaussian-mixture state of the IMM.

        Returns a Gaussian mixture whose components are the current subfilter states
        and whose weights are the current mode probabilities. Components with zero
        probability are omitted to avoid downstream issues with zero-weight mixtures.
        """
        active_indices = [
            i for i, prob in enumerate(self.mode_probabilities) if float(prob) > 0.0
        ]
        if not active_indices:
            raise ValueError(
                "At least one model probability must be strictly positive."
            )

        active_states = [
            self._as_gaussian(self.filter_bank[i].filter_state) for i in active_indices
        ]
        active_weights = self.mode_probabilities[active_indices]
        active_weights = active_weights / active_weights.sum()
        return GaussianMixture(active_states, active_weights)

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, GaussianMixture):
            if len(new_state.dists) != self.n_models:
                raise ValueError(
                    "GaussianMixture must have one component per model in the IMM."
                )
            for curr_filter, curr_state in zip(self.filter_bank, new_state.dists):
                curr_filter.filter_state = curr_state
            self.mode_probabilities = self._prepare_mode_probabilities(
                new_state.w, self.n_models
            )
            return

        if isinstance(new_state, Sequence):
            if len(new_state) != self.n_models:
                raise ValueError(
                    "Sequences assigned to filter_state must have one entry per "
                    "model in the IMM."
                )
            if all(hasattr(curr_state, "filter_state") for curr_state in new_state):
                self.filter_bank = [
                    copy.deepcopy(curr_filter) for curr_filter in new_state
                ]
                self._validate_filter_bank()
                return
            if all(
                isinstance(curr_state, GaussianDistribution) for curr_state in new_state
            ):
                for curr_filter, curr_state in zip(self.filter_bank, new_state):
                    curr_filter.filter_state = curr_state
                return

        raise ValueError(
            "new_state must be a GaussianMixture, a sequence of filters, or a "
            "sequence of GaussianDistribution instances."
        )

    @property
    def combined_filter_state(self) -> GaussianDistribution:
        """Moment-matched single-Gaussian approximation of the IMM state."""
        curr_states = [
            self._as_gaussian(curr_filter.filter_state)
            for curr_filter in self.filter_bank
        ]
        return self._moment_match_gaussians(curr_states, self.mode_probabilities)

    @property
    def most_likely_model_index(self) -> int:
        """Index of the most likely current model."""
        return int(argmax(self.mode_probabilities))

    def interact(self):
        """Perform the IMM interaction (mixing) step.

        The current model probabilities are propagated through the transition matrix,
        and a mixed Gaussian prior is computed for each destination model.
        """
        previous_probabilities = asarray(self.mode_probabilities, dtype=float).reshape(
            -1
        )
        predicted_probabilities = previous_probabilities @ self.transition_matrix
        predicted_probabilities = self._prepare_mode_probabilities(
            predicted_probabilities, self.n_models
        )

        previous_states = [
            self._as_gaussian(curr_filter.filter_state)
            for curr_filter in self.filter_bank
        ]
        mixing_probabilities = zeros((self.n_models, self.n_models))

        for curr_model in range(self.n_models):
            curr_normalizer = predicted_probabilities[curr_model]
            if curr_normalizer > 0.0:
                curr_weights = (
                    previous_probabilities
                    * self.transition_matrix[:, curr_model]
                    / curr_normalizer
                )
                mixing_probabilities[:, curr_model] = curr_weights
                self.filter_bank[curr_model].filter_state = (
                    self._moment_match_gaussians(previous_states, curr_weights)
                )
            else:
                mixing_probabilities[curr_model, curr_model] = 1.0
                self.filter_bank[curr_model].filter_state = previous_states[curr_model]

        self.mode_probabilities = predicted_probabilities
        self.latest_mixing_probabilities = mixing_probabilities
        return mixing_probabilities

    def predict_identity(self, sys_noise_covs, sys_inputs=None):
        """Predict each model with an identity system model.

        ``sys_noise_covs`` and ``sys_inputs`` can either be shared across all models
        or be provided as lists/tuples with one entry per model.
        """
        self.interact()
        sys_noise_covs = self._broadcast_model_argument(
            sys_noise_covs, "sys_noise_covs"
        )
        sys_inputs = self._broadcast_model_argument(sys_inputs, "sys_inputs")

        for curr_filter, curr_noise_cov, curr_input in zip(
            self.filter_bank, sys_noise_covs, sys_inputs
        ):
            if curr_input is None:
                curr_filter.predict_identity(curr_noise_cov)
            else:
                curr_filter.predict_identity(curr_noise_cov, curr_input)

    def predict_linear(self, system_matrices, sys_noise_covs, sys_inputs=None):
        """Predict each model with a linear system model.

        ``system_matrices``, ``sys_noise_covs``, and ``sys_inputs`` can either be
        shared across all models or be provided as lists/tuples with one entry per
        model.
        """
        self.interact()
        system_matrices = self._broadcast_model_argument(
            system_matrices, "system_matrices"
        )
        sys_noise_covs = self._broadcast_model_argument(
            sys_noise_covs, "sys_noise_covs"
        )
        sys_inputs = self._broadcast_model_argument(sys_inputs, "sys_inputs")

        for curr_filter, curr_system_matrix, curr_noise_cov, curr_input in zip(
            self.filter_bank, system_matrices, sys_noise_covs, sys_inputs
        ):
            if curr_input is None:
                curr_filter.predict_linear(curr_system_matrix, curr_noise_cov)
            else:
                curr_filter.predict_linear(
                    curr_system_matrix, curr_noise_cov, curr_input
                )

    def predict_nonlinear(
        self,
        transition_functions,
        sys_noise_covs,
        dts=None,
        fx_args=None,
    ):
        """Predict each model with a nonlinear transition function.

        Parameters can be shared across all models or supplied per model. ``fx_args``
        may be ``None``, a single dictionary shared across all models, or a list/tuple
        of dictionaries.
        """
        self.interact()
        transition_functions = self._broadcast_model_argument(
            transition_functions, "transition_functions"
        )
        sys_noise_covs = self._broadcast_model_argument(
            sys_noise_covs, "sys_noise_covs"
        )
        dts = self._broadcast_model_argument(dts, "dts")
        fx_args = self._broadcast_keyword_argument(fx_args, "fx_args")

        for curr_filter, curr_fx, curr_noise_cov, curr_dt, curr_fx_args in zip(
            self.filter_bank, transition_functions, sys_noise_covs, dts, fx_args
        ):
            curr_fx_args = {} if curr_fx_args is None else dict(curr_fx_args)
            if curr_dt is None:
                curr_filter.predict_nonlinear(curr_fx, curr_noise_cov, **curr_fx_args)
            else:
                curr_filter.predict_nonlinear(
                    curr_fx, curr_noise_cov, dt=curr_dt, **curr_fx_args
                )

    def update_identity(self, measurement, meas_noises):
        """Update each model with an identity measurement model.

        ``meas_noises`` can either be shared across all models or be provided as a
        list/tuple with one entry per model.
        """
        identity_matrix = eye(self.dim)
        self.update_linear(measurement, identity_matrix, meas_noises)

    def update_linear(self, measurement, measurement_matrices, meas_noises):
        """Update each model with a linear measurement model.

        ``measurement_matrices`` and ``meas_noises`` can either be shared across all
        models or be provided as lists/tuples with one entry per model.
        """
        measurement_matrices = self._broadcast_model_argument(
            measurement_matrices, "measurement_matrices"
        )
        meas_noises = self._broadcast_model_argument(meas_noises, "meas_noises")

        log_likelihoods = empty(self.n_models)
        for model_index, (
            curr_filter,
            curr_measurement_matrix,
            curr_meas_noise,
        ) in enumerate(zip(self.filter_bank, measurement_matrices, meas_noises)):
            predicted_state = self._as_gaussian(curr_filter.filter_state)
            log_likelihoods[model_index] = self._log_linear_measurement_likelihood(
                measurement,
                predicted_state,
                curr_measurement_matrix,
                curr_meas_noise,
            )

        self.latest_log_model_likelihoods = log_likelihoods
        self.latest_model_likelihoods = exp(log_likelihoods)
        self.update_mode_probabilities(log_likelihoods=log_likelihoods)

        for curr_filter, curr_measurement_matrix, curr_meas_noise in zip(
            self.filter_bank, measurement_matrices, meas_noises
        ):
            curr_filter.update_linear(
                measurement, curr_measurement_matrix, curr_meas_noise
            )

    def update_nonlinear(
        self,
        measurement,
        measurement_functions,
        meas_noises,
        likelihoods=None,
        log_likelihoods=None,
        hx_args=None,
    ):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Update each model with a nonlinear measurement function.

        Because current nonlinear filters in PyRecEst do not expose a uniform
        predictive-likelihood interface, nonlinear IMM updates require externally
        supplied model likelihoods (or log-likelihoods).
        """
        if (likelihoods is None) == (log_likelihoods is None):
            raise ValueError(
                "Provide exactly one of likelihoods or log_likelihoods for a nonlinear "
                "IMM update."
            )

        measurement_functions = self._broadcast_model_argument(
            measurement_functions, "measurement_functions"
        )
        meas_noises = self._broadcast_model_argument(meas_noises, "meas_noises")
        hx_args = self._broadcast_keyword_argument(hx_args, "hx_args")

        self.update_mode_probabilities(
            likelihoods=likelihoods, log_likelihoods=log_likelihoods
        )

        for curr_filter, curr_hx, curr_meas_noise, curr_hx_args in zip(
            self.filter_bank, measurement_functions, meas_noises, hx_args
        ):
            curr_hx_args = {} if curr_hx_args is None else dict(curr_hx_args)
            curr_filter.update_nonlinear(
                measurement, curr_hx, curr_meas_noise, **curr_hx_args
            )

    def update_mode_probabilities(self, likelihoods=None, log_likelihoods=None):
        """Update model probabilities from external per-model likelihoods."""
        if (likelihoods is None) == (log_likelihoods is None):
            raise ValueError(
                "Provide exactly one of likelihoods or log_likelihoods to update "
                "mode probabilities."
            )

        if likelihoods is not None:
            likelihoods = asarray(likelihoods, dtype=float).reshape(-1)
            if likelihoods.shape != (self.n_models,):
                raise ValueError(
                    "likelihoods must contain exactly one value per model in the IMM."
                )
            if pyrecest.backend.any(likelihoods < 0.0):
                raise ValueError("likelihoods must be nonnegative.")
            log_likelihoods = full(self.n_models, -float("inf"))
            positive = likelihoods > 0.0
            log_likelihoods[positive] = log(likelihoods[positive])
            self.latest_model_likelihoods = likelihoods
        else:
            log_likelihoods = asarray(log_likelihoods, dtype=float).reshape(-1)
            if log_likelihoods.shape != (self.n_models,):
                raise ValueError(
                    "log_likelihoods must contain exactly one value per model in the IMM."
                )
            self.latest_model_likelihoods = exp(log_likelihoods)

        prior_probabilities = asarray(self.mode_probabilities, dtype=float).reshape(-1)
        log_prior = full(self.n_models, -float("inf"))
        positive = prior_probabilities > 0.0
        log_prior[positive] = log(prior_probabilities[positive])

        log_posterior_unnormalized = log_prior + log_likelihoods
        if not isfinite(log_posterior_unnormalized).any():
            raise ValueError(
                "All model posterior weights are numerically zero. Check the supplied "
                "likelihoods and priors."
            )

        log_normalizer = logsumexp(log_posterior_unnormalized)
        posterior_probabilities = exp(log_posterior_unnormalized - log_normalizer)

        self.mode_probabilities = self._prepare_mode_probabilities(
            posterior_probabilities, self.n_models
        )
        self.latest_log_model_likelihoods = log_likelihoods
        return self.mode_probabilities

    def get_point_estimate(self):
        """Return the IMM mean estimate."""
        return self.combined_filter_state.mean()

    def _validate_filter_bank(self):
        if not self.filter_bank:
            raise ValueError("filter_bank must contain at least one filter.")

        curr_dims = [
            self._as_gaussian(curr_filter.filter_state).dim
            for curr_filter in self.filter_bank
        ]
        if any(curr_dim != curr_dims[0] for curr_dim in curr_dims[1:]):
            raise ValueError(
                "All filters in filter_bank must have the same state dimension."
            )

    @staticmethod
    def _prepare_transition_matrix(transition_matrix, n_models):
        transition_matrix = asarray(transition_matrix, dtype=float)
        if transition_matrix.shape != (n_models, n_models):
            raise ValueError("transition_matrix must have shape (n_models, n_models).")
        if pyrecest.backend.any(transition_matrix < 0.0):
            raise ValueError("transition_matrix must be elementwise nonnegative.")

        row_sums = transition_matrix.sum(axis=1)
        if pyrecest.backend.any(row_sums <= 0.0):
            raise ValueError(
                "Each row of transition_matrix must sum to a positive value."
            )
        if not allclose(row_sums, 1.0):
            warnings.warn(
                "Rows of transition_matrix do not sum to one. Renormalizing rows.",
                UserWarning,
            )
            transition_matrix = transition_matrix / row_sums[:, None]
        return transition_matrix

    @staticmethod
    def _prepare_mode_probabilities(mode_probabilities, n_models):
        if mode_probabilities is None:
            mode_probabilities = ones(n_models) / n_models
        else:
            mode_probabilities = asarray(mode_probabilities, dtype=float).reshape(-1)
            if mode_probabilities.shape != (n_models,):
                raise ValueError(
                    "mode_probabilities must contain exactly one value per model."
                )
            if pyrecest.backend.any(mode_probabilities < 0.0):
                raise ValueError("mode_probabilities must be elementwise nonnegative.")
            curr_sum = mode_probabilities.sum()
            if curr_sum <= 0.0:
                raise ValueError(
                    "At least one model probability must be strictly positive."
                )
            if not isclose(curr_sum, 1.0):
                warnings.warn(
                    "mode_probabilities do not sum to one. Renormalizing.",
                    UserWarning,
                )
                mode_probabilities = mode_probabilities / curr_sum
        return array(mode_probabilities)

    def _broadcast_model_argument(self, value, name):
        if isinstance(value, (list, tuple)):
            if len(value) != self.n_models:
                raise ValueError(
                    f"{name} must have one entry per model when provided as a list or tuple."
                )
            return list(value)
        return [value] * self.n_models

    def _broadcast_keyword_argument(self, value, name):
        if value is None:
            return [None] * self.n_models
        if isinstance(value, dict):
            return [value] * self.n_models
        if isinstance(value, (list, tuple)):
            if len(value) != self.n_models:
                raise ValueError(
                    f"{name} must have one entry per model when provided as a list or tuple."
                )
            return list(value)
        raise ValueError(
            f"{name} must be None, a dict shared across models, or a list/tuple of dicts."
        )

    @staticmethod
    def _as_gaussian(state) -> GaussianDistribution:
        if isinstance(state, GaussianDistribution):
            return state

        mu = None
        covariance = None

        if hasattr(state, "mu") and hasattr(state, "C"):
            mu = getattr(state, "mu")
            covariance = getattr(state, "C")
        elif hasattr(state, "mean") and hasattr(state, "covariance"):
            mu = state.mean()
            covariance = state.covariance()

        if mu is None or covariance is None:
            raise ValueError(
                "Subfilter states must be Gaussian or expose mean/covariance information."
            )

        mu = asarray(mu, dtype=float).reshape(-1)
        covariance = asarray(covariance, dtype=float)
        if covariance.ndim == 0:
            covariance = covariance.reshape((1, 1))
        elif covariance.ndim == 1:
            covariance = diag(covariance)
        return GaussianDistribution(mu, covariance, check_validity=False)

    @staticmethod
    def _moment_match_gaussians(gaussians, weights) -> GaussianDistribution:
        weights = asarray(weights, dtype=float).reshape(-1)
        if weights.shape != (len(gaussians),):
            raise ValueError("weights must have one entry per Gaussian component.")
        curr_sum = weights.sum()
        if curr_sum <= 0.0:
            raise ValueError("At least one mixture weight must be strictly positive.")
        if not isclose(curr_sum, 1.0):
            weights = weights / curr_sum

        means = stack([asarray(curr_state.mu, dtype=float) for curr_state in gaussians])
        covariance = zeros_like(asarray(gaussians[0].C, dtype=float))
        mean = weights @ means
        for curr_weight, curr_state in zip(weights, gaussians):
            curr_covariance = asarray(curr_state.C, dtype=float)
            curr_diff = asarray(curr_state.mu, dtype=float) - mean
            covariance += curr_weight * (curr_covariance + outer(curr_diff, curr_diff))
        covariance = 0.5 * (covariance + covariance.T)
        return GaussianDistribution(mean, covariance, check_validity=False)

    @staticmethod
    def _log_linear_measurement_likelihood(
        measurement,
        predicted_state: GaussianDistribution,
        measurement_matrix,
        meas_noise,
    ) -> float:
        measurement = asarray(measurement, dtype=float).reshape(-1)
        measurement_matrix = asarray(measurement_matrix, dtype=float)
        meas_noise = asarray(meas_noise, dtype=float)
        if meas_noise.ndim == 0:
            meas_noise = meas_noise.reshape((1, 1))
        elif meas_noise.ndim == 1:
            meas_noise = diag(meas_noise)

        innovation = measurement - measurement_matrix @ asarray(
            predicted_state.mu, dtype=float
        )
        innovation_covariance = (
            measurement_matrix
            @ asarray(predicted_state.C, dtype=float)
            @ measurement_matrix.T
            + meas_noise
        )
        det_value = float(linalg.det(innovation_covariance))
        if det_value <= 0.0:
            raise ValueError(
                "Innovation covariance must be positive definite to evaluate the IMM likelihood."
            )
        logdet = float(log(array(det_value)))
        mahalanobis_distance = float(
            innovation.T @ linalg.solve(innovation_covariance, innovation)
        )
        return float(
            -0.5
            * (
                innovation.shape[0] * float(log(2.0 * pi))
                + logdet
                + mahalanobis_distance
            )
        )


IMM = InteractingMultipleModelFilter
