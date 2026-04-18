"""Goal-conditioned replay IMM-style filter.

This filter combines

- a discrete latent goal index over a fixed bank of candidate goals,
- a discrete motion mode with two modes: ``"smooth"`` and ``"jump"``,
- a continuous Gaussian state over concatenated position and velocity.

Compared with the earlier draft, this version

- uses ``position_dim`` consistently,
- exposes a replay-oriented public interface shared with the particle filter,
- provides moment-matched position/velocity marginals and a discrete goal
  posterior,
- supports factorized position/velocity initialization, and
- exposes predictive / update marginal likelihoods for model-comparison
  workflows.

The continuous state is always the concatenated vector ``x = [z, v]`` with
dimension ``2 * position_dim``.
"""

from __future__ import annotations

from typing import Callable

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,duplicate-code
from pyrecest.backend import (
    argmax,
    argmin,
    array,
    atleast_1d,
    concatenate,
    diag,
    exp,
    eye,
    full,
    linalg,
    log,
)
from pyrecest.backend import max as backend_max
from pyrecest.backend import (
    maximum,
    ndim,
    outer,
    pi,
    reshape,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    zeros,
)
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin


class GoalConditionedReplayIMMFilter(  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    AbstractFilter, EuclideanFilterMixin
):
    """Goal-conditioned replay filter with discrete goals and IMM-style mode switching.

    Parameters
    ----------
    initial_state:
        Either a GaussianDistribution or a tuple ``(mean, covariance)``.
        The mean/covariance may describe either:
        - position only, with shape ``(position_dim,)`` and
          ``(position_dim, position_dim)``
        - concatenated position/velocity, with shape ``(2 * position_dim,)`` and
          ``(2 * position_dim, 2 * position_dim)``.
    candidate_goals:
        Candidate replay goals with shape ``(n_goals, position_dim)``.
        For 1-D position spaces, a 1-D array is interpreted as multiple scalar
        candidate goals.
    dt:
        Default time step used by :meth:`predict_replay`.
    attraction_strength:
        Strength of the smooth goal attraction.
    velocity_decay:
        Velocity retention in the smooth mode.
    jump_fraction:
        Fraction of the remaining distance to the goal traversed during a
        jump-mode prediction.
    jump_velocity_decay:
        Velocity retention in the jump mode.
    jump_probability:
        Default probability of switching from the smooth mode to the jump mode.
        Ignored if ``mode_transition_matrix`` is provided.
    jump_stickiness:
        Default probability of staying in the jump mode. Ignored if
        ``mode_transition_matrix`` is provided.
    smooth_sys_noise_cov, jump_sys_noise_cov:
        Process-noise covariances for the smooth and jump modes.
    goal_transition_matrix:
        Row-stochastic matrix with shape ``(n_goals, n_goals)`` and entries
        ``P(goal_t = j | goal_{t-1} = i)``.
    mode_transition_matrix:
        Row-stochastic matrix with shape ``(2, 2)`` and entries
        ``P(mode_t = j | mode_{t-1} = i)``.
    goal_prior, mode_prior:
        Initial probability vectors over candidate goals and motion modes.
    initial_velocity_covariance:
        Covariance used to augment a position-only initial state.
    covariance_regularization:
        Small diagonal term added after covariance updates for numerical
        stability.
    weight_floor:
        Lower bound used when taking logarithms of mixture weights.
    finite_difference_epsilon:
        Step size for numerical Jacobians in :meth:`predict_nonlinear` and
        :meth:`update_nonlinear` if no analytic Jacobian is supplied.
    """

    mode_names = ("smooth", "jump")

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        initial_state,
        candidate_goals=None,
        *,
        goal_candidates=None,
        dt: float = 1.0,
        attraction_strength: float = 1.0,
        velocity_decay: float = 0.95,
        jump_fraction: float = 0.9,
        jump_velocity_decay: float = 0.25,
        jump_probability: float = 0.05,
        jump_stickiness: float = 0.05,
        smooth_sys_noise_cov=None,
        jump_sys_noise_cov=None,
        goal_transition_matrix=None,
        mode_transition_matrix=None,
        goal_prior=None,
        mode_prior=None,
        initial_velocity_covariance=None,
        covariance_regularization: float = 1e-9,
        weight_floor: float = 1e-300,
        finite_difference_epsilon: float = 1e-5,
    ):
        assert (
            pyrecest.backend.__backend_name__ != "jax"
        ), "GoalConditionedReplayIMMFilter is not supported on the JAX backend"
        raw_mean, raw_cov = self._extract_mean_and_cov(initial_state)
        raw_state_dim = raw_mean.shape[0]

        if candidate_goals is None:
            candidate_goals = goal_candidates
        elif goal_candidates is not None:
            goal_candidates = self._parse_candidate_goals(
                goal_candidates, raw_state_dim
            )
            candidate_goals = self._parse_candidate_goals(
                candidate_goals, raw_state_dim
            )
            if (
                candidate_goals.shape != goal_candidates.shape
                or backend_sum((candidate_goals - goal_candidates) ** 2) > 0
            ):
                raise ValueError(
                    "candidate_goals and goal_candidates must match when both are provided"
                )
        if candidate_goals is None:
            raise ValueError("candidate_goals or goal_candidates must be provided")

        parsed_goals = self._parse_candidate_goals(candidate_goals, raw_state_dim)
        self._candidate_goals = parsed_goals
        self.position_dim = parsed_goals.shape[1]
        self.state_dim = 2 * self.position_dim
        self.n_goals = parsed_goals.shape[0]
        self.n_modes = 2
        self.n_components = self.n_goals * self.n_modes

        self.dt = float(dt)
        self.attraction_strength = float(attraction_strength)
        self.velocity_decay = float(velocity_decay)
        self.jump_fraction = float(jump_fraction)
        self.jump_velocity_decay = float(jump_velocity_decay)
        self.covariance_regularization = float(covariance_regularization)
        self.weight_floor = float(weight_floor)
        self.finite_difference_epsilon = float(finite_difference_epsilon)
        self._last_update_log_marginal = None

        self._initial_velocity_covariance = self._prepare_initial_velocity_covariance(
            raw_cov, raw_state_dim, initial_velocity_covariance
        )

        self.smooth_sys_noise_cov = self._as_square_matrix(
            (
                smooth_sys_noise_cov
                if smooth_sys_noise_cov is not None
                else 0.05 * eye(self.state_dim)
            ),
            self.state_dim,
            "smooth_sys_noise_cov",
        )
        self.jump_sys_noise_cov = self._as_square_matrix(
            (
                jump_sys_noise_cov
                if jump_sys_noise_cov is not None
                else 1.0 * eye(self.state_dim)
            ),
            self.state_dim,
            "jump_sys_noise_cov",
        )

        if goal_transition_matrix is None:
            self.goal_transition_matrix = eye(self.n_goals)
        else:
            self.goal_transition_matrix = self._validate_transition_matrix(
                goal_transition_matrix,
                self.n_goals,
                "goal_transition_matrix",
            )

        if mode_transition_matrix is None:
            self.mode_transition_matrix = self._validate_transition_matrix(
                array(
                    [
                        [1.0 - jump_probability, jump_probability],
                        [1.0 - jump_stickiness, jump_stickiness],
                    ]
                ),
                self.n_modes,
                "mode_transition_matrix",
            )
        else:
            self.mode_transition_matrix = self._validate_transition_matrix(
                mode_transition_matrix,
                self.n_modes,
                "mode_transition_matrix",
            )

        self.goal_prior = self._validate_probability_vector(
            (
                full((self.n_goals,), 1.0 / self.n_goals)
                if goal_prior is None
                else goal_prior
            ),
            self.n_goals,
            "goal_prior",
        )
        self.mode_prior = self._validate_probability_vector(
            array([1.0, 0.0]) if mode_prior is None else mode_prior,
            self.n_modes,
            "mode_prior",
        )

        self._component_states: list[GaussianDistribution] = []
        self._component_weights = self._build_initial_component_weights()

        EuclideanFilterMixin.__init__(self)
        AbstractFilter.__init__(self, None)
        self.filter_state = initial_state

    @classmethod
    def from_factorized_priors(
        cls,
        position_distribution,
        candidate_goals=None,
        velocity_distribution=None,
        *,
        goal_candidates=None,
        goal_prior=None,
        mode_prior=None,
        initial_velocity_covariance=None,
        **kwargs,
    ) -> "GoalConditionedReplayIMMFilter":
        """Construct the filter from separate position/velocity priors.

        ``position_distribution`` and ``velocity_distribution`` may be Gaussian
        distributions, tuples ``(mean, covariance)``, or any objects exposing
        compatible first and second moments through ``mean()`` and either
        ``covariance()`` or ``C``.
        """
        if candidate_goals is None:
            candidate_goals = goal_candidates
        elif goal_candidates is not None:
            raise ValueError(
                "Provide at most one of candidate_goals and goal_candidates"
            )

        position_mean, position_cov = cls._extract_distribution_moments(
            position_distribution,
            "position_distribution",
        )
        position_mean = atleast_1d(array(position_mean))
        position_cov = cls._as_square_matrix(
            position_cov,
            position_mean.shape[0],
            "position_distribution covariance",
        )

        if velocity_distribution is None:
            initial_state = (position_mean, position_cov)
            return cls(
                initial_state=initial_state,
                candidate_goals=candidate_goals,
                goal_prior=goal_prior,
                mode_prior=mode_prior,
                initial_velocity_covariance=initial_velocity_covariance,
                **kwargs,
            )

        velocity_mean, velocity_cov = cls._extract_distribution_moments(
            velocity_distribution,
            "velocity_distribution",
        )
        velocity_mean = atleast_1d(array(velocity_mean))
        velocity_cov = cls._as_square_matrix(
            velocity_cov,
            velocity_mean.shape[0],
            "velocity_distribution covariance",
        )

        if velocity_mean.shape != position_mean.shape:
            raise ValueError(
                "position_distribution and velocity_distribution must have the "
                "same dimension"
            )

        zeros_block = zeros((position_mean.shape[0], position_mean.shape[0]))
        full_mean = concatenate([position_mean, velocity_mean], axis=0)
        full_cov = concatenate(
            [
                concatenate([position_cov, zeros_block], axis=1),
                concatenate([zeros_block, velocity_cov], axis=1),
            ],
            axis=0,
        )

        return cls(
            initial_state=(full_mean, full_cov),
            candidate_goals=candidate_goals,
            goal_prior=goal_prior,
            mode_prior=mode_prior,
            initial_velocity_covariance=initial_velocity_covariance,
            **kwargs,
        )

    @staticmethod
    def _extract_mean_and_cov(initial_state):
        if isinstance(initial_state, GaussianDistribution):
            mean = initial_state.mu
            cov = initial_state.C
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            mean = initial_state[0]
            cov = initial_state[1]
        else:
            raise ValueError(
                "initial_state must be a GaussianDistribution or a tuple of "
                "(mean, covariance)"
            )

        mean = atleast_1d(array(mean))
        if ndim(cov) == 0:
            cov = reshape(array(cov), (1, 1))
        else:
            cov = array(cov)

        if mean.ndim != 1:
            raise ValueError("Initial-state mean must be one-dimensional")
        if cov.ndim != 2:
            raise ValueError("Initial-state covariance must be two-dimensional")
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Initial-state covariance must be square")
        if cov.shape[0] != mean.shape[0]:
            raise ValueError("Initial-state mean/covariance size mismatch")

        return mean, cov

    @staticmethod
    def _extract_distribution_moments(distribution, name: str):
        if isinstance(distribution, GaussianDistribution):
            return distribution.mu, distribution.C
        if isinstance(distribution, tuple) and len(distribution) == 2:
            return distribution[0], distribution[1]
        if hasattr(distribution, "mean") and callable(distribution.mean):
            mean = distribution.mean()
            if hasattr(distribution, "covariance") and callable(
                distribution.covariance
            ):
                cov = distribution.covariance()
            elif hasattr(distribution, "cov") and callable(distribution.cov):
                cov = distribution.cov()
            elif hasattr(distribution, "C"):
                cov = distribution.C
            else:
                raise ValueError(
                    f"{name} must expose covariance() / cov() / C to build a "
                    "Gaussian initialization"
                )
            return mean, cov
        raise ValueError(
            f"{name} must be a GaussianDistribution, a (mean, covariance) tuple, "
            "or expose compatible moment methods"
        )

    @staticmethod
    def _parse_candidate_goals(candidate_goals, raw_state_dim: int):
        parsed = array(candidate_goals)
        if ndim(parsed) == 0:
            parsed = reshape(parsed, (1, 1))
        elif ndim(parsed) == 1:
            if raw_state_dim > 2:
                raise ValueError(
                    "For position dimensions larger than one, please pass "
                    "candidate_goals as a 2-D array of shape "
                    "(n_goals, position_dim)."
                )
            parsed = reshape(parsed, (-1, 1))
        elif ndim(parsed) != 2:
            raise ValueError("candidate_goals must be a 1-D or 2-D array")

        if parsed.shape[0] == 0:
            raise ValueError("At least one candidate goal is required")

        return parsed

    def _prepare_initial_velocity_covariance(
        self,
        raw_cov,
        raw_state_dim: int,
        initial_velocity_covariance,
    ):
        if initial_velocity_covariance is not None:
            return self._as_square_matrix(
                initial_velocity_covariance,
                self.position_dim,
                "initial_velocity_covariance",
            )

        if raw_state_dim == self.position_dim:
            return eye(self.position_dim)
        if raw_state_dim == self.state_dim:
            return raw_cov[self.velocity_slice, self.velocity_slice]
        raise ValueError(
            "Initial state dimension must match either position_dim or "
            "2 * position_dim"
        )

    @staticmethod
    def _validate_transition_matrix(matrix, expected_size: int, name: str):
        matrix = array(matrix)
        if matrix.shape != (expected_size, expected_size):
            raise ValueError(
                f"{name} must have shape ({expected_size}, {expected_size})"
            )

        row_sums = backend_sum(matrix, axis=1)
        for idx in range(expected_size):
            if abs(float(row_sums[idx]) - 1.0) > 1e-10:
                raise ValueError(f"Each row of {name} must sum to one")
            for jdx in range(expected_size):
                if float(matrix[idx, jdx]) < 0.0:
                    raise ValueError(f"{name} must not contain negative entries")

        return matrix

    @staticmethod
    def _validate_probability_vector(vector, expected_size: int, name: str):
        vector = atleast_1d(array(vector))
        if vector.shape != (expected_size,):
            raise ValueError(f"{name} must have shape ({expected_size},)")
        if float(backend_sum(vector)) <= 0.0:
            raise ValueError(f"{name} must have positive total mass")
        for idx in range(expected_size):
            if float(vector[idx]) < 0.0:
                raise ValueError(f"{name} must not contain negative entries")
        return vector / backend_sum(vector)

    def _build_initial_component_weights(self):
        weights = zeros((self.n_components,))
        for goal_index in range(self.n_goals):
            for mode_index in range(self.n_modes):
                weights[self._component_index(goal_index, mode_index)] = (
                    self.goal_prior[goal_index] * self.mode_prior[mode_index]
                )
        return weights / backend_sum(weights)

    def _component_index(self, goal_index: int, mode_index: int) -> int:
        return goal_index * self.n_modes + mode_index

    def _component_meta(self, component_index: int):
        goal_index = component_index // self.n_modes
        mode_index = component_index % self.n_modes
        return goal_index, mode_index

    @staticmethod
    def _copy_gaussian_state(state: GaussianDistribution) -> GaussianDistribution:
        return GaussianDistribution(state.mu, state.C, check_validity=False)

    def _regularize_covariance(self, covariance):
        covariance = 0.5 * (covariance + covariance.T)
        return covariance + self.covariance_regularization * eye(covariance.shape[0])

    @staticmethod
    def _as_square_matrix(matrix_like, dim: int, name: str):
        matrix = array(matrix_like)
        if ndim(matrix) == 0:
            matrix = reshape(matrix, (1, 1)) if dim == 1 else matrix * eye(dim)
        elif ndim(matrix) == 1:
            if matrix.shape[0] != dim:
                raise ValueError(f"{name} diagonal length must equal {dim}")
            matrix = diag(matrix)
        elif ndim(matrix) != 2:
            raise ValueError(f"{name} must be scalar, 1-D, or 2-D")

        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")

        return matrix

    @staticmethod
    def _coerce_measurement_noise_cov(meas_noise, dim: int, name: str = "meas_noise"):
        if hasattr(meas_noise, "dim") and not callable(meas_noise.dim):
            if meas_noise.dim != dim:
                raise ValueError(f"{name}.dim must equal {dim}")
            if hasattr(meas_noise, "C"):
                return GoalConditionedReplayIMMFilter._as_square_matrix(
                    meas_noise.C,
                    dim,
                    name,
                )
            if hasattr(meas_noise, "covariance"):
                return GoalConditionedReplayIMMFilter._as_square_matrix(
                    meas_noise.covariance(),
                    dim,
                    name,
                )
        return GoalConditionedReplayIMMFilter._as_square_matrix(
            meas_noise,
            dim,
            name,
        )

    @staticmethod
    def _augment_position_state(position_mean, position_cov, velocity_cov):
        position_dim = position_mean.shape[0]
        zeros_block = zeros((position_dim, position_dim))
        full_mean = concatenate([position_mean, zeros((position_dim,))], axis=0)
        top = concatenate([position_cov, zeros_block], axis=1)
        bottom = concatenate([zeros_block, velocity_cov], axis=1)
        full_cov = concatenate([top, bottom], axis=0)
        return full_mean, full_cov

    def _canonical_state(self, mean, cov):
        if mean.shape[0] == self.position_dim:
            return self._augment_position_state(
                mean,
                cov,
                self._initial_velocity_covariance,
            )
        if mean.shape[0] == self.state_dim:
            return mean, cov
        raise ValueError(
            "State dimension must equal either position_dim or 2 * position_dim"
        )

    def _moment_match(self, weights, states):
        weights = atleast_1d(array(weights))
        weights = weights / backend_sum(weights)

        mean = zeros((self.state_dim,))
        for weight, state in zip(weights, states):
            mean = mean + weight * state.mu

        covariance = zeros((self.state_dim, self.state_dim))
        for weight, state in zip(weights, states):
            diff = state.mu - mean
            covariance = covariance + weight * (state.C + outer(diff, diff))

        return GaussianDistribution(
            mean,
            self._regularize_covariance(covariance),
            check_validity=False,
        )

    def _update_collapsed_state(self):
        self._filter_state = self._moment_match(
            self._component_weights,
            self._component_states,
        )

    @property
    def filter_state(self) -> GaussianDistribution:
        return self._copy_gaussian_state(self._filter_state)

    @filter_state.setter
    def filter_state(self, new_state):
        mean, cov = self._extract_mean_and_cov(new_state)
        canonical_mean, canonical_cov = self._canonical_state(mean, cov)
        canonical_cov = self._regularize_covariance(canonical_cov)

        self._component_states = [
            GaussianDistribution(
                canonical_mean,
                canonical_cov,
                check_validity=False,
            )
            for _ in range(self.n_components)
        ]
        self._component_weights = self._build_initial_component_weights()
        self._update_collapsed_state()

    def initialize_from_state_priors(
        self,
        position_prior,
        velocity_prior=None,
        *,
        goal_prior=None,
        mode_prior=None,
    ):
        """Reset the filter from separate position / velocity priors.

        This is a convenience instance-level analogue of
        :meth:`from_factorized_priors`.
        """
        position_mean, position_cov = self._extract_distribution_moments(
            position_prior,
            "position_prior",
        )
        position_mean = atleast_1d(array(position_mean))
        position_cov = self._as_square_matrix(
            position_cov,
            position_mean.shape[0],
            "position_prior covariance",
        )

        if position_mean.shape != (self.position_dim,):
            raise ValueError(
                f"position_prior mean must have shape ({self.position_dim},)"
            )

        if velocity_prior is None:
            canonical_mean, canonical_cov = self._canonical_state(
                position_mean,
                position_cov,
            )
        else:
            velocity_mean, velocity_cov = self._extract_distribution_moments(
                velocity_prior,
                "velocity_prior",
            )
            velocity_mean = atleast_1d(array(velocity_mean))
            velocity_cov = self._as_square_matrix(
                velocity_cov,
                velocity_mean.shape[0],
                "velocity_prior covariance",
            )
            if velocity_mean.shape != (self.position_dim,):
                raise ValueError(
                    f"velocity_prior mean must have shape ({self.position_dim},)"
                )
            zeros_cross = zeros((self.position_dim, self.position_dim))
            canonical_mean = concatenate([position_mean, velocity_mean], axis=0)
            canonical_cov = concatenate(
                [
                    concatenate([position_cov, zeros_cross], axis=1),
                    concatenate([zeros_cross, velocity_cov], axis=1),
                ],
                axis=0,
            )

        if goal_prior is not None:
            self.goal_prior = self._validate_probability_vector(
                goal_prior,
                self.n_goals,
                "goal_prior",
            )
        if mode_prior is not None:
            self.mode_prior = self._validate_probability_vector(
                mode_prior,
                self.n_modes,
                "mode_prior",
            )

        self.filter_state = (canonical_mean, canonical_cov)
        return self

    @property
    def dim(self):
        return self.state_dim

    @property
    def position_slice(self) -> slice:
        return slice(0, self.position_dim)

    @property
    def velocity_slice(self) -> slice:
        return slice(self.position_dim, 2 * self.position_dim)

    @property
    def candidate_goals(self):
        return array(self._candidate_goals)

    @property
    def goal_candidates(self):
        return self.candidate_goals

    @property
    def component_weights(self):
        return array(self._component_weights)

    @property
    def component_filter_states(self):
        return [self._copy_gaussian_state(state) for state in self._component_states]

    @property
    def goal_probabilities(self):
        probabilities = zeros((self.n_goals,))
        for goal_index in range(self.n_goals):
            probabilities[goal_index] = (
                self._component_weights[self._component_index(goal_index, 0)]
                + self._component_weights[self._component_index(goal_index, 1)]
            )
        return probabilities

    @property
    def mode_probabilities(self):
        probabilities = zeros((self.n_modes,))
        for mode_index in range(self.n_modes):
            total = 0.0
            for goal_index in range(self.n_goals):
                total = (
                    total
                    + self._component_weights[
                        self._component_index(goal_index, mode_index)
                    ]
                )
            probabilities[mode_index] = total
        return probabilities

    @property
    def last_update_log_marginal(self):
        return self._last_update_log_marginal

    def get_point_estimate(self):
        return self.filter_state.mean()

    def get_state_estimate(self):
        return self.get_point_estimate()

    def get_position_estimate(self):
        return self.filter_state.mean()[self.position_slice]

    def get_velocity_estimate(self):
        return self.filter_state.mean()[self.velocity_slice]

    def get_goal_estimate(self):
        probabilities = self.goal_probabilities
        return probabilities @ self._candidate_goals

    def get_position_distribution(self) -> GaussianDistribution:
        mean = self.get_position_estimate()
        covariance = self.filter_state.C[self.position_slice, self.position_slice]
        return GaussianDistribution(mean, covariance, check_validity=False)

    def get_velocity_distribution(self) -> GaussianDistribution:
        mean = self.get_velocity_estimate()
        covariance = self.filter_state.C[self.velocity_slice, self.velocity_slice]
        return GaussianDistribution(mean, covariance, check_validity=False)

    def get_goal_distribution(self) -> LinearDiracDistribution:
        return LinearDiracDistribution(self._candidate_goals, self.goal_probabilities)

    def get_goal_posterior_weights(self, candidate_goals=None):
        if candidate_goals is None:
            return self.goal_probabilities

        candidate_goals = self._parse_candidate_goals(
            candidate_goals, self.position_dim
        )
        diff = self._candidate_goals[:, None, :] - candidate_goals[None, :, :]
        sq_dist = backend_sum(diff * diff, axis=2)
        nearest = argmin(sq_dist, axis=1)

        posterior_weights = zeros((candidate_goals.shape[0],))
        for idx in range(candidate_goals.shape[0]):
            posterior_weights[idx] = backend_sum(
                self.goal_probabilities[nearest == idx]
            )

        total = backend_sum(posterior_weights)
        if float(total) <= 0.0:
            raise ValueError("Projected goal posterior is degenerate")
        return posterior_weights / total

    def most_likely_goal_index(self) -> int:
        return int(argmax(self.goal_probabilities))

    def most_likely_goal(self):
        return self._candidate_goals[self.most_likely_goal_index()]

    def most_likely_mode_index(self) -> int:
        return int(argmax(self.mode_probabilities))

    def most_likely_mode(self) -> str:
        return self.mode_names[self.most_likely_mode_index()]

    def _logsumexp(self, values):
        max_value = backend_max(values)
        shifted = exp(values - max_value)
        normalizer = backend_sum(shifted)
        if float(normalizer) <= 0.0:
            return max_value
        return max_value + log(normalizer)

    def reweight_components(
        self,
        likelihoods=None,
        log_likelihoods=None,
        *,
        return_log_marginal: bool = False,
    ):
        """Reweight only the discrete hypotheses.

        This is useful when an external spike / replay likelihood is already
        available and only the discrete goal/mode posterior should be updated.
        """
        if (likelihoods is None) == (log_likelihoods is None):
            raise ValueError("Provide exactly one of likelihoods or log_likelihoods")

        if likelihoods is not None:
            log_likelihoods = log(
                maximum(atleast_1d(array(likelihoods)), self.weight_floor)
            )
        else:
            log_likelihoods = atleast_1d(array(log_likelihoods))

        if log_likelihoods.shape != (self.n_components,):
            raise ValueError(f"Expected {self.n_components} component likelihoods")

        raw_log_weights = log_likelihoods + log(
            maximum(self._component_weights, self.weight_floor)
        )
        self._last_update_log_marginal = self._logsumexp(raw_log_weights)
        self._component_weights = self._normalize_log_weights(raw_log_weights)
        self._update_collapsed_state()

        if return_log_marginal:
            return self._last_update_log_marginal
        return self

    def _normalize_log_weights(self, log_weights):
        max_log_weight = backend_max(log_weights)
        shifted = exp(log_weights - max_log_weight)
        normalizer = backend_sum(shifted)
        if float(normalizer) <= 0.0:
            return full((log_weights.shape[0],), 1.0 / log_weights.shape[0])
        return shifted / normalizer

    def _mix_components(self):
        mixed_states = [None] * self.n_components
        mixed_target_weights = zeros((self.n_components,))
        global_state = self._filter_state

        for target_goal in range(self.n_goals):
            for target_mode in range(self.n_modes):
                target_index = self._component_index(target_goal, target_mode)
                source_states = []
                source_weights = []

                for source_goal in range(self.n_goals):
                    for source_mode in range(self.n_modes):
                        source_index = self._component_index(source_goal, source_mode)
                        transition_probability = (
                            self.goal_transition_matrix[source_goal, target_goal]
                            * self.mode_transition_matrix[source_mode, target_mode]
                        )
                        source_states.append(self._component_states[source_index])
                        source_weights.append(
                            self._component_weights[source_index]
                            * transition_probability
                        )

                source_weights = array(source_weights)
                mixed_target_weights[target_index] = backend_sum(source_weights)

                if float(mixed_target_weights[target_index]) > 0.0:
                    normalized_weights = (
                        source_weights / mixed_target_weights[target_index]
                    )
                    mixed_states[target_index] = self._moment_match(
                        normalized_weights,
                        source_states,
                    )
                else:
                    mixed_states[target_index] = self._copy_gaussian_state(global_state)

        total = backend_sum(mixed_target_weights)
        self._component_weights = (
            mixed_target_weights / total
            if float(total) > 0.0
            else self._build_initial_component_weights()
        )
        self._component_states = mixed_states

    def _goal_conditioned_model(self, goal_index: int, mode_index: int, dt: float):
        identity = eye(self.position_dim)
        zeros_block = zeros((self.position_dim, self.position_dim))
        goal = self._candidate_goals[goal_index]

        if mode_index == 0:
            beta = self.attraction_strength
            pos_pos = (1.0 - 0.5 * beta * dt * dt) * identity
            pos_vel = dt * identity
            vel_pos = -beta * dt * identity
            vel_vel = self.velocity_decay * identity
            top = concatenate([pos_pos, pos_vel], axis=1)
            bottom = concatenate([vel_pos, vel_vel], axis=1)
            transition = concatenate([top, bottom], axis=0)
            bias = concatenate(
                [
                    0.5 * beta * dt * dt * goal,
                    beta * dt * goal,
                ],
                axis=0,
            )
            noise_cov = self.smooth_sys_noise_cov
        else:
            lambda_jump = self.jump_fraction
            pos_pos = (1.0 - lambda_jump) * identity
            pos_vel = dt * identity
            vel_pos = zeros_block
            vel_vel = self.jump_velocity_decay * identity
            top = concatenate([pos_pos, pos_vel], axis=1)
            bottom = concatenate([vel_pos, vel_vel], axis=1)
            transition = concatenate([top, bottom], axis=0)
            bias = concatenate(
                [
                    lambda_jump * goal,
                    zeros((self.position_dim,)),
                ],
                axis=0,
            )
            noise_cov = self.jump_sys_noise_cov

        return transition, bias, noise_cov

    def _generic_component_model(self, system_matrix, sys_noise_cov, sys_input=None):
        transition = self._as_square_matrix(
            system_matrix,
            self.state_dim,
            "system_matrix",
        )
        noise_cov = self._as_square_matrix(
            sys_noise_cov,
            self.state_dim,
            "sys_noise_cov",
        )
        if sys_input is None:
            bias = zeros((self.state_dim,))
        else:
            bias = atleast_1d(array(sys_input))
            if bias.shape != (self.state_dim,):
                raise ValueError(f"sys_input must have shape ({self.state_dim},)")
        return transition, bias, noise_cov

    def _predict_with_models(self, model_builder: Callable[[int, int], tuple]):
        self._mix_components()

        predicted_states = []
        for component_index in range(self.n_components):
            goal_index, mode_index = self._component_meta(component_index)
            state = self._component_states[component_index]
            transition, bias, noise_cov = model_builder(goal_index, mode_index)
            predicted_mean = transition @ state.mu + bias
            predicted_cov = transition @ state.C @ transition.T + noise_cov
            predicted_states.append(
                GaussianDistribution(
                    predicted_mean,
                    self._regularize_covariance(predicted_cov),
                    check_validity=False,
                )
            )

        self._component_states = predicted_states
        self._update_collapsed_state()
        return self

    def predict(self, dt=None, smooth_sys_noise_cov=None, jump_sys_noise_cov=None):
        return self.predict_replay(
            dt=dt,
            smooth_sys_noise_cov=smooth_sys_noise_cov,
            jump_sys_noise_cov=jump_sys_noise_cov,
        )

    def predict_replay(
        self,
        dt=None,
        smooth_sys_noise_cov=None,
        jump_sys_noise_cov=None,
    ):
        return self.predict_goal_conditioned(
            dt=dt,
            smooth_sys_noise_cov=smooth_sys_noise_cov,
            jump_sys_noise_cov=jump_sys_noise_cov,
        )

    def predict_goal_conditioned(
        self,
        dt=None,
        smooth_sys_noise_cov=None,
        jump_sys_noise_cov=None,
    ):
        """Predict with the built-in goal-conditioned replay dynamics."""
        if dt is None:
            dt = self.dt
        if smooth_sys_noise_cov is not None:
            self.smooth_sys_noise_cov = self._as_square_matrix(
                smooth_sys_noise_cov,
                self.state_dim,
                "smooth_sys_noise_cov",
            )
        if jump_sys_noise_cov is not None:
            self.jump_sys_noise_cov = self._as_square_matrix(
                jump_sys_noise_cov,
                self.state_dim,
                "jump_sys_noise_cov",
            )

        return self._predict_with_models(
            lambda goal_index, mode_index: self._goal_conditioned_model(
                goal_index,
                mode_index,
                dt,
            )
        )

    def predict_identity(self, sys_noise_cov, sys_input=None):
        system_matrix = eye(self.state_dim)
        return self.predict_linear(system_matrix, sys_noise_cov, sys_input=sys_input)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        transition, bias, noise_cov = self._generic_component_model(
            system_matrix,
            sys_noise_cov,
            sys_input=sys_input,
        )

        return self._predict_with_models(
            lambda _goal_index, _mode_index: (transition, bias, noise_cov)
        )

    def _call_fx(self, fx, state, dt, fx_args):
        if dt is None:
            return atleast_1d(array(fx(state, **fx_args)))
        return atleast_1d(array(fx(state, dt, **fx_args)))

    def _call_jacobian(self, jacobian, state, dt, fx_args):
        if dt is None:
            return array(jacobian(state, **fx_args))
        return array(jacobian(state, dt, **fx_args))

    def _numerical_jacobian(self, function, x):
        base_value = atleast_1d(array(function(x)))
        jacobian = zeros((base_value.shape[0], x.shape[0]))

        for dim_index in range(x.shape[0]):
            delta = zeros((x.shape[0],))
            delta[dim_index] = self.finite_difference_epsilon
            forward = atleast_1d(array(function(x + delta)))
            backward = atleast_1d(array(function(x - delta)))
            jacobian[:, dim_index] = (forward - backward) / (
                2.0 * self.finite_difference_epsilon
            )

        return jacobian, base_value

    def predict_nonlinear(
        self,
        fx: Callable,
        sys_noise_cov,
        jacobian: Callable | None = None,
        dt=None,
        **fx_args,
    ):
        """Predict with a shared nonlinear dynamics model for all components."""
        if dt is None:
            dt = self.dt
        noise_cov = self._as_square_matrix(
            sys_noise_cov,
            self.state_dim,
            "sys_noise_cov",
        )

        self._mix_components()
        predicted_states = []

        for component_index in range(self.n_components):
            state = self._component_states[component_index]

            if jacobian is None:
                transition_jacobian, predicted_mean = self._numerical_jacobian(
                    lambda x: self._call_fx(fx, x, dt, fx_args),
                    state.mu,
                )
            else:
                predicted_mean = self._call_fx(fx, state.mu, dt, fx_args)
                transition_jacobian = self._call_jacobian(
                    jacobian,
                    state.mu,
                    dt,
                    fx_args,
                )

            predicted_cov = (
                transition_jacobian @ state.C @ transition_jacobian.T + noise_cov
            )
            predicted_states.append(
                GaussianDistribution(
                    predicted_mean,
                    self._regularize_covariance(predicted_cov),
                    check_validity=False,
                )
            )

        self._component_states = predicted_states
        self._update_collapsed_state()
        return self

    def _linear_update(
        self,
        measurement,
        measurement_matrices,
        measurement_noises,
        *,
        predicted_measurements=None,
        return_log_marginal: bool = False,
    ):
        measurement = atleast_1d(array(measurement))
        updated_states = []
        raw_log_weights = zeros((self.n_components,))
        identity = eye(self.state_dim)

        for component_index in range(self.n_components):
            state = self._component_states[component_index]
            measurement_matrix = measurement_matrices[component_index]
            meas_noise = measurement_noises[component_index]

            expected_measurement = (
                measurement_matrix @ state.mu
                if predicted_measurements is None
                else predicted_measurements[component_index]
            )
            innovation = measurement - expected_measurement
            innovation_cov = (
                measurement_matrix @ state.C @ measurement_matrix.T + meas_noise
            )
            innovation_cov = self._regularize_covariance(innovation_cov)

            kalman_gain = (
                state.C
                @ measurement_matrix.T
                @ linalg.solve(innovation_cov, eye(innovation_cov.shape[0]))
            )

            updated_mean = state.mu + kalman_gain @ innovation
            i_minus_kh = identity - kalman_gain @ measurement_matrix
            updated_cov = (
                i_minus_kh @ state.C @ i_minus_kh.T
                + kalman_gain @ meas_noise @ kalman_gain.T
            )
            updated_cov = self._regularize_covariance(updated_cov)

            updated_states.append(
                GaussianDistribution(
                    updated_mean,
                    updated_cov,
                    check_validity=False,
                )
            )

            innovation_log_density = self._log_gaussian_density(
                innovation, innovation_cov
            )
            raw_log_weights[component_index] = (
                log(
                    maximum(self._component_weights[component_index], self.weight_floor)
                )
                + innovation_log_density
            )

        self._component_states = updated_states
        self._last_update_log_marginal = self._logsumexp(raw_log_weights)
        self._component_weights = self._normalize_log_weights(raw_log_weights)
        self._update_collapsed_state()

        if return_log_marginal:
            return self._last_update_log_marginal
        return self

    def _association_log_likelihood_linear(
        self,
        measurement,
        measurement_matrices,
        measurement_noises,
        predicted_measurements=None,
    ):
        measurement = atleast_1d(array(measurement))
        raw_log_weights = zeros((self.n_components,))

        for component_index in range(self.n_components):
            state = self._component_states[component_index]
            measurement_matrix = measurement_matrices[component_index]
            meas_noise = measurement_noises[component_index]
            expected_measurement = (
                measurement_matrix @ state.mu
                if predicted_measurements is None
                else predicted_measurements[component_index]
            )
            innovation = measurement - expected_measurement
            innovation_cov = (
                measurement_matrix @ state.C @ measurement_matrix.T + meas_noise
            )
            innovation_cov = self._regularize_covariance(innovation_cov)

            raw_log_weights[component_index] = log(
                maximum(self._component_weights[component_index], self.weight_floor)
            ) + self._log_gaussian_density(innovation, innovation_cov)

        return self._logsumexp(raw_log_weights)

    def _log_gaussian_density(self, innovation, covariance):
        dim = innovation.shape[0]
        solved = linalg.solve(covariance, innovation)
        quadratic_form = innovation.T @ solved
        det_covariance = maximum(linalg.det(covariance), self.weight_floor)
        return -0.5 * (dim * log(2.0 * pi) + log(det_covariance) + quadratic_form)

    @staticmethod
    def _looks_like_measurement_distribution(value) -> bool:
        return hasattr(value, "dim") and (
            hasattr(value, "sample") or hasattr(value, "pdf")
        )

    def _resolve_measurement_and_noise_args(self, measurement, meas_noise):
        if self._looks_like_measurement_distribution(
            measurement
        ) and not self._looks_like_measurement_distribution(meas_noise):
            return meas_noise, measurement
        return measurement, meas_noise

    def update_identity(
        self, measurement, meas_noise, return_log_marginal: bool = False
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.state_dim,
            "meas_noise",
        )
        measurement_matrix = eye(self.state_dim)
        return self.update_linear(
            measurement,
            measurement_matrix,
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_position(
        self,
        measurement,
        meas_noise,
        return_log_marginal: bool = False,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        measurement = atleast_1d(array(measurement))
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.position_dim,
            "meas_noise",
        )
        measurement_matrix = concatenate(
            [eye(self.position_dim), zeros((self.position_dim, self.position_dim))],
            axis=1,
        )
        return self.update_linear(
            measurement,
            measurement_matrix,
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_position_measurement(
        self,
        measurement,
        meas_noise,
        return_log_marginal: bool = False,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        return self.update_position(
            measurement=measurement,
            meas_noise=meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_linear(
        self,
        measurement,
        measurement_matrix,
        meas_noise,
        return_log_marginal: bool = False,
    ):
        measurement_matrix = array(measurement_matrix)
        if measurement_matrix.ndim != 2:
            raise ValueError("measurement_matrix must be two-dimensional")
        if measurement_matrix.shape[1] != self.state_dim:
            raise ValueError(f"measurement_matrix must have {self.state_dim} columns")

        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            measurement_matrix.shape[0],
            "meas_noise",
        )

        measurement_matrices = [measurement_matrix for _ in range(self.n_components)]
        measurement_noises = [meas_noise for _ in range(self.n_components)]
        return self._linear_update(
            measurement,
            measurement_matrices,
            measurement_noises,
            return_log_marginal=return_log_marginal,
        )

    def update_nonlinear(
        self,
        measurement,
        hx: Callable,
        cov_mat_meas,
        jacobian: Callable | None = None,
        return_log_marginal: bool = False,
        **hx_args,
    ):
        """Update with a nonlinear measurement function via EKF linearization."""
        measurement = atleast_1d(array(measurement))
        measurement_dim = measurement.shape[0]
        cov_mat_meas = self._coerce_measurement_noise_cov(
            cov_mat_meas,
            measurement_dim,
            "cov_mat_meas",
        )

        measurement_matrices = []
        predicted_measurements = []

        for component_index in range(self.n_components):
            state = self._component_states[component_index]

            if jacobian is None:
                measurement_jacobian, predicted_measurement = self._numerical_jacobian(
                    lambda x: hx(x, **hx_args),
                    state.mu,
                )
            else:
                predicted_measurement = atleast_1d(array(hx(state.mu, **hx_args)))
                measurement_jacobian = array(jacobian(state.mu, **hx_args))

            if predicted_measurement.shape != (measurement_dim,):
                raise ValueError(
                    "The nonlinear measurement function returned an unexpected "
                    "measurement dimension"
                )
            if measurement_jacobian.shape != (measurement_dim, self.state_dim):
                raise ValueError(
                    "The nonlinear measurement Jacobian has an unexpected shape"
                )

            measurement_matrices.append(measurement_jacobian)
            predicted_measurements.append(predicted_measurement)

        measurement_noises = [cov_mat_meas for _ in range(self.n_components)]
        return self._linear_update(
            measurement,
            measurement_matrices,
            measurement_noises,
            predicted_measurements=predicted_measurements,
            return_log_marginal=return_log_marginal,
        )

    def update_velocity(
        self,
        measurement,
        meas_noise,
        return_log_marginal: bool = False,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        measurement = atleast_1d(array(measurement))
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.position_dim,
            "meas_noise",
        )
        measurement_matrix = concatenate(
            [zeros((self.position_dim, self.position_dim)), eye(self.position_dim)],
            axis=1,
        )
        return self.update_linear(
            measurement,
            measurement_matrix,
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_goal(
        self,
        measurement,
        meas_noise,
        return_log_marginal: bool = False,
    ):
        """Reweight discrete goal hypotheses from a measurement in goal space.

        This does not alter the continuous component states. It only updates the
        posterior weights over discrete goal/mode hypotheses.
        """
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        measurement = atleast_1d(array(measurement))
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.position_dim,
            "meas_noise",
        )
        log_likelihoods = zeros((self.n_components,))

        for component_index in range(self.n_components):
            goal_index, _mode_index = self._component_meta(component_index)
            innovation = measurement - self._candidate_goals[goal_index]
            log_likelihoods[component_index] = self._log_gaussian_density(
                innovation,
                meas_noise,
            )

        return self.reweight_components(
            log_likelihoods=log_likelihoods,
            return_log_marginal=return_log_marginal,
        )

    def association_likelihood_identity(self, measurement, meas_noise):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.state_dim,
            "meas_noise",
        )
        measurement_matrix = eye(self.state_dim)
        return self.association_likelihood_linear(
            measurement, measurement_matrix, meas_noise
        )

    def association_likelihood_linear(
        self, measurement, measurement_matrix, meas_noise
    ):
        measurement_matrix = array(measurement_matrix)
        if measurement_matrix.ndim != 2:
            raise ValueError("measurement_matrix must be two-dimensional")
        if measurement_matrix.shape[1] != self.state_dim:
            raise ValueError(f"measurement_matrix must have {self.state_dim} columns")

        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            measurement_matrix.shape[0],
            "meas_noise",
        )
        measurement_matrices = [measurement_matrix for _ in range(self.n_components)]
        measurement_noises = [meas_noise for _ in range(self.n_components)]
        return exp(
            self._association_log_likelihood_linear(
                measurement, measurement_matrices, measurement_noises
            )
        )

    def association_likelihood_position(self, measurement, meas_noise):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.position_dim,
            "meas_noise",
        )
        measurement_matrix = concatenate(
            [eye(self.position_dim), zeros((self.position_dim, self.position_dim))],
            axis=1,
        )
        return self.association_likelihood_linear(
            measurement, measurement_matrix, meas_noise
        )

    def association_likelihood_velocity(self, measurement, meas_noise):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.position_dim,
            "meas_noise",
        )
        measurement_matrix = concatenate(
            [zeros((self.position_dim, self.position_dim)), eye(self.position_dim)],
            axis=1,
        )
        return self.association_likelihood_linear(
            measurement, measurement_matrix, meas_noise
        )

    def association_likelihood_goal(self, measurement, meas_noise):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        measurement = atleast_1d(array(measurement))
        meas_noise = self._coerce_measurement_noise_cov(
            meas_noise,
            self.position_dim,
            "meas_noise",
        )

        raw_log_weights = zeros((self.n_components,))
        for component_index in range(self.n_components):
            goal_index, _mode_index = self._component_meta(component_index)
            innovation = measurement - self._candidate_goals[goal_index]
            raw_log_weights[component_index] = log(
                maximum(self._component_weights[component_index], self.weight_floor)
            ) + self._log_gaussian_density(innovation, meas_noise)

        return exp(self._logsumexp(raw_log_weights))
