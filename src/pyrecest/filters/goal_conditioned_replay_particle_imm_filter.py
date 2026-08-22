"""Goal-conditioned replay particle filter with per-particle IMM modes.

The filter extends :class:`GoalConditionedReplayParticleFilter` by attaching a
discrete motion mode to each particle.  The continuous state remains
``x = [z, v, g]`` with position ``z``, velocity ``v``, and latent goal ``g``.
The discrete mode follows a Markov transition matrix and selects one of several
replay dynamics:

- ``stationary``: remain near the current position,
- ``diffusion``: local random-walk-like motion,
- ``momentum``: velocity-persistent motion,
- ``goal_directed``: acceleration toward the latent goal, and
- ``jump``: rapid displacement toward the latent goal.
"""

from __future__ import annotations

import math
from numbers import Integral

# pylint: disable=no-name-in-module,no-member,duplicate-code
from pyrecest import copy
from pyrecest.backend import (
    arange,
    argmax,
    array,
    atleast_1d,
    concatenate,
    ndim,
    ones,
    ones_like,
    random,
    reshape,
    sum,
    where,
    zeros,
)

from .goal_conditioned_replay_particle_filter import GoalConditionedReplayParticleFilter


def _as_mode_index(value, n_modes: int) -> int:
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    except (RuntimeError, ValueError) as exc:
        raise ValueError("mode indices must be scalar integers") from exc

    if isinstance(scalar, bool) or isinstance(scalar, str):
        raise ValueError("mode indices must be integers")
    if isinstance(scalar, Integral):
        mode_index = int(scalar)
    else:
        try:
            scalar_float = float(scalar)
            mode_index = int(scalar)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("mode indices must be integers") from exc
        if not math.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError("mode indices must be finite integers")

    if mode_index < 0 or mode_index >= n_modes:
        raise ValueError(f"mode_indices must lie in [0, {n_modes - 1}]")
    return mode_index


def _contains_boolean(value) -> bool:
    if isinstance(value, bool):
        return True
    try:
        scalar = value.item()
    except (AttributeError, RuntimeError, ValueError):
        scalar = None
    if isinstance(scalar, bool):
        return True
    if isinstance(value, str):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_contains_boolean(item) for item in iterator)


class GoalConditionedReplayParticleIMMFilter(  # pylint: disable=too-many-instance-attributes
    GoalConditionedReplayParticleFilter
):
    """Goal-conditioned particle filter with IMM-style motion-mode switching.

    The mode variable is represented by one discrete mode index per particle and
    is resampled together with the continuous particles during measurement
    updates.  This keeps mode posteriors usable after likelihood updates while
    retaining the particle-filter interface used by
    :class:`GoalConditionedReplayParticleFilter`.
    """

    mode_names = ("stationary", "diffusion", "momentum", "goal_directed", "jump")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_particles,
        position_dim=None,
        initial_state=None,
        *,
        spatial_dim=None,
        mode_transition_matrix=None,
        mode_prior=None,
        mode_stickiness: float = 0.95,
        stationary_velocity_decay: float = 0.0,
        diffusion_velocity_decay: float = 0.0,
        momentum_velocity_decay: float = 0.95,
        jump_fraction: float = 0.9,
        jump_velocity_decay: float = 0.25,
        **kwargs,
    ):
        self.n_modes = len(self.mode_names)
        self.mode_transition_matrix = self._prepare_mode_transition_matrix(
            mode_transition_matrix,
            mode_stickiness,
        )
        self.mode_prior = self._validate_probability_vector(
            ones((self.n_modes,)) / self.n_modes if mode_prior is None else mode_prior,
            self.n_modes,
            "mode_prior",
        )
        self.stationary_velocity_decay = float(stationary_velocity_decay)
        self.diffusion_velocity_decay = float(diffusion_velocity_decay)
        self.momentum_velocity_decay = float(momentum_velocity_decay)
        self.jump_fraction = float(jump_fraction)
        self.jump_velocity_decay = float(jump_velocity_decay)
        self._mode_indices = None
        self._last_mode_transition_mask = None

        super().__init__(
            n_particles=n_particles,
            position_dim=position_dim,
            initial_state=initial_state,
            spatial_dim=spatial_dim,
            **kwargs,
        )
        self.sample_modes_from_prior(self.mode_prior)
        self._last_mode_transition_mask = zeros((self.n_particles,))

    @property
    def mode_indices(self):
        """Current discrete mode index for each particle."""

        if self._mode_indices is None:
            return None
        return copy(self._mode_indices)

    @property
    def mode_probabilities(self):
        """Weighted posterior probability for each motion mode."""

        probabilities = zeros((self.n_modes,))
        for mode_index in range(self.n_modes):
            probabilities[mode_index] = sum(
                self.filter_state.w[self._mode_indices == mode_index]
            )
        total = sum(probabilities)
        if self._to_scalar(total) <= 0.0:
            raise ValueError("Mode probabilities are degenerate")
        return probabilities / total

    @property
    def last_mode_transition_fraction(self) -> float:
        if self._last_mode_transition_mask is None:
            return 0.0
        return float(sum(self._last_mode_transition_mask) / self.n_particles)

    def most_likely_mode_index(self) -> int:
        return int(argmax(self.mode_probabilities))

    def most_likely_mode(self) -> str:
        return self.mode_names[self.most_likely_mode_index()]

    def set_mode_indices(self, mode_indices):
        """Set one discrete mode index per particle."""

        if _contains_boolean(mode_indices):
            raise ValueError("mode_indices must be integers, not booleans")
        try:
            mode_indices = atleast_1d(array(mode_indices))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("mode_indices must be integers") from exc
        if mode_indices.shape != (self.n_particles,):
            raise ValueError(
                "mode_indices must contain one mode index per particle; "
                f"got {mode_indices.shape}"
            )
        self._mode_indices = array(
            [_as_mode_index(mode_index, self.n_modes) for mode_index in mode_indices]
        )
        return self

    def sample_modes_from_prior(self, mode_prior=None):
        """Sample particle mode indices from a prior probability vector."""

        if mode_prior is None:
            mode_prior = self.mode_prior
        mode_prior = self._validate_probability_vector(
            mode_prior,
            self.n_modes,
            "mode_prior",
        )
        self._mode_indices = random.choice(
            arange(self.n_modes),
            self.n_particles,
            p=mode_prior,
        )
        return self

    def _apply_likelihood_values(
        self,
        likelihood_values,
        *,
        return_log_marginal: bool = False,
    ):
        likelihood_values = self._ensure_likelihood_vector(likelihood_values)
        reweighted = likelihood_values * self.filter_state.w
        marginal = sum(reweighted)
        if self._to_scalar(marginal) <= 0.0:
            raise ValueError("Likelihood collapsed to zero for all particles")

        normalized_weights = reweighted / marginal
        indices = random.choice(
            arange(self.n_particles),
            self.n_particles,
            p=normalized_weights,
        )
        new_state = type(self.filter_state)(
            copy(self.filter_state.d[indices]),
            ones_like(self.filter_state.w) / self.n_particles,
        )
        if self._mode_indices is None:
            raise ValueError("Mode indices must be initialized before updating")
        self.filter_state = new_state
        self._mode_indices = self._mode_indices[indices]
        self._last_update_log_marginal = self._safe_log(marginal)

        if return_log_marginal:
            return self._last_update_log_marginal
        return self

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def predict_replay(
        self,
        dt: float | None = None,
        alpha=None,
        beta=None,
        attraction_field=None,
        goal_transition=None,
        process_noise=None,
        goal_noise=None,
        jump_probability: float | None = None,
        jump_distribution=None,
        position_jump_distribution=None,
        goal_reset_probability: float | None = None,
        goal_reset_distribution=None,
        attraction_field_is_vectorized: bool | None = None,
        gradient_is_vectorized: bool | None = None,
        function_is_vectorized: bool | None = None,
        use_semi_implicit_position_update: bool = False,
        *,
        mode_transition_matrix=None,
        stationary_velocity_decay: float | None = None,
        diffusion_velocity_decay: float | None = None,
        momentum_velocity_decay: float | None = None,
        jump_fraction: float | None = None,
        jump_velocity_decay: float | None = None,
    ):
        """Predict one replay step after Markov switching the particle modes."""

        dt = self.dt if dt is None else self._validate_dt(dt)
        alpha = self.alpha if alpha is None else alpha
        beta = self.beta if beta is None else beta
        attraction_field = (
            self.attraction_field if attraction_field is None else attraction_field
        )
        goal_transition = (
            self.goal_transition if goal_transition is None else goal_transition
        )
        process_noise = self.process_noise if process_noise is None else process_noise
        goal_noise = self.goal_noise if goal_noise is None else goal_noise
        jump_probability = (
            self.jump_probability
            if jump_probability is None
            else float(jump_probability)
        )
        jump_distribution = (
            self.jump_distribution if jump_distribution is None else jump_distribution
        )
        position_jump_distribution = (
            self.position_jump_distribution
            if position_jump_distribution is None
            else position_jump_distribution
        )
        goal_reset_probability = (
            self.goal_reset_probability
            if goal_reset_probability is None
            else float(goal_reset_probability)
        )
        goal_reset_distribution = (
            self.goal_reset_distribution
            if goal_reset_distribution is None
            else goal_reset_distribution
        )
        transition_matrix = (
            self.mode_transition_matrix
            if mode_transition_matrix is None
            else self._validate_transition_matrix(
                mode_transition_matrix,
                self.n_modes,
                "mode_transition_matrix",
            )
        )
        stationary_velocity_decay = (
            self.stationary_velocity_decay
            if stationary_velocity_decay is None
            else float(stationary_velocity_decay)
        )
        diffusion_velocity_decay = (
            self.diffusion_velocity_decay
            if diffusion_velocity_decay is None
            else float(diffusion_velocity_decay)
        )
        momentum_velocity_decay = (
            self.momentum_velocity_decay
            if momentum_velocity_decay is None
            else float(momentum_velocity_decay)
        )
        jump_fraction = (
            self.jump_fraction if jump_fraction is None else float(jump_fraction)
        )
        jump_velocity_decay = (
            self.jump_velocity_decay
            if jump_velocity_decay is None
            else float(jump_velocity_decay)
        )

        if function_is_vectorized is not None:
            attraction_field_is_vectorized = function_is_vectorized
        if gradient_is_vectorized is not None:
            attraction_field_is_vectorized = gradient_is_vectorized

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._validate_probability(jump_probability, "jump_probability")
        self._validate_probability(goal_reset_probability, "goal_reset_probability")
        self._validate_component_distribution(process_noise, "process_noise")
        self._validate_component_distribution(goal_noise, "goal_noise")
        self._validate_component_distribution(jump_distribution, "jump_distribution")
        self._validate_component_distribution(
            position_jump_distribution, "position_jump_distribution"
        )
        self._validate_component_distribution(
            goal_reset_distribution, "goal_reset_distribution"
        )

        previous_mode_indices = copy(self._mode_indices)
        self._mode_indices = self._sample_transitioned_modes(transition_matrix)
        self._last_mode_transition_mask = array(
            self._mode_indices != previous_mode_indices
        )

        particles = self.filter_state.d
        positions = particles[:, self.position_slice]
        velocities = particles[:, self.velocity_slice]
        current_goals = particles[:, self.goal_slice]

        goal_reset_mask = self._draw_bernoulli_mask(goal_reset_probability)
        if self._to_scalar(sum(goal_reset_mask)) > 0.0:
            reset_goals = self._sample_goal_reset_targets(
                goal_reset_distribution,
                self.n_particles,
            )
            dynamic_goals = where(
                reshape(goal_reset_mask, (-1, 1)) > 0,
                reset_goals,
                current_goals,
            )
        else:
            dynamic_goals = current_goals

        goals_for_dynamics = self._apply_goal_transition(
            dynamic_goals,
            positions,
            velocities,
            dt,
            goal_transition,
        )
        goal_noise_samples = self._sample_zero_or_distribution(
            goal_noise,
            n_samples=self.n_particles,
            name="goal_noise",
        )
        goals_new = goals_for_dynamics + goal_noise_samples

        control_field = self._evaluate_attraction_field(
            positions,
            goals_for_dynamics,
            attraction_field,
            attraction_field_is_vectorized,
        )
        process_noise_samples = self._sample_zero_or_distribution(
            process_noise,
            n_samples=self.n_particles,
            name="process_noise",
        )

        jump_mode_mask = array(self._mode_indices == self._mode_index("jump"))
        sparse_jump_mask = self._draw_bernoulli_mask(jump_probability)
        jump_mask = where(
            reshape(jump_mode_mask, (-1,)) + reshape(sparse_jump_mask, (-1,)) > 0,
            ones((self.n_particles,)),
            zeros((self.n_particles,)),
        )
        velocity_jump = self._sample_sparse_vectors(
            jump_distribution,
            jump_mask,
            name="jump_distribution",
        )
        position_jump = self._sample_sparse_vectors(
            position_jump_distribution,
            jump_mask,
            name="position_jump_distribution",
        )

        stationary_velocity = (
            stationary_velocity_decay * velocities
            + process_noise_samples
            + velocity_jump
        )
        stationary_position = positions + position_jump

        diffusion_velocity = (
            diffusion_velocity_decay * velocities
            + process_noise_samples
            + velocity_jump
        )
        diffusion_position = (
            positions
            + dt
            * self._position_velocity(
                velocities,
                diffusion_velocity,
                use_semi_implicit_position_update,
            )
            + position_jump
        )

        momentum_velocity = (
            momentum_velocity_decay * velocities + process_noise_samples + velocity_jump
        )
        momentum_position = (
            positions
            + dt
            * self._position_velocity(
                velocities,
                momentum_velocity,
                use_semi_implicit_position_update,
            )
            + position_jump
        )

        goal_velocity = (
            self._apply_linear_coefficient(alpha, velocities)
            + self._apply_linear_coefficient(beta, control_field)
            + process_noise_samples
            + velocity_jump
        )
        goal_position = (
            positions
            + dt
            * self._position_velocity(
                velocities,
                goal_velocity,
                use_semi_implicit_position_update,
            )
            + position_jump
        )

        jump_velocity = (
            jump_velocity_decay * velocities + process_noise_samples + velocity_jump
        )
        jump_position = (
            positions + jump_fraction * (goals_for_dynamics - positions) + position_jump
        )

        positions_new = zeros(positions.shape)
        velocities_new = zeros(velocities.shape)
        positions_new, velocities_new = self._select_mode_dynamics(
            positions_new,
            velocities_new,
            "stationary",
            stationary_position,
            stationary_velocity,
        )
        positions_new, velocities_new = self._select_mode_dynamics(
            positions_new,
            velocities_new,
            "diffusion",
            diffusion_position,
            diffusion_velocity,
        )
        positions_new, velocities_new = self._select_mode_dynamics(
            positions_new,
            velocities_new,
            "momentum",
            momentum_position,
            momentum_velocity,
        )
        positions_new, velocities_new = self._select_mode_dynamics(
            positions_new,
            velocities_new,
            "goal_directed",
            goal_position,
            goal_velocity,
        )
        positions_new, velocities_new = self._select_mode_dynamics(
            positions_new,
            velocities_new,
            "jump",
            jump_position,
            jump_velocity,
        )

        new_state = type(self.filter_state)(
            concatenate(
                [positions_new, velocities_new, goals_new],
                axis=1,
            ),
            copy(self.filter_state.w),
        )
        self.filter_state = new_state
        self._last_transition_diagnostics = {
            "jump_mask": jump_mask,
            "jump_mode_mask": jump_mode_mask,
            "sparse_jump_mask": sparse_jump_mask,
            "goal_reset_mask": goal_reset_mask,
            "mode_transition_mask": self._last_mode_transition_mask,
            "previous_mode_indices": previous_mode_indices,
            "mode_indices": copy(self._mode_indices),
            "velocity_jump": velocity_jump,
            "position_jump": position_jump,
            "position_proposal_mask": zeros((self.n_particles,)),
            "position_proposal_samples": zeros((self.n_particles, self.position_dim)),
            "control_field": control_field,
            "dynamic_goals": dynamic_goals,
            "goals_for_dynamics": goals_for_dynamics,
        }
        return self

    def _sample_transitioned_modes(self, transition_matrix):
        mode_indices = self._mode_indices
        if mode_indices is None:
            raise ValueError("Mode indices must be initialized before prediction")
        transitioned = copy(mode_indices)
        for source_mode in range(self.n_modes):
            source_mask = mode_indices == source_mode
            count = int(self._to_scalar(sum(source_mask)))
            if count == 0:
                continue
            transitioned[source_mask] = random.choice(
                arange(self.n_modes),
                count,
                p=transition_matrix[source_mode],
            )
        return transitioned

    def _select_mode_dynamics(
        self,
        positions_new,
        velocities_new,
        mode_name: str,
        candidate_positions,
        candidate_velocities,
    ):
        mode_mask = reshape(
            self._mode_indices == self._mode_index(mode_name),
            (-1, 1),
        )
        positions_new = where(mode_mask, candidate_positions, positions_new)
        velocities_new = where(mode_mask, candidate_velocities, velocities_new)
        return positions_new, velocities_new

    def _mode_index(self, mode_name: str) -> int:
        return self.mode_names.index(mode_name)

    @staticmethod
    def _position_velocity(
        old_velocities,
        new_velocities,
        use_semi_implicit_position_update: bool,
    ):
        return new_velocities if use_semi_implicit_position_update else old_velocities

    def _prepare_mode_transition_matrix(
        self,
        mode_transition_matrix,
        mode_stickiness: float,
    ):
        if mode_transition_matrix is not None:
            return self._validate_transition_matrix(
                mode_transition_matrix,
                self.n_modes,
                "mode_transition_matrix",
            )
        if not 0.0 <= mode_stickiness <= 1.0:
            raise ValueError("mode_stickiness must lie in [0, 1]")
        if self.n_modes == 1:
            return ones((1, 1))
        off_diagonal = (1.0 - mode_stickiness) / (self.n_modes - 1)
        transition_matrix = ones((self.n_modes, self.n_modes)) * off_diagonal
        for mode_index in range(self.n_modes):
            transition_matrix[mode_index, mode_index] = mode_stickiness
        return transition_matrix

    @staticmethod
    def _validate_probability_vector(vector, expected_size: int, name: str):
        vector = atleast_1d(array(vector))
        if vector.shape != (expected_size,):
            raise ValueError(f"{name} must have shape ({expected_size},)")
        if float(sum(vector)) <= 0.0:
            raise ValueError(f"{name} must have positive total mass")
        for value in vector:
            if float(value) < 0.0:
                raise ValueError(f"{name} must not contain negative entries")
        return vector / sum(vector)

    @staticmethod
    def _validate_transition_matrix(matrix, expected_size: int, name: str):
        matrix = array(matrix)
        if ndim(matrix) != 2 or matrix.shape != (expected_size, expected_size):
            raise ValueError(
                f"{name} must have shape ({expected_size}, {expected_size})"
            )
        for row_index in range(expected_size):
            row = matrix[row_index]
            if abs(float(sum(row)) - 1.0) > 1e-10:
                raise ValueError(f"Each row of {name} must sum to one")
            for value in row:
                if float(value) < 0.0:
                    raise ValueError(f"{name} must not contain negative entries")
        return matrix

    @staticmethod
    def _safe_log(value):
        from pyrecest.backend import log  # pylint: disable=import-outside-toplevel

        return log(value)
