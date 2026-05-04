"""Goal-conditioned replay particle filter with sparse jumps and goal remapping.

This version consolidates the strongest ideas from the earlier particle-filter
drafts:

- continuous latent state ``x = [z, v, g]`` with ``z``, ``v``, ``g`` in
  ``R^position_dim``,
- explicit sparse velocity jumps and optional direct position jumps,
- optional goal resets / remapping for piecewise replay trajectories,
- factorized initialization from separate position / velocity / goal priors,
- optional candidate-goal banks with empirical posterior weights,
- replay-oriented diagnostics and marginal-likelihood access, and
- a public interface aligned with ``GoalConditionedReplayIMMFilter``.

The built-in prediction step implements

    z_{t+1} = z_t + dt * v_t + delta_pos_t
    v_{t+1} = alpha * v_t + beta * a(z_t, g_t) + eps_t + delta_vel_t
    g_{t+1} = T_g(g_t, z_t, v_t, dt) + eta_t

where ``a`` is an attraction field, ``T_g`` is an optional deterministic goal
transition, ``eps_t`` is dense process noise, and ``delta_*`` are
Bernoulli-thinned sparse jumps.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin,duplicate-code
from pyrecest.backend import (
    arange,
    argmin,
    array,
    atleast_1d,
    concatenate,
    eye,
    int32,
    int64,
    log,
    matmul,
    ndim,
    ones,
    ones_like,
    random,
    reshape,
    squeeze,
    sum,
    vstack,
    where,
    zeros,
)
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from pyrecest.distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)

from .euclidean_particle_filter import EuclideanParticleFilter


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class GoalConditionedReplayParticleFilter(EuclideanParticleFilter):
    """Particle filter for goal-conditioned replay with sparse jumps.

    Parameters
    ----------
    n_particles:
        Number of particles.
    position_dim:
        Dimension of replayed position, velocity, and latent goal.
    initial_state:
        Optional prior over the full augmented state ``[z, v, g]``. Supported
        inputs are a ``LinearDiracDistribution``, any linear distribution with
        matching dimension, or a tuple ``(mean, covariance)`` which is converted
        to ``GaussianDistribution`` and sampled.
    dt, alpha, beta:
        Parameters of the default replay transition.
    attraction_field:
        Callable implementing the goal-conditioned control field. It may be
        vectorized across particles or evaluated particle by particle. If
        omitted, the default field is ``g - z``.
    goal_transition:
        Optional deterministic transition for goals. It may accept one of the
        signatures ``goal_transition(goals)``,
        ``goal_transition(goals, positions, velocities)``, or
        ``goal_transition(goals, positions, velocities, dt)``.
    process_noise:
        Dense process noise added in velocity space.
    goal_noise:
        Additive noise applied after the deterministic goal transition.
    jump_probability:
        Bernoulli probability of a sparse jump for each particle and prediction
        step.
    jump_distribution:
        Distribution for sparse velocity jumps.
    position_jump_distribution:
        Distribution for sparse direct position jumps.
    goal_reset_probability:
        Bernoulli probability of resetting / remapping the latent goal before
        the control field is evaluated.
    goal_reset_distribution:
        Distribution from which reset goals are drawn. If omitted and a
        candidate-goal bank is stored, resets are sampled from that bank.
    candidate_goals, candidate_goal_weights:
        Optional discrete goal bank used for initialization, goal resets, and
        posterior summaries.
    initial_position_distribution, initial_velocity_distribution,
    initial_goal_distribution:
        Optional factorized priors used when ``initial_state`` is omitted.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
    def __init__(
        self,
        n_particles: int | int32 | int64,
        position_dim: int | int32 | int64 | None = None,
        initial_state: (
            AbstractLinearDistribution
            | LinearDiracDistribution
            | tuple[Any, Any]
            | None
        ) = None,
        *,
        spatial_dim: int | int32 | int64 | None = None,
        dt: float = 1.0,
        alpha: float | Any = 0.95,
        beta: float | Any = 1.0,
        attraction_field: Callable | None = None,
        goal_transition: Callable | None = None,
        process_noise: AbstractLinearDistribution | None = None,
        goal_noise: AbstractLinearDistribution | None = None,
        jump_probability: float = 0.0,
        jump_distribution: AbstractLinearDistribution | None = None,
        position_jump_distribution: AbstractLinearDistribution | None = None,
        goal_reset_probability: float = 0.0,
        goal_reset_distribution: AbstractLinearDistribution | None = None,
        candidate_goals=None,
        candidate_goal_weights=None,
        initial_position_distribution: AbstractLinearDistribution | None = None,
        initial_velocity_distribution: AbstractLinearDistribution | None = None,
        initial_goal_distribution: AbstractLinearDistribution | None = None,
    ):
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError(
                "GoalConditionedReplayParticleFilter is not supported on the JAX backend."
            )

        if position_dim is None:
            position_dim = spatial_dim
        elif spatial_dim is not None and int(spatial_dim) != int(position_dim):
            raise ValueError(
                "position_dim and spatial_dim must match when both are provided"
            )

        if position_dim is None:
            raise ValueError("Either position_dim or spatial_dim must be provided")

        try:
            position_dim = int(position_dim)
            n_particles = int(n_particles)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "position_dim/spatial_dim and n_particles must be positive integers"
            ) from exc

        if position_dim <= 0:
            raise ValueError("position_dim/spatial_dim must be a positive integer")
        if n_particles <= 0:
            raise ValueError("n_particles must be a positive integer")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.position_dim = position_dim
        self.spatial_dim = position_dim
        self.state_dim = 3 * position_dim

        self.dt = float(dt)
        self.alpha = alpha
        self.beta = beta
        self.attraction_field = (
            attraction_field
            if attraction_field is not None
            else self._default_attraction_field
        )
        self.goal_transition = goal_transition

        self.process_noise = process_noise
        self.goal_noise = goal_noise
        self.jump_probability = float(jump_probability)
        self.jump_distribution = jump_distribution
        self.position_jump_distribution = position_jump_distribution
        self.goal_reset_probability = float(goal_reset_probability)
        self.goal_reset_distribution = goal_reset_distribution

        self._candidate_goals = None
        self._candidate_goal_weights = None
        self._last_update_log_marginal = None
        self._last_transition_diagnostics: dict[str, Any] = {}

        self._validate_probability(self.jump_probability, "jump_probability")
        self._validate_probability(
            self.goal_reset_probability, "goal_reset_probability"
        )

        self._validate_component_distribution(process_noise, "process_noise")
        self._validate_component_distribution(goal_noise, "goal_noise")
        self._validate_component_distribution(jump_distribution, "jump_distribution")
        self._validate_component_distribution(
            position_jump_distribution, "position_jump_distribution"
        )
        self._validate_component_distribution(
            goal_reset_distribution, "goal_reset_distribution"
        )

        if self.jump_probability > 0.0 and (
            self.jump_distribution is None and self.position_jump_distribution is None
        ):
            raise ValueError(
                "At least one jump distribution must be provided when "
                "jump_probability is positive"
            )

        EuclideanParticleFilter.__init__(
            self,
            n_particles=n_particles,
            dim=self.state_dim,
        )

        if candidate_goals is not None:
            self.set_candidate_goals(candidate_goals, candidate_goal_weights)

        if initial_state is not None:
            self.filter_state = self._coerce_initial_state(initial_state)
        else:
            self.initialize_from_factorized_priors(
                position_distribution=initial_position_distribution,
                velocity_distribution=initial_velocity_distribution,
                goal_distribution=initial_goal_distribution,
                candidate_goals=candidate_goals,
                goal_prior_weights=candidate_goal_weights,
            )

        self._initialize_transition_diagnostics()

    @classmethod
    def from_factorized_priors(
        cls,
        n_particles: int,
        position_dim: int | None = None,
        position_distribution: AbstractLinearDistribution | None = None,
        velocity_distribution: AbstractLinearDistribution | None = None,
        goal_distribution: AbstractLinearDistribution | None = None,
        *,
        spatial_dim: int | None = None,
        candidate_goals=None,
        goal_prior_weights=None,
        **kwargs,
    ) -> "GoalConditionedReplayParticleFilter":
        if position_dim is None:
            position_dim = spatial_dim
        elif spatial_dim is not None and int(spatial_dim) != int(position_dim):
            raise ValueError(
                "position_dim and spatial_dim must match when both are provided"
            )
        replay_filter = cls(
            n_particles=n_particles,
            position_dim=position_dim,
            initial_state=None,
            candidate_goals=None,
            candidate_goal_weights=None,
            **kwargs,
        )
        replay_filter.initialize_from_factorized_priors(
            position_distribution=position_distribution,
            velocity_distribution=velocity_distribution,
            goal_distribution=goal_distribution,
            candidate_goals=candidate_goals,
            goal_prior_weights=goal_prior_weights,
        )
        if candidate_goals is not None:
            replay_filter.set_candidate_goals(candidate_goals, goal_prior_weights)
        return replay_filter

    @property
    def n_particles(self) -> int:
        return self.filter_state.d.shape[0]

    @property
    def position_slice(self) -> slice:
        return slice(0, self.position_dim)

    @property
    def velocity_slice(self) -> slice:
        return slice(self.position_dim, 2 * self.position_dim)

    @property
    def goal_slice(self) -> slice:
        return slice(2 * self.position_dim, 3 * self.position_dim)

    @property
    def position_particles(self):
        return self.filter_state.d[:, self.position_slice]

    @property
    def velocity_particles(self):
        return self.filter_state.d[:, self.velocity_slice]

    @property
    def goal_particles(self):
        return self.filter_state.d[:, self.goal_slice]

    @property
    def candidate_goals(self):
        if self._candidate_goals is None:
            return None
        return array(self._candidate_goals)

    @property
    def last_update_log_marginal(self):
        return self._last_update_log_marginal

    @property
    def last_transition_diagnostics(self):
        return copy.deepcopy(self._last_transition_diagnostics)

    @property
    def last_jump_fraction(self) -> float:
        return float(
            sum(self._last_transition_diagnostics["jump_mask"])
            / self._last_transition_diagnostics["jump_mask"].shape[0]
        )

    @property
    def last_goal_remap_fraction(self) -> float:
        return float(
            sum(self._last_transition_diagnostics["goal_reset_mask"])
            / self._last_transition_diagnostics["goal_reset_mask"].shape[0]
        )

    @property
    def last_position_proposal_fraction(self) -> float:
        proposal_mask = self._last_transition_diagnostics.get("position_proposal_mask")
        if proposal_mask is None:
            return 0.0
        return float(sum(proposal_mask) / proposal_mask.shape[0])

    def _initialize_transition_diagnostics(self):
        self._last_transition_diagnostics = {
            "jump_mask": zeros((self.n_particles,)),
            "goal_reset_mask": zeros((self.n_particles,)),
            "velocity_jump": zeros((self.n_particles, self.position_dim)),
            "position_jump": zeros((self.n_particles, self.position_dim)),
            "position_proposal_mask": zeros((self.n_particles,)),
            "position_proposal_samples": zeros((self.n_particles, self.position_dim)),
            "control_field": zeros((self.n_particles, self.position_dim)),
            "dynamic_goals": zeros((self.n_particles, self.position_dim)),
            "goals_for_dynamics": zeros((self.n_particles, self.position_dim)),
        }

    @staticmethod
    def _default_attraction_field(positions, goals):
        return goals - positions

    @staticmethod
    def _to_scalar(value) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _validate_probability(self, probability: float, name: str):
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"{name} must lie in [0, 1]")

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

    def _validate_component_distribution(
        self,
        distribution: AbstractLinearDistribution | None,
        name: str,
    ):
        if distribution is not None and distribution.dim != self.position_dim:
            raise ValueError(
                f"{name} must have dimension {self.position_dim}; "
                f"got {distribution.dim}"
            )

    def _coerce_initial_state(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            state = GaussianDistribution(state[0], state[1])

        if isinstance(state, LinearDiracDistribution):
            if state.dim != self.state_dim:
                raise ValueError(f"initial_state must have dimension {self.state_dim}")
            if state.d.shape[0] == self.n_particles:
                return state
            samples = state.sample(self.n_particles)
            samples = self._coerce_particle_matrix(
                samples,
                self.state_dim,
                expected_rows=self.n_particles,
                name="initial_state samples",
            )
            return LinearDiracDistribution(samples)

        if isinstance(state, AbstractLinearDistribution):
            if state.dim != self.state_dim:
                raise ValueError(f"initial_state must have dimension {self.state_dim}")
            samples = state.sample(self.n_particles)
            samples = self._coerce_particle_matrix(
                samples,
                self.state_dim,
                expected_rows=self.n_particles,
                name="initial_state samples",
            )
            return LinearDiracDistribution(samples)

        samples = self._coerce_particle_matrix(
            state,
            self.state_dim,
            expected_rows=self.n_particles,
            name="initial_state",
        )
        return LinearDiracDistribution(samples)

    def initialize_from_factorized_priors(
        self,
        position_distribution: AbstractLinearDistribution | None = None,
        velocity_distribution: AbstractLinearDistribution | None = None,
        goal_distribution: AbstractLinearDistribution | None = None,
        *,
        candidate_goals=None,
        goal_prior_weights=None,
    ):
        """Initialize the particle cloud from factorized component priors."""
        positions = self._sample_component_distribution(
            position_distribution,
            dim=self.position_dim,
            n_samples=self.n_particles,
            name="position_distribution",
            default=zeros((self.n_particles, self.position_dim)),
        )

        velocities = self._sample_component_distribution(
            velocity_distribution,
            dim=self.position_dim,
            n_samples=self.n_particles,
            name="velocity_distribution",
            default=zeros((self.n_particles, self.position_dim)),
        )

        if goal_distribution is not None:
            goals = self._sample_component_distribution(
                goal_distribution,
                dim=self.position_dim,
                n_samples=self.n_particles,
                name="goal_distribution",
            )
            if candidate_goals is not None:
                self.set_candidate_goals(candidate_goals, goal_prior_weights)
        elif candidate_goals is not None:
            goals = self._sample_goals_from_candidates(
                candidate_goals,
                goal_prior_weights,
                self.n_particles,
            )
        else:
            goals = zeros((self.n_particles, self.position_dim))

        self.filter_state = LinearDiracDistribution(
            concatenate((positions, velocities, goals), axis=1)
        )
        self._initialize_transition_diagnostics()
        return self

    def initialize_from_state_priors(
        self,
        position_prior=None,
        velocity_prior=None,
        goal_prior=None,
        weights=None,
        *,
        candidate_goals=None,
        goal_prior_weights=None,
    ):
        """Backward-compatible alias for factorized initialization.

        Each prior may be given either as a distribution, a single state vector
        to broadcast to all particles, or an explicit particle matrix.
        """
        self.set_state_components(
            positions=position_prior,
            velocities=velocity_prior,
            goals=goal_prior,
            weights=weights,
        )
        if candidate_goals is not None:
            self.set_candidate_goals(candidate_goals, goal_prior_weights)
            if goal_prior is None:
                self.sample_goals_from_candidates(
                    candidate_goals,
                    goal_prior_weights,
                )
        return self

    def set_state_components(
        self, positions, velocities=None, goals=None, weights=None
    ):
        """Directly set particle states from component samples or distributions."""
        position_samples = self._coerce_component_samples(
            positions, self.n_particles, "positions"
        )
        velocity_samples = self._coerce_component_samples(
            velocities, self.n_particles, "velocities"
        )
        goal_samples = self._coerce_component_samples(goals, self.n_particles, "goals")

        self.filter_state = LinearDiracDistribution(
            concatenate([position_samples, velocity_samples, goal_samples], axis=1),
            weights,
        )
        self._initialize_transition_diagnostics()
        return self

    def _coerce_component_samples(self, component, n_particles: int, name: str):
        if component is None:
            return zeros((n_particles, self.position_dim))

        if isinstance(component, tuple) and len(component) == 2:
            component = GaussianDistribution(component[0], component[1])

        if isinstance(component, AbstractLinearDistribution):
            self._validate_component_distribution(component, name)
            samples = component.sample(n_particles)
            return self._coerce_particle_matrix(
                samples,
                self.position_dim,
                expected_rows=n_particles,
                name=name,
            )

        values = array(component)
        if ndim(values) == 0:
            values = reshape(values, (1,))
        if ndim(values) == 1:
            values = self._coerce_state_vector(values, self.position_dim, name=name)
            return vstack([values for _ in range(n_particles)])

        return self._coerce_particle_matrix(
            values,
            self.position_dim,
            expected_rows=n_particles,
            name=name,
        )

    def set_candidate_goals(self, candidate_goals, goal_prior_weights=None):
        candidate_goals = self._coerce_particle_matrix(
            candidate_goals,
            self.position_dim,
            name="candidate_goals",
        )
        n_goals = candidate_goals.shape[0]
        if n_goals <= 0:
            raise ValueError("candidate_goals must contain at least one goal")

        if goal_prior_weights is None:
            goal_prior_weights = ones((n_goals,)) / n_goals
        else:
            goal_prior_weights = atleast_1d(array(goal_prior_weights))
            if goal_prior_weights.shape != (n_goals,):
                raise ValueError(
                    "goal_prior_weights must have one weight per candidate goal"
                )
            if self._to_scalar(sum(goal_prior_weights)) <= 0.0:
                raise ValueError("goal_prior_weights must have positive total mass")
            if any(self._to_scalar(weight) < 0.0 for weight in goal_prior_weights):
                raise ValueError("goal_prior_weights must be nonnegative")
            goal_prior_weights = goal_prior_weights / sum(goal_prior_weights)

        self._candidate_goals = candidate_goals
        self._candidate_goal_weights = goal_prior_weights
        return self

    def sample_goals_from_candidates(
        self, candidate_goals=None, goal_prior_weights=None
    ):
        """Replace the current goal particles by samples from a goal bank."""
        if candidate_goals is None:
            if self._candidate_goals is None:
                raise ValueError(
                    "candidate_goals must be provided when no goal bank is stored"
                )
            candidate_goals = self._candidate_goals
            if goal_prior_weights is None:
                goal_prior_weights = self._candidate_goal_weights

        goals = self._sample_goals_from_candidates(
            candidate_goals,
            goal_prior_weights,
            self.n_particles,
        )
        self._filter_state.d[:, self.goal_slice] = goals
        return self

    def _sample_goals_from_candidates(
        self,
        candidate_goals,
        goal_prior_weights=None,
        n_samples: int | None = None,
    ):
        candidate_goals = self._coerce_particle_matrix(
            candidate_goals,
            self.position_dim,
            name="candidate_goals",
        )
        n_goals = candidate_goals.shape[0]
        if n_goals <= 0:
            raise ValueError("candidate_goals must contain at least one goal")
        if goal_prior_weights is None:
            goal_prior_weights = ones((n_goals,)) / n_goals
        else:
            goal_prior_weights = atleast_1d(array(goal_prior_weights))
            if goal_prior_weights.shape != (n_goals,):
                raise ValueError(
                    "goal_prior_weights must have one weight per candidate goal"
                )
            if self._to_scalar(sum(goal_prior_weights)) <= 0.0:
                raise ValueError("goal_prior_weights must have positive total mass")
            goal_prior_weights = goal_prior_weights / sum(goal_prior_weights)

        indices = random.choice(
            arange(n_goals),
            self.n_particles if n_samples is None else n_samples,
            p=goal_prior_weights,
        )
        self._candidate_goals = candidate_goals
        self._candidate_goal_weights = goal_prior_weights
        return candidate_goals[indices]

    def _sample_position_proposal(
        self,
        position_proposal,
        proposal_weights=None,
        *,
        n_samples: int,
    ):
        if isinstance(position_proposal, tuple) and len(position_proposal) == 2:
            position_proposal = GaussianDistribution(
                position_proposal[0],
                position_proposal[1],
            )

        if isinstance(position_proposal, AbstractLinearDistribution):
            if proposal_weights is not None:
                raise ValueError(
                    "proposal_weights are only supported for discrete proposals"
                )
            self._validate_component_distribution(
                position_proposal,
                "position_proposal",
            )
            return self._sample_component_distribution(
                position_proposal,
                dim=self.position_dim,
                n_samples=n_samples,
                name="position_proposal",
            )

        proposal_positions = self._coerce_particle_matrix(
            position_proposal,
            self.position_dim,
            name="position_proposal",
        )
        n_positions = proposal_positions.shape[0]
        if n_positions <= 0:
            raise ValueError("position_proposal must contain at least one position")
        if proposal_weights is None:
            proposal_weights = ones((n_positions,)) / n_positions
        proposal_weights = self._validate_probability_vector(
            proposal_weights,
            n_positions,
            "proposal_weights",
        )
        proposal_indices = random.choice(
            arange(n_positions),
            n_samples,
            p=proposal_weights,
        )
        return proposal_positions[proposal_indices]

    def get_state_estimate(self):
        return self.filter_state.mean()

    def get_point_estimate(self):
        return self.get_state_estimate()

    def get_position_estimate(self):
        return self.get_state_estimate()[self.position_slice]

    def get_velocity_estimate(self):
        return self.get_state_estimate()[self.velocity_slice]

    def get_goal_estimate(self):
        return self.get_state_estimate()[self.goal_slice]

    def get_position_distribution(self) -> LinearDiracDistribution:
        return LinearDiracDistribution(
            copy.deepcopy(self.position_particles),
            copy.deepcopy(self.filter_state.w),
        )

    def get_velocity_distribution(self) -> LinearDiracDistribution:
        return LinearDiracDistribution(
            copy.deepcopy(self.velocity_particles),
            copy.deepcopy(self.filter_state.w),
        )

    def get_goal_distribution(self) -> LinearDiracDistribution:
        return LinearDiracDistribution(
            copy.deepcopy(self.goal_particles),
            copy.deepcopy(self.filter_state.w),
        )

    def position_distribution(self):
        return self.get_position_distribution()

    def velocity_distribution(self):
        return self.get_velocity_distribution()

    def goal_distribution(self):
        return self.get_goal_distribution()

    def get_goal_posterior_weights(self, candidate_goals=None):
        """Approximate posterior mass over a candidate-goal bank."""
        if candidate_goals is None:
            if self._candidate_goals is None:
                raise ValueError(
                    "candidate_goals must be provided when no goal bank is stored"
                )
            candidate_goals = self._candidate_goals

        candidate_goals = self._coerce_particle_matrix(
            candidate_goals,
            self.position_dim,
            name="candidate_goals",
        )
        diff = self.goal_particles[:, None, :] - candidate_goals[None, :, :]
        sq_dist = sum(diff * diff, axis=2)
        nearest = argmin(sq_dist, axis=1)

        posterior_weights = zeros((candidate_goals.shape[0],))
        for goal_index in range(candidate_goals.shape[0]):
            posterior_weights[goal_index] = sum(
                self.filter_state.w[nearest == goal_index]
            )

        total = sum(posterior_weights)
        if self._to_scalar(total) <= 0.0:
            raise ValueError("Posterior goal weights are degenerate")
        return posterior_weights / total

    def _sample_component_distribution(
        self,
        distribution: AbstractLinearDistribution | None,
        *,
        dim: int,
        n_samples: int,
        name: str,
        default=None,
    ):
        if distribution is None:
            if default is None:
                raise ValueError(f"{name} must be provided")
            return default
        if distribution.dim != dim:
            raise ValueError(f"{name} must have dimension {dim}")
        samples = distribution.sample(n_samples)
        return self._coerce_particle_matrix(
            samples,
            dim,
            expected_rows=n_samples,
            name=name,
        )

    @staticmethod
    def _coerce_state_vector(values, expected_dim: int, name: str):
        values = array(values)
        if ndim(values) == 0:
            values = reshape(values, (1,))
        if ndim(values) != 1:
            raise ValueError(f"{name} must be a 1-D array with shape ({expected_dim},)")
        if values.shape[0] != expected_dim:
            raise ValueError(
                f"{name} must have dimension {expected_dim}; got {values.shape[0]}"
            )
        return values

    @staticmethod
    def _coerce_particle_matrix(
        values,
        expected_dim: int,
        expected_rows: int | None = None,
        name: str = "values",
    ):
        values = array(values)
        if ndim(values) == 0:
            values = reshape(values, (1, 1))
        elif ndim(values) == 1:
            values = (
                reshape(values, (-1, 1))
                if expected_dim == 1
                else reshape(values, (1, -1))
            )

        if ndim(values) != 2:
            raise ValueError(f"{name} must be a 2-D array")
        if values.shape[1] != expected_dim:
            raise ValueError(
                f"{name} must have shape (n, {expected_dim}); got {values.shape}"
            )
        if expected_rows is not None and values.shape[0] != expected_rows:
            raise ValueError(
                f"{name} must have {expected_rows} rows; got {values.shape[0]}"
            )
        return values

    def _sample_zero_or_distribution(
        self,
        distribution: AbstractLinearDistribution | None,
        *,
        n_samples: int,
        name: str,
    ):
        if distribution is None:
            return zeros((n_samples, self.position_dim))
        return self._sample_component_distribution(
            distribution,
            dim=self.position_dim,
            n_samples=n_samples,
            name=name,
        )

    def _apply_linear_coefficient(self, coefficient, values):
        coeff = array(coefficient)
        if ndim(coeff) == 0:
            return coeff * values
        if ndim(coeff) == 1:
            if coeff.shape[0] != self.position_dim:
                raise ValueError(
                    "Vector-valued coefficients must have shape "
                    f"({self.position_dim},)"
                )
            return values * reshape(coeff, (1, -1))
        if coeff.shape != (self.position_dim, self.position_dim):
            raise ValueError(
                "Matrix-valued coefficients must have shape "
                f"({self.position_dim}, {self.position_dim})"
            )
        return values @ coeff.T

    def _apply_goal_transition(
        self,
        goals,
        positions,
        velocities,
        dt: float,
        goal_transition: Callable | None,
    ):
        if goal_transition is None:
            return goals

        try:
            updated = goal_transition(goals, positions, velocities, dt)
        except TypeError:
            try:
                updated = goal_transition(goals, positions, velocities)
            except TypeError:
                updated = goal_transition(goals)

        return self._coerce_particle_matrix(
            updated,
            self.position_dim,
            expected_rows=self.n_particles,
            name="goal_transition output",
        )

    def _evaluate_attraction_field(
        self,
        positions,
        goals,
        attraction_field: Callable,
        attraction_field_is_vectorized: bool | None,
    ):
        if attraction_field_is_vectorized is not False:
            try:
                values = attraction_field(positions, goals)
                return self._coerce_particle_matrix(
                    values,
                    self.position_dim,
                    expected_rows=self.n_particles,
                    name="attraction_field output",
                )
            except (TypeError, ValueError, AssertionError):
                if attraction_field_is_vectorized is True:
                    raise

        field = []
        for particle_index in range(self.n_particles):
            value = attraction_field(positions[particle_index], goals[particle_index])
            field.append(
                self._coerce_state_vector(
                    value,
                    self.position_dim,
                    name="attraction_field output",
                )
            )
        return array(field)

    def _draw_bernoulli_mask(self, probability: float):
        if probability <= 0.0:
            return zeros((self.n_particles,))
        if probability >= 1.0:
            return ones((self.n_particles,))
        return array(random.uniform(size=(self.n_particles,)) < probability)

    def _sample_sparse_vectors(
        self,
        distribution: AbstractLinearDistribution | None,
        jump_mask,
        *,
        name: str,
    ):
        if distribution is None:
            return zeros((self.n_particles, self.position_dim))
        samples = self._sample_component_distribution(
            distribution,
            dim=self.position_dim,
            n_samples=self.n_particles,
            name=name,
        )
        return samples * reshape(jump_mask, (-1, 1))

    def _sample_goal_reset_targets(
        self,
        goal_reset_distribution: AbstractLinearDistribution | None,
        n_samples: int,
    ):
        if goal_reset_distribution is not None:
            return self._sample_component_distribution(
                goal_reset_distribution,
                dim=self.position_dim,
                n_samples=n_samples,
                name="goal_reset_distribution",
            )
        if self._candidate_goals is not None:
            return self._sample_goals_from_candidates(
                self._candidate_goals,
                self._candidate_goal_weights,
                n_samples,
            )
        raise ValueError(
            "goal_reset_distribution must be provided when goal_reset_probability "
            "is positive and no candidate-goal bank is stored"
        )

    def predict(self, **kwargs):
        return self.predict_replay(**kwargs)

    def predict_goal_conditioned(self, **kwargs):
        return self.predict_replay(**kwargs)

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def predict_replay(
        self,
        dt: float | None = None,
        alpha=None,
        beta=None,
        attraction_field: Callable | None = None,
        goal_transition: Callable | None = None,
        process_noise: AbstractLinearDistribution | None = None,
        goal_noise: AbstractLinearDistribution | None = None,
        jump_probability: float | None = None,
        jump_distribution: AbstractLinearDistribution | None = None,
        position_jump_distribution: AbstractLinearDistribution | None = None,
        goal_reset_probability: float | None = None,
        goal_reset_distribution: AbstractLinearDistribution | None = None,
        attraction_field_is_vectorized: bool | None = None,
        gradient_is_vectorized: bool | None = None,
        function_is_vectorized: bool | None = None,
        use_semi_implicit_position_update: bool = False,
    ):
        """Predict one replay step under the goal-conditioned sparse-jump model."""
        dt = self.dt if dt is None else float(dt)
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

        if jump_probability > 0.0 and (
            jump_distribution is None and position_jump_distribution is None
        ):
            raise ValueError(
                "At least one jump distribution must be provided when "
                "jump_probability is positive"
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

        jump_mask = self._draw_bernoulli_mask(jump_probability)
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

        velocities_new = (
            self._apply_linear_coefficient(alpha, velocities)
            + self._apply_linear_coefficient(beta, control_field)
            + process_noise_samples
            + velocity_jump
        )
        velocity_for_position = (
            velocities_new if use_semi_implicit_position_update else velocities
        )
        positions_new = positions + dt * velocity_for_position + position_jump

        self._filter_state.d = concatenate(
            [positions_new, velocities_new, goals_new],
            axis=1,
        )
        self._last_transition_diagnostics = {
            "jump_mask": jump_mask,
            "goal_reset_mask": goal_reset_mask,
            "velocity_jump": velocity_jump,
            "position_jump": position_jump,
            "position_proposal_mask": zeros((self.n_particles,)),
            "position_proposal_samples": zeros((self.n_particles, self.position_dim)),
            "control_field": control_field,
            "dynamic_goals": dynamic_goals,
            "goals_for_dynamics": goals_for_dynamics,
        }
        return self

    def _ensure_likelihood_vector(self, likelihood_values):
        likelihood_values = atleast_1d(squeeze(likelihood_values))
        if likelihood_values.shape != (self.n_particles,):
            raise ValueError(
                "Likelihood must return one value per particle; "
                f"got {likelihood_values.shape}"
            )
        return likelihood_values

    def _center_measurement_distribution(
        self,
        meas_noise: AbstractLinearDistribution,
        measurement,
    ):
        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (meas_noise.dim,):
            raise ValueError(
                f"measurement must have shape ({meas_noise.dim},); "
                f"got {measurement.shape}"
            )
        if hasattr(meas_noise, "set_mode"):
            return meas_noise.set_mode(measurement)
        if hasattr(meas_noise, "set_mean"):
            return meas_noise.set_mean(measurement)
        raise AttributeError(
            "Measurement noise distribution must implement set_mode or set_mean"
        )

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

        new_state = copy.deepcopy(self.filter_state)
        new_state.w = reweighted / marginal
        new_state.d = new_state.sample(new_state.w.shape[0])
        new_state.w = ones_like(new_state.w) / new_state.w.shape[0]
        self._filter_state = new_state
        self._last_update_log_marginal = log(marginal)

        if return_log_marginal:
            return self._last_update_log_marginal
        return self

    def association_likelihood(self, likelihood, measurement=None):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            likelihood_values = likelihood(self.filter_state.d)
        else:
            likelihood_values = likelihood(measurement, self.filter_state.d)

        likelihood_values = self._ensure_likelihood_vector(likelihood_values)
        return sum(likelihood_values * self.filter_state.w)

    def update_nonlinear_using_likelihood(
        self,
        likelihood,
        measurement=None,
        return_log_marginal: bool = False,
    ):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            likelihood_values = likelihood(self.filter_state.d)
        else:
            likelihood_values = likelihood(measurement, self.filter_state.d)

        return self._apply_likelihood_values(
            likelihood_values,
            return_log_marginal=return_log_marginal,
        )

    def update_position_likelihood(
        self,
        likelihood,
        measurement=None,
        return_log_marginal: bool = False,
    ):
        if measurement is None:
            return self.update_nonlinear_using_likelihood(
                lambda particles: likelihood(particles[:, self.position_slice]),
                return_log_marginal=return_log_marginal,
            )

        return self.update_nonlinear_using_likelihood(
            lambda meas, particles: likelihood(meas, particles[:, self.position_slice]),
            measurement=measurement,
            return_log_marginal=return_log_marginal,
        )

    def apply_position_proposal(
        self,
        position_proposal,
        proposal_weights=None,
        *,
        proposal_probability: float = 1.0,
    ):
        """Rejuvenate position particles from a measurement-guided proposal.

        ``position_proposal`` may be a linear distribution, a ``(mean,
        covariance)`` tuple, or a matrix of candidate positions.  Candidate
        matrices are sampled with ``proposal_weights`` when provided.  Velocity
        and goal particles, particle weights, and IMM mode indices in subclasses
        are left unchanged.
        """

        proposal_probability = float(proposal_probability)
        self._validate_probability(proposal_probability, "proposal_probability")

        proposal_mask = self._draw_bernoulli_mask(proposal_probability)
        proposal_samples = self._sample_position_proposal(
            position_proposal,
            proposal_weights,
            n_samples=self.n_particles,
        )
        particles = copy.deepcopy(self.filter_state.d)
        particles[:, self.position_slice] = where(
            reshape(proposal_mask, (-1, 1)) > 0,
            proposal_samples,
            particles[:, self.position_slice],
        )

        new_state = type(self.filter_state)(
            particles,
            copy.deepcopy(self.filter_state.w),
        )
        self._filter_state = new_state
        diagnostics = copy.deepcopy(self._last_transition_diagnostics)
        diagnostics["position_proposal_mask"] = proposal_mask
        diagnostics["position_proposal_samples"] = proposal_samples
        self._last_transition_diagnostics = diagnostics
        return self

    def update_position_likelihood_with_proposal(
        self,
        likelihood,
        measurement=None,
        *,
        position_proposal,
        proposal_weights=None,
        proposal_probability: float = 1.0,
        return_log_marginal: bool = False,
    ):
        """Update by position likelihood, then refresh positions from a proposal.

        This is useful when the observation likelihood is available on a grid or
        candidate bank: the usual likelihood update preserves Bayesian
        reweighting/resampling of the augmented particles, while the proposal
        step moves a configurable subset of position particles onto
        measurement-supported states for subsequent replay predictions.
        """

        update_result = self.update_position_likelihood(
            likelihood,
            measurement=measurement,
            return_log_marginal=return_log_marginal,
        )
        self.apply_position_proposal(
            position_proposal,
            proposal_weights,
            proposal_probability=proposal_probability,
        )
        if return_log_marginal:
            return update_result
        return self

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

    def _component_measurement_matrix(self, component: str):
        if component == "position":
            return concatenate(
                [
                    eye(self.position_dim),
                    zeros((self.position_dim, 2 * self.position_dim)),
                ],
                axis=1,
            )
        if component == "velocity":
            return concatenate(
                [
                    zeros((self.position_dim, self.position_dim)),
                    eye(self.position_dim),
                    zeros((self.position_dim, self.position_dim)),
                ],
                axis=1,
            )
        if component == "goal":
            return concatenate(
                [
                    zeros((self.position_dim, 2 * self.position_dim)),
                    eye(self.position_dim),
                ],
                axis=1,
            )
        raise ValueError("component must be 'position', 'velocity', or 'goal'")

    def update_linear(
        self,
        measurement,
        measurement_matrix,
        meas_noise: AbstractLinearDistribution,
        return_log_marginal: bool = False,
    ):
        measurement_matrix = array(measurement_matrix)
        if ndim(measurement_matrix) != 2:
            raise ValueError("measurement_matrix must be two-dimensional")
        if measurement_matrix.shape[1] != self.state_dim:
            raise ValueError(
                "measurement_matrix must have one column per augmented state dimension"
            )

        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (measurement_matrix.shape[0],):
            raise ValueError(
                f"measurement must have shape ({measurement_matrix.shape[0]},); "
                f"got {measurement.shape}"
            )

        if meas_noise.dim != measurement_matrix.shape[0]:
            raise ValueError(
                "Measurement noise dimension must match the number of rows of the "
                "measurement matrix"
            )

        centered_meas_noise = self._center_measurement_distribution(
            meas_noise,
            measurement,
        )
        predicted_measurements = matmul(self.filter_state.d, measurement_matrix.T)
        likelihood_values = centered_meas_noise.pdf(predicted_measurements)
        return self._apply_likelihood_values(
            likelihood_values,
            return_log_marginal=return_log_marginal,
        )

    def association_likelihood_linear(
        self,
        measurement,
        measurement_matrix,
        meas_noise: AbstractLinearDistribution,
    ):
        measurement_matrix = array(measurement_matrix)
        if ndim(measurement_matrix) != 2:
            raise ValueError("measurement_matrix must be two-dimensional")
        if measurement_matrix.shape[1] != self.state_dim:
            raise ValueError(
                "measurement_matrix must have one column per augmented state dimension"
            )

        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (measurement_matrix.shape[0],):
            raise ValueError(
                f"measurement must have shape ({measurement_matrix.shape[0]},); "
                f"got {measurement.shape}"
            )

        if meas_noise.dim != measurement_matrix.shape[0]:
            raise ValueError(
                "Measurement noise dimension must match the number of rows of the "
                "measurement matrix"
            )

        centered_meas_noise = self._center_measurement_distribution(
            meas_noise,
            measurement,
        )
        predicted_measurements = matmul(self.filter_state.d, measurement_matrix.T)
        likelihood_values = centered_meas_noise.pdf(predicted_measurements)
        likelihood_values = self._ensure_likelihood_vector(likelihood_values)
        return sum(likelihood_values * self.filter_state.w)

    def update_identity(
        self,
        meas_noise,
        measurement,
        shift_instead_of_add: bool = True,
        return_log_marginal: bool = False,
    ):
        if not shift_instead_of_add:
            raise NotImplementedError()

        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (self.state_dim,):
            raise ValueError(
                f"measurement must have shape ({self.state_dim},); "
                f"got {measurement.shape}"
            )
        if meas_noise.dim != self.state_dim:
            raise ValueError("meas_noise.dim must equal state_dim")
        return self.update_linear(
            measurement,
            eye(self.state_dim),
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_position(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
        return_log_marginal: bool = False,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        if meas_noise.dim != self.position_dim:
            raise ValueError("meas_noise.dim must equal position_dim")
        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (self.position_dim,):
            raise ValueError(
                f"measurement must have shape ({self.position_dim},); "
                f"got {measurement.shape}"
            )
        return self.update_linear(
            measurement,
            self._component_measurement_matrix("position"),
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_velocity(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
        return_log_marginal: bool = False,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        if meas_noise.dim != self.position_dim:
            raise ValueError("meas_noise.dim must equal position_dim")
        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (self.position_dim,):
            raise ValueError(
                f"measurement must have shape ({self.position_dim},); "
                f"got {measurement.shape}"
            )
        return self.update_linear(
            measurement,
            self._component_measurement_matrix("velocity"),
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_goal(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
        return_log_marginal: bool = False,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        if meas_noise.dim != self.position_dim:
            raise ValueError("meas_noise.dim must equal position_dim")
        measurement = atleast_1d(squeeze(measurement))
        if measurement.shape != (self.position_dim,):
            raise ValueError(
                f"measurement must have shape ({self.position_dim},); "
                f"got {measurement.shape}"
            )
        return self.update_linear(
            measurement,
            self._component_measurement_matrix("goal"),
            meas_noise,
            return_log_marginal=return_log_marginal,
        )

    def update_position_measurement(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
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

    def association_likelihood_position(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        return self.association_likelihood_linear(
            measurement,
            self._component_measurement_matrix("position"),
            meas_noise,
        )

    def association_likelihood_velocity(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        return self.association_likelihood_linear(
            measurement,
            self._component_measurement_matrix("velocity"),
            meas_noise,
        )

    def association_likelihood_goal(
        self,
        measurement,
        meas_noise: AbstractLinearDistribution,
    ):
        measurement, meas_noise = self._resolve_measurement_and_noise_args(
            measurement,
            meas_noise,
        )
        return self.association_likelihood_linear(
            measurement,
            self._component_measurement_matrix("goal"),
            meas_noise,
        )
