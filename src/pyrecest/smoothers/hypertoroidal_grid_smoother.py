"""Backward-information smoothers for hypertoroidal grid distributions."""

from __future__ import annotations

import math

from pyrecest.backend import allclose, asarray, ones, pi, reshape
from pyrecest.distributions.conditional.td_cond_td_grid_distribution import (
    TdCondTdGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)

from .abstract_smoother import AbstractSmoother


class HypertoroidalGridSmoother(AbstractSmoother):
    """Fixed-interval backward-information smoother on hypertoroidal grids.

    The transition model is supplied as a conditional grid distribution with
    ``grid_values[i, j] = p(x_{t+1}=grid[i] | x_t=grid[j])``. The backward
    recursion is approximated by the quadrature rule induced by the grid.
    """

    def smooth(self, filtered_states, likelihoods, transitions):
        """Run the grid backward-information smoother.

        Parameters
        ----------
        filtered_states : sequence of HypertoroidalGridDistribution
            Filtering posterior densities ``p(x_t | z_1, ..., z_t)``.
        likelihoods : sequence of HypertoroidalGridDistribution
            Likelihood functions for the corresponding measurements. Only their
            relative scale matters.
        transitions : TdCondTdGridDistribution or sequence
            Conditional transition densities for transitions ``t -> t + 1``. A
            single transition distribution is reused for every transition.

        Returns
        -------
        smoothed_states : list of HypertoroidalGridDistribution
            Smoothed posteriors ``p(x_t | z_1, ..., z_T)``.
        backward_messages : list of HypertoroidalGridDistribution
            Backward information messages. Their scale is arbitrary.
        """

        filtered_states = self._as_grid_distribution_list(
            filtered_states, "filtered_states"
        )
        likelihoods = self._as_grid_distribution_list(likelihoods, "likelihoods")

        if len(likelihoods) != len(filtered_states):
            raise ValueError(
                "likelihoods must have the same length as filtered_states."
            )

        n_steps = len(filtered_states)
        reference = filtered_states[0]
        filtered_states = [
            self._ensure_compatible_distribution(
                dist, reference, f"filtered_states[{idx}]"
            )
            for idx, dist in enumerate(filtered_states)
        ]
        likelihoods = [
            self._ensure_compatible_distribution(dist, reference, f"likelihoods[{idx}]")
            for idx, dist in enumerate(likelihoods)
        ]
        transitions = self._normalize_transition_sequence(
            transitions, max(n_steps - 1, 0), reference
        )

        backward_messages: list[HypertoroidalGridDistribution | None] = [None] * n_steps
        smoothed_states: list[HypertoroidalGridDistribution | None] = [None] * n_steps

        backward_messages[-1] = self.constant_message_like(reference)
        smoothed_states[-1] = filtered_states[-1].multiply(backward_messages[-1])

        cell_volume = self._cell_volume(reference)
        for t in range(n_steps - 2, -1, -1):
            next_backward = backward_messages[t + 1]
            assert next_backward is not None

            future_values = self._flat_grid_values(
                likelihoods[t + 1]
            ) * self._flat_grid_values(next_backward)
            transition_values = asarray(transitions[t].grid_values)
            beta_values = transition_values.T @ (cell_volume * future_values)
            backward_messages[t] = self._make_grid_distribution_like(
                beta_values, reference
            )
            smoothed_states[t] = filtered_states[t].multiply(backward_messages[t])

        return (
            [state for state in smoothed_states if state is not None],
            [message for message in backward_messages if message is not None],
        )

    @staticmethod
    def constant_message_like(
        reference: HypertoroidalGridDistribution,
    ) -> HypertoroidalGridDistribution:
        """Return a constant backward message compatible with ``reference``."""

        if not isinstance(reference, HypertoroidalGridDistribution):
            raise TypeError("reference must be a HypertoroidalGridDistribution.")
        return HypertoroidalGridSmoother._make_grid_distribution_like(
            ones(reference.grid_values.shape), reference
        )

    @staticmethod
    def _flat_grid_values(distribution: HypertoroidalGridDistribution):
        return reshape(distribution.grid_values, (-1,))

    @staticmethod
    def _cell_volume(reference: HypertoroidalGridDistribution) -> float:
        return float(
            (2.0 * pi) ** reference.dim / math.prod(reference.grid_values.shape)
        )

    @staticmethod
    def _make_grid_distribution_like(
        values, reference: HypertoroidalGridDistribution
    ) -> HypertoroidalGridDistribution:
        return HypertoroidalGridDistribution(
            grid_values=reshape(values, reference.grid_values.shape),
            grid_type=reference.grid_type,
            grid=reference.grid,
            enforce_pdf_nonnegative=reference.enforce_pdf_nonnegative,
            dim=reference.dim,
        )

    @staticmethod
    def _as_grid_distribution_list(
        distributions, name: str
    ) -> list[HypertoroidalGridDistribution]:
        if isinstance(distributions, HypertoroidalGridDistribution):
            raise ValueError(f"{name} must be a sequence, not a single distribution.")
        try:
            distribution_list = list(distributions)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a sequence of HypertoroidalGridDistribution instances."
            ) from exc

        if len(distribution_list) == 0:
            raise ValueError(f"{name} must contain at least one distribution.")
        for idx, distribution in enumerate(distribution_list):
            if not isinstance(distribution, HypertoroidalGridDistribution):
                raise TypeError(
                    f"{name}[{idx}] must be a HypertoroidalGridDistribution."
                )
        return distribution_list

    @staticmethod
    def _ensure_compatible_distribution(
        distribution: HypertoroidalGridDistribution,
        reference: HypertoroidalGridDistribution,
        name: str,
    ) -> HypertoroidalGridDistribution:
        if distribution.dim != reference.dim:
            raise ValueError(
                f"{name} has dimension {distribution.dim}, expected {reference.dim}."
            )
        if distribution.grid_type != reference.grid_type:
            raise ValueError(
                f"{name} has grid type {distribution.grid_type!r}, expected {reference.grid_type!r}."
            )
        if distribution.grid_values.shape != reference.grid_values.shape:
            raise ValueError(
                f"{name} has grid value shape {distribution.grid_values.shape}, expected {reference.grid_values.shape}."
            )
        if distribution.enforce_pdf_nonnegative != reference.enforce_pdf_nonnegative:
            raise ValueError(
                f"{name} must agree with the reference distribution on enforce_pdf_nonnegative."
            )
        if (distribution.grid is None) != (reference.grid is None):
            raise ValueError(
                f"{name} and the reference distribution must either both store grids or both omit them."
            )
        if distribution.grid is not None and not allclose(
            distribution.grid, reference.grid
        ):
            raise ValueError(
                f"{name} has grid coordinates that differ from the reference distribution."
            )
        return distribution

    @classmethod
    def _normalize_transition_sequence(
        cls,
        transitions,
        expected_length: int,
        reference: HypertoroidalGridDistribution,
    ) -> list[TdCondTdGridDistribution]:
        if expected_length == 0:
            return []

        if isinstance(transitions, TdCondTdGridDistribution):
            transition_list = [transitions] * expected_length
        else:
            try:
                transition_list = list(transitions)
            except TypeError as exc:
                raise TypeError(
                    "transitions must be a TdCondTdGridDistribution or a sequence of them."
                ) from exc

        if len(transition_list) != expected_length:
            raise ValueError(
                "transitions must be a single distribution or contain one distribution per transition."
            )

        for idx, transition in enumerate(transition_list):
            cls._ensure_compatible_transition(
                transition, reference, f"transitions[{idx}]"
            )
        return transition_list

    @staticmethod
    def _ensure_compatible_transition(
        transition: TdCondTdGridDistribution,
        reference: HypertoroidalGridDistribution,
        name: str,
    ) -> None:
        if not isinstance(transition, TdCondTdGridDistribution):
            raise TypeError(f"{name} must be a TdCondTdGridDistribution.")

        n_points = math.prod(reference.grid_values.shape)
        if transition.grid_values.shape != (n_points, n_points):
            raise ValueError(
                f"{name}.grid_values must have shape {(n_points, n_points)}, got {transition.grid_values.shape}."
            )
        if transition.grid.shape != (n_points, reference.dim):
            raise ValueError(
                f"{name}.grid must have shape {(n_points, reference.dim)}, got {transition.grid.shape}."
            )
        reference_grid = reference.get_grid()
        if not allclose(transition.grid, reference_grid):
            raise ValueError(
                f"{name}.grid must match the grid of the state distributions."
            )


HypertoroidalGridBackwardInformationSmoother = HypertoroidalGridSmoother
HGSmoother = HypertoroidalGridSmoother
