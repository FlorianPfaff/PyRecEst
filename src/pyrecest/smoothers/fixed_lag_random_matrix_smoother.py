"""Fixed-lag smoother for random-matrix extended object trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pyrecest.backend import asarray, copy as backend_copy, eye, linalg
from pyrecest.filters.random_matrix_tracker import RandomMatrixTracker

from .abstract_smoother import AbstractSmoother


@dataclass
class RandomMatrixTrackerState:
    """Detached snapshot of a :class:`RandomMatrixTracker` state."""

    kinematic_state: object
    covariance: object
    extent: object
    alpha: float = 0.0
    kinematic_state_to_pos_matrix: object | None = None

    @classmethod
    def from_tracker(cls, tracker: RandomMatrixTracker) -> "RandomMatrixTrackerState":
        """Create a detached state snapshot from ``tracker``."""

        return cls(
            backend_copy(tracker.kinematic_state),
            backend_copy(tracker.covariance),
            backend_copy(tracker.extent),
            float(tracker.alpha),
            None
            if tracker.kinematic_state_to_pos_matrix is None
            else backend_copy(tracker.kinematic_state_to_pos_matrix),
        )

    def copy(self) -> "RandomMatrixTrackerState":
        """Return a detached copy of this state."""

        return RandomMatrixTrackerState(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.extent),
            float(self.alpha),
            None
            if self.kinematic_state_to_pos_matrix is None
            else backend_copy(self.kinematic_state_to_pos_matrix),
        )

    def to_tracker(self) -> RandomMatrixTracker:
        """Convert this snapshot back to a mutable tracker instance."""

        tracker = RandomMatrixTracker(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.extent),
            None
            if self.kinematic_state_to_pos_matrix is None
            else backend_copy(self.kinematic_state_to_pos_matrix),
        )
        tracker.alpha = float(self.alpha)
        return tracker


class FixedLagRandomMatrixSmoother(AbstractSmoother):
    """Fixed-lag smoother for ``RandomMatrixTracker`` posterior sequences.

    The kinematic state is smoothed with a finite-window RTS recursion. The
    random-matrix extent is smoothed by an SPD-preserving information-weighted
    average, where the alpha increase beyond the one-step prediction is used as
    a proxy for future extent information. Set ``extent_smoothing='none'`` to
    smooth only the kinematic component.
    """

    _EXTENT_SMOOTHING_MODES = ("information", "none")

    def __init__(
        self,
        lag: int = 1,
        extent_smoothing: str = "information",
        extent_smoothing_factor: float = 1.0,
        minimum_extent_weight: float = 1e-12,
    ):
        lag = int(lag)
        if lag < 0:
            raise ValueError("lag must be a non-negative integer.")
        if extent_smoothing not in self._EXTENT_SMOOTHING_MODES:
            raise ValueError(
                f"extent_smoothing must be one of {', '.join(self._EXTENT_SMOOTHING_MODES)}."
            )
        if extent_smoothing_factor < 0:
            raise ValueError("extent_smoothing_factor must be non-negative.")
        if minimum_extent_weight <= 0:
            raise ValueError("minimum_extent_weight must be positive.")

        self.lag = lag
        self.extent_smoothing = extent_smoothing
        self.extent_smoothing_factor = float(extent_smoothing_factor)
        self.minimum_extent_weight = float(minimum_extent_weight)
        self._filtered_buffer: list[RandomMatrixTrackerState] = []
        self._predicted_buffer: list[RandomMatrixTrackerState] = []
        self._system_matrix_buffer: list = []

    @staticmethod
    def _as_state(state) -> RandomMatrixTrackerState:
        if isinstance(state, RandomMatrixTrackerState):
            return state.copy()
        if isinstance(state, RandomMatrixTracker):
            return RandomMatrixTrackerState.from_tracker(state)
        if isinstance(state, tuple) and len(state) in (3, 4, 5):
            alpha = 0.0 if len(state) < 4 else float(state[3])
            pos_matrix = None if len(state) < 5 else state[4]
            return RandomMatrixTrackerState(
                asarray(state[0]).reshape(-1),
                asarray(state[1]),
                asarray(state[2]),
                alpha,
                None if pos_matrix is None else asarray(pos_matrix),
            )
        raise ValueError(
            "State must be a RandomMatrixTracker, RandomMatrixTrackerState, "
            "or a tuple (kinematic_state, covariance, extent[, alpha[, H_pos]])."
        )

    @classmethod
    def _normalize_state_sequence(cls, states: Sequence) -> list[RandomMatrixTrackerState]:
        return [cls._as_state(state) for state in states]

    def _positive_extent_weight(self, alpha) -> float:
        return max(float(alpha), self.minimum_extent_weight)

    def _smooth_extent(
        self,
        filtered_state: RandomMatrixTrackerState,
        predicted_state: RandomMatrixTrackerState,
        next_smoothed_state: RandomMatrixTrackerState,
    ) -> tuple[object, float]:
        if self.extent_smoothing == "none" or self.extent_smoothing_factor == 0.0:
            return backend_copy(filtered_state.extent), float(filtered_state.alpha)

        current_weight = self._positive_extent_weight(filtered_state.alpha)
        future_information = max(
            float(next_smoothed_state.alpha) - float(predicted_state.alpha), 0.0
        )
        future_weight = self.extent_smoothing_factor * future_information
        if future_weight <= 0.0:
            return backend_copy(filtered_state.extent), float(filtered_state.alpha)

        weight_sum = current_weight + future_weight
        smoothed_extent = (
            current_weight * filtered_state.extent
            + future_weight * next_smoothed_state.extent
        ) / weight_sum
        return self._symmetrize(smoothed_extent), weight_sum

    def _smooth_window(
        self,
        filtered_states: Sequence[RandomMatrixTrackerState],
        predicted_states: Sequence[RandomMatrixTrackerState],
        system_matrices: Sequence,
    ) -> tuple[list[RandomMatrixTrackerState], list]:
        n_states = len(filtered_states)
        if n_states == 0:
            return [], []
        if len(predicted_states) != n_states - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )
        if len(system_matrices) != n_states - 1:
            raise ValueError(
                "system_matrices must contain one entry fewer than filtered_states."
            )

        smoothed: list[RandomMatrixTrackerState | None] = [None] * n_states
        gains: list = [None] * max(n_states - 1, 0)
        smoothed[-1] = filtered_states[-1].copy()

        for time_idx in range(n_states - 2, -1, -1):
            filtered_state = filtered_states[time_idx]
            predicted_state = predicted_states[time_idx]
            system_matrix = system_matrices[time_idx]
            next_smoothed = smoothed[time_idx + 1]
            assert next_smoothed is not None

            smoother_gain = linalg.solve(
                predicted_state.covariance.T,
                (filtered_state.covariance @ system_matrix.T).T,
            ).T
            gains[time_idx] = smoother_gain

            smoothed_kinematic_state = filtered_state.kinematic_state + smoother_gain @ (
                next_smoothed.kinematic_state - predicted_state.kinematic_state
            )
            smoothed_covariance = filtered_state.covariance + smoother_gain @ (
                next_smoothed.covariance - predicted_state.covariance
            ) @ smoother_gain.T
            smoothed_extent, smoothed_alpha = self._smooth_extent(
                filtered_state,
                predicted_state,
                next_smoothed,
            )
            smoothed[time_idx] = RandomMatrixTrackerState(
                smoothed_kinematic_state,
                self._symmetrize(smoothed_covariance),
                smoothed_extent,
                smoothed_alpha,
                filtered_state.kinematic_state_to_pos_matrix,
            )

        return [state for state in smoothed if state is not None], gains

    def smooth(
        self,
        filtered_states: Sequence,
        predicted_states: Sequence | None = None,
        system_matrices=None,
        lag: int | None = None,
    ) -> tuple[list[RandomMatrixTrackerState], list[list]]:
        """Return fixed-lag smoothed random-matrix tracker states."""

        lag_value = self.lag if lag is None else int(lag)
        if lag_value < 0:
            raise ValueError("lag must be a non-negative integer.")

        filt_list = self._normalize_state_sequence(filtered_states)
        if len(filt_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if lag_value == 0 or len(filt_list) == 1:
            return [state.copy() for state in filt_list], [[] for _ in filt_list]

        if predicted_states is None:
            raise ValueError("predicted_states must be provided for non-zero lag smoothing.")
        pred_list = self._normalize_state_sequence(predicted_states)
        if len(pred_list) != len(filt_list) - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )

        state_dim = filt_list[0].kinematic_state.shape[0]
        sys_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            len(filt_list) - 1,
            "system_matrices",
            state_dim,
            default=eye(state_dim),
        )

        smoothed_states: list[RandomMatrixTrackerState] = []
        smoother_gains: list[list] = []
        for time_idx in range(len(filt_list)):
            window_end = min(time_idx + lag_value, len(filt_list) - 1)
            if window_end == time_idx:
                smoothed_states.append(filt_list[time_idx].copy())
                smoother_gains.append([])
                continue

            window_smoothed, window_gains = self._smooth_window(
                filt_list[time_idx : window_end + 1],
                pred_list[time_idx:window_end],
                sys_matrices_list[time_idx:window_end],
            )
            smoothed_states.append(window_smoothed[0])
            smoother_gains.append(window_gains)

        return smoothed_states, smoother_gains

    def append(
        self,
        filtered_state,
        predicted_state=None,
        system_matrix=None,
    ) -> RandomMatrixTrackerState | None:
        """Append a filtered state and emit the oldest fixed-lag state if ready."""

        new_filtered_state = self._as_state(filtered_state)
        if self.lag == 0:
            return new_filtered_state

        if self._filtered_buffer:
            if predicted_state is None:
                raise ValueError(
                    "predicted_state is required for the second and later filtered states."
                )
            self._predicted_buffer.append(self._as_state(predicted_state))
            state_dim = self._filtered_buffer[-1].kinematic_state.shape[0]
            self._system_matrix_buffer.append(
                eye(state_dim) if system_matrix is None else asarray(system_matrix)
            )
        elif predicted_state is not None:
            raise ValueError("predicted_state must not be provided for the first filtered state.")

        self._filtered_buffer.append(new_filtered_state)
        if len(self._filtered_buffer) <= self.lag:
            return None
        return self._emit_oldest()

    def _emit_oldest(self) -> RandomMatrixTrackerState:
        smoothed_states, _ = self._smooth_window(
            self._filtered_buffer,
            self._predicted_buffer,
            self._system_matrix_buffer,
        )
        emitted = smoothed_states[0]
        self._filtered_buffer.pop(0)
        if self._predicted_buffer:
            self._predicted_buffer.pop(0)
        if self._system_matrix_buffer:
            self._system_matrix_buffer.pop(0)
        return emitted

    def flush(self) -> list[RandomMatrixTrackerState]:
        """Return all still-buffered states with truncated look-ahead windows."""

        if self.lag == 0:
            return []

        remaining: list[RandomMatrixTrackerState] = []
        while self._filtered_buffer:
            if len(self._filtered_buffer) == 1:
                remaining.append(self._filtered_buffer.pop(0).copy())
                self._predicted_buffer.clear()
                self._system_matrix_buffer.clear()
            else:
                remaining.append(self._emit_oldest())
        return remaining


FixedLagRMTSmoother = FixedLagRandomMatrixSmoother
FLRMSmoother = FixedLagRandomMatrixSmoother
