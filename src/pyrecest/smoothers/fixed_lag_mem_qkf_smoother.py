"""Fixed-lag and fixed-interval smoothers for MEM-QKF trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from pyrecest.backend import abs as backend_abs
from pyrecest.backend import (
    array,
    asarray,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    diag,
    eye,
    linalg,
    maximum,
    pi,
    where,
)
from pyrecest.filters.mem_qkf_tracker import MEMQKFTracker

from .abstract_smoother import AbstractSmoother


@dataclass
class MEMQKFTrackerState:
    """Detached snapshot of a :class:`MEMQKFTracker` state."""

    kinematic_state: Any
    covariance: Any
    shape_state: Any
    shape_covariance: Any
    measurement_matrix: Any | None = None
    multiplicative_noise_cov: Any | None = None
    covariance_regularization: float = 0.0
    default_meas_noise_cov: Any | None = None
    update_mode: str = "sequential"
    minimum_axis_length: float = 1e-9
    minimum_covariance_eigenvalue: float = 0.0

    @classmethod
    def from_tracker(cls, tracker: MEMQKFTracker) -> "MEMQKFTrackerState":
        """Create a detached snapshot from ``tracker``."""
        return cls(
            backend_copy(tracker.kinematic_state),
            backend_copy(tracker.covariance),
            backend_copy(tracker.shape_state),
            backend_copy(tracker.shape_covariance),
            (
                None
                if tracker.measurement_matrix is None
                else backend_copy(tracker.measurement_matrix)
            ),
            (
                None
                if tracker.multiplicative_noise_cov is None
                else backend_copy(tracker.multiplicative_noise_cov)
            ),
            float(tracker.covariance_regularization),
            (
                None
                if tracker.default_meas_noise_cov is None
                else backend_copy(tracker.default_meas_noise_cov)
            ),
            str(tracker.update_mode),
            float(tracker.minimum_axis_length),
            float(tracker.minimum_covariance_eigenvalue),
        )

    def copy(self) -> "MEMQKFTrackerState":
        """Return a detached copy of this state."""
        return MEMQKFTrackerState(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.shape_state),
            backend_copy(self.shape_covariance),
            (
                None
                if self.measurement_matrix is None
                else backend_copy(self.measurement_matrix)
            ),
            (
                None
                if self.multiplicative_noise_cov is None
                else backend_copy(self.multiplicative_noise_cov)
            ),
            float(self.covariance_regularization),
            (
                None
                if self.default_meas_noise_cov is None
                else backend_copy(self.default_meas_noise_cov)
            ),
            str(self.update_mode),
            float(self.minimum_axis_length),
            float(self.minimum_covariance_eigenvalue),
        )

    def to_tracker(self) -> MEMQKFTracker:
        """Convert this snapshot back to a mutable tracker instance."""
        return MEMQKFTracker(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.shape_state),
            backend_copy(self.shape_covariance),
            measurement_matrix=(
                None
                if self.measurement_matrix is None
                else backend_copy(self.measurement_matrix)
            ),
            multiplicative_noise_cov=(
                None
                if self.multiplicative_noise_cov is None
                else backend_copy(self.multiplicative_noise_cov)
            ),
            covariance_regularization=float(self.covariance_regularization),
            default_meas_noise_cov=(
                None
                if self.default_meas_noise_cov is None
                else backend_copy(self.default_meas_noise_cov)
            ),
            update_mode=str(self.update_mode),
            minimum_axis_length=float(self.minimum_axis_length),
            minimum_covariance_eigenvalue=float(self.minimum_covariance_eigenvalue),
        )


@dataclass
class MEMQKFSmootherGain:
    """Smoother gains for one MEM-QKF backward recursion step."""

    kinematic: Any
    shape: Any | None = None


class FixedLagMEMQKFSmoother(AbstractSmoother):
    """Fixed-lag RTS smoother for ``MEMQKFTracker`` posterior sequences.

    The kinematic state is smoothed with the standard finite-window RTS
    recursion. The MEM-QKF shape state ``[orientation, semi_axis_1,
    semi_axis_2]`` can either be smoothed with a separate RTS recursion or be
    passed through unchanged. Orientation residuals are treated as axial,
    pi-periodic ellipse-orientation differences.
    """

    _SHAPE_SMOOTHING_MODES = ("rts", "none")

    def __init__(self, lag: int = 1, shape_smoothing: str = "rts"):
        lag = int(lag)
        if lag < 0:
            raise ValueError("lag must be a non-negative integer.")
        if shape_smoothing not in self._SHAPE_SMOOTHING_MODES:
            raise ValueError("shape_smoothing must be 'rts' or 'none'.")
        self.lag = lag
        self.shape_smoothing = shape_smoothing
        self._filtered_buffer: list[MEMQKFTrackerState] = []
        self._predicted_buffer: list[MEMQKFTrackerState] = []
        self._system_matrix_buffer: list[Any] = []
        self._shape_system_matrix_buffer: list[Any] = []

    @classmethod
    def _as_state(cls, state) -> MEMQKFTrackerState:
        if isinstance(state, MEMQKFTrackerState):
            return state.copy()
        if isinstance(state, MEMQKFTracker):
            return MEMQKFTrackerState.from_tracker(state)
        if isinstance(state, tuple) and len(state) in (4, 5):
            kwargs = {} if len(state) == 4 else dict(state[4])
            return MEMQKFTrackerState(
                asarray(state[0]).reshape(-1),
                asarray(state[1]),
                asarray(state[2]).reshape(3),
                asarray(state[3]),
                **kwargs,
            )
        raise ValueError(
            "State must be a MEMQKFTracker, MEMQKFTrackerState, or a tuple "
            "(kinematic_state, covariance, shape_state, shape_covariance[, kwargs])."
        )

    @classmethod
    def _normalize_state_sequence(cls, states: Sequence) -> list[MEMQKFTrackerState]:
        return [cls._as_state(state) for state in states]

    @classmethod
    def _project_symmetric_covariance(cls, covariance, minimum_eigenvalue=0.0):
        covariance = cls._symmetrize(asarray(covariance))
        eigenvalues, eigenvectors = linalg.eigh(covariance)
        if float(eigenvalues[0]) >= minimum_eigenvalue:
            return covariance
        eigenvalues = maximum(eigenvalues, minimum_eigenvalue)
        return cls._symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)

    @classmethod
    def _canonicalize_shape(
        cls, state: MEMQKFTrackerState, shape_state, shape_covariance
    ):
        shape_state = asarray(shape_state).reshape(3)
        shape_covariance = cls._project_symmetric_covariance(
            shape_covariance, state.minimum_covariance_eigenvalue
        )
        axes = shape_state[1:]
        signs = where(axes < 0.0, -1.0, 1.0)
        sign_matrix = diag(signs)
        axis_covariance = sign_matrix @ shape_covariance[1:, 1:] @ sign_matrix.T
        axes = maximum(backend_abs(axes), state.minimum_axis_length)
        axis_covariance = cls._project_symmetric_covariance(
            axis_covariance, state.minimum_covariance_eigenvalue
        )
        orientation_variance = maximum(
            shape_covariance[0, 0], state.minimum_covariance_eigenvalue
        )
        return array([shape_state[0], axes[0], axes[1]]), cls._symmetrize(
            linalg.block_diag(array([[orientation_variance]]), axis_covariance)
        )

    @staticmethod
    def _axial_angle_delta(reference, theta):
        return ((theta - reference + pi / 2.0) % pi) - pi / 2.0

    @classmethod
    def _wrap_axial_to_reference(cls, reference, theta):
        return reference + cls._axial_angle_delta(reference, theta)

    @classmethod
    def _shape_residual(cls, reference_shape_state, shape_state):
        reference_shape_state = asarray(reference_shape_state).reshape(3)
        shape_state = asarray(shape_state).reshape(3)
        return array(
            [
                cls._axial_angle_delta(reference_shape_state[0], shape_state[0]),
                shape_state[1] - reference_shape_state[1],
                shape_state[2] - reference_shape_state[2],
            ]
        )

    def _postprocess_state(
        self,
        reference_state,
        kinematic_state,
        covariance,
        shape_state,
        shape_covariance,
    ):
        covariance = self._project_symmetric_covariance(
            covariance, reference_state.minimum_covariance_eigenvalue
        )
        shape_state = asarray(shape_state).reshape(3)
        shape_state = array(
            [
                self._wrap_axial_to_reference(
                    reference_state.shape_state[0], shape_state[0]
                ),
                shape_state[1],
                shape_state[2],
            ]
        )
        shape_state, shape_covariance = self._canonicalize_shape(
            reference_state, shape_state, shape_covariance
        )
        return MEMQKFTrackerState(
            backend_copy(kinematic_state),
            covariance,
            shape_state,
            self._project_symmetric_covariance(
                shape_covariance, reference_state.minimum_covariance_eigenvalue
            ),
            (
                None
                if reference_state.measurement_matrix is None
                else backend_copy(reference_state.measurement_matrix)
            ),
            (
                None
                if reference_state.multiplicative_noise_cov is None
                else backend_copy(reference_state.multiplicative_noise_cov)
            ),
            float(reference_state.covariance_regularization),
            (
                None
                if reference_state.default_meas_noise_cov is None
                else backend_copy(reference_state.default_meas_noise_cov)
            ),
            str(reference_state.update_mode),
            float(reference_state.minimum_axis_length),
            float(reference_state.minimum_covariance_eigenvalue),
        )

    @staticmethod
    def _shape_system_matrices(shape_system_matrices, length: int) -> list[Any]:
        return AbstractSmoother._normalize_matrix_sequence(
            shape_system_matrices, length, "shape_system_matrices", 3, default=eye(3)
        )

    def _smooth_shape(
        self, filtered_state, predicted_state, next_smoothed_state, shape_system_matrix
    ):
        if self.shape_smoothing == "none":
            return (
                backend_copy(filtered_state.shape_state),
                backend_copy(filtered_state.shape_covariance),
                None,
            )
        shape_gain = linalg.solve(
            predicted_state.shape_covariance.T,
            (filtered_state.shape_covariance @ shape_system_matrix.T).T,
        ).T
        shape_state = filtered_state.shape_state + shape_gain @ self._shape_residual(
            predicted_state.shape_state, next_smoothed_state.shape_state
        )
        shape_state = array(
            [
                self._wrap_axial_to_reference(
                    filtered_state.shape_state[0], shape_state[0]
                ),
                shape_state[1],
                shape_state[2],
            ]
        )
        shape_covariance = (
            filtered_state.shape_covariance
            + shape_gain
            @ (next_smoothed_state.shape_covariance - predicted_state.shape_covariance)
            @ shape_gain.T
        )
        return shape_state, self._symmetrize(shape_covariance), shape_gain

    def _smooth_window(
        self, filtered_states, predicted_states, system_matrices, shape_system_matrices
    ):
        n_states = len(filtered_states)
        if n_states == 0:
            return [], []
        if len(predicted_states) != n_states - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )
        smoothed: list[MEMQKFTrackerState | None] = [None] * n_states
        gains: list[MEMQKFSmootherGain | None] = [None] * max(n_states - 1, 0)
        smoothed[-1] = self._postprocess_state(
            filtered_states[-1],
            filtered_states[-1].kinematic_state,
            filtered_states[-1].covariance,
            filtered_states[-1].shape_state,
            filtered_states[-1].shape_covariance,
        )
        for time_idx in range(n_states - 2, -1, -1):
            filtered_state = filtered_states[time_idx]
            predicted_state = predicted_states[time_idx]
            next_smoothed = smoothed[time_idx + 1]
            assert next_smoothed is not None
            system_matrix = system_matrices[time_idx]
            kinematic_gain = linalg.solve(
                predicted_state.covariance.T,
                (filtered_state.covariance @ system_matrix.T).T,
            ).T
            kinematic_state = filtered_state.kinematic_state + kinematic_gain @ (
                next_smoothed.kinematic_state - predicted_state.kinematic_state
            )
            covariance = (
                filtered_state.covariance
                + kinematic_gain
                @ (next_smoothed.covariance - predicted_state.covariance)
                @ kinematic_gain.T
            )
            shape_state, shape_covariance, shape_gain = self._smooth_shape(
                filtered_state,
                predicted_state,
                next_smoothed,
                shape_system_matrices[time_idx],
            )
            smoothed[time_idx] = self._postprocess_state(
                filtered_state,
                kinematic_state,
                self._symmetrize(covariance),
                shape_state,
                shape_covariance,
            )
            gains[time_idx] = MEMQKFSmootherGain(kinematic_gain, shape_gain)
        return [state for state in smoothed if state is not None], gains

    def smooth(
        self,
        filtered_states: Sequence,
        predicted_states: Sequence | None = None,
        system_matrices=None,
        shape_system_matrices=None,
        lag: int | None = None,
    ) -> tuple[list[MEMQKFTrackerState], list[list[Any]]]:
        """Return fixed-lag smoothed MEM-QKF tracker states."""
        lag_value = self.lag if lag is None else int(lag)
        if lag_value < 0:
            raise ValueError("lag must be a non-negative integer.")
        filt_list = self._normalize_state_sequence(filtered_states)
        if len(filt_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if lag_value == 0 or len(filt_list) == 1:
            return [
                self._postprocess_state(
                    s,
                    s.kinematic_state,
                    s.covariance,
                    s.shape_state,
                    s.shape_covariance,
                )
                for s in filt_list
            ], [[] for _ in filt_list]
        if predicted_states is None:
            raise ValueError(
                "predicted_states must be provided for non-zero lag smoothing."
            )
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
        shape_matrices_list = self._shape_system_matrices(
            shape_system_matrices, len(filt_list) - 1
        )
        smoothed_states: list[MEMQKFTrackerState] = []
        smoother_gains: list[list[Any]] = []
        for time_idx in range(len(filt_list)):
            window_end = min(time_idx + lag_value, len(filt_list) - 1)
            if window_end == time_idx:
                smoothed_states.append(
                    self._postprocess_state(
                        filt_list[time_idx],
                        filt_list[time_idx].kinematic_state,
                        filt_list[time_idx].covariance,
                        filt_list[time_idx].shape_state,
                        filt_list[time_idx].shape_covariance,
                    )
                )
                smoother_gains.append([])
                continue
            window_smoothed, window_gains = self._smooth_window(
                filt_list[time_idx : window_end + 1],
                pred_list[time_idx:window_end],
                sys_matrices_list[time_idx:window_end],
                shape_matrices_list[time_idx:window_end],
            )
            smoothed_states.append(window_smoothed[0])
            smoother_gains.append(window_gains)
        return smoothed_states, smoother_gains

    def append(
        self,
        filtered_state,
        predicted_state=None,
        system_matrix=None,
        shape_system_matrix=None,
    ):
        """Append a filtered state and emit the oldest fixed-lag state if ready."""
        new_filtered_state = self._as_state(filtered_state)
        if self.lag == 0:
            return self._postprocess_state(
                new_filtered_state,
                new_filtered_state.kinematic_state,
                new_filtered_state.covariance,
                new_filtered_state.shape_state,
                new_filtered_state.shape_covariance,
            )
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
            self._shape_system_matrix_buffer.append(
                eye(3) if shape_system_matrix is None else asarray(shape_system_matrix)
            )
        elif predicted_state is not None:
            raise ValueError(
                "predicted_state must not be provided for the first filtered state."
            )
        self._filtered_buffer.append(new_filtered_state)
        if len(self._filtered_buffer) <= self.lag:
            return None
        return self._emit_oldest()

    def _emit_oldest(self):
        smoothed_states, _ = self._smooth_window(
            self._filtered_buffer,
            self._predicted_buffer,
            self._system_matrix_buffer,
            self._shape_system_matrix_buffer,
        )
        emitted = smoothed_states[0]
        self._filtered_buffer.pop(0)
        if self._predicted_buffer:
            self._predicted_buffer.pop(0)
        if self._system_matrix_buffer:
            self._system_matrix_buffer.pop(0)
        if self._shape_system_matrix_buffer:
            self._shape_system_matrix_buffer.pop(0)
        return emitted

    def flush(self) -> list[MEMQKFTrackerState]:
        """Return all still-buffered states with truncated look-ahead windows."""
        if self.lag == 0:
            return []
        remaining: list[MEMQKFTrackerState] = []
        while self._filtered_buffer:
            if len(self._filtered_buffer) == 1:
                state = self._filtered_buffer.pop(0)
                remaining.append(
                    self._postprocess_state(
                        state,
                        state.kinematic_state,
                        state.covariance,
                        state.shape_state,
                        state.shape_covariance,
                    )
                )
                self._predicted_buffer.clear()
                self._system_matrix_buffer.clear()
                self._shape_system_matrix_buffer.clear()
            else:
                remaining.append(self._emit_oldest())
        return remaining


class FixedIntervalMEMQKFSmoother(FixedLagMEMQKFSmoother):
    """Full fixed-interval RTS smoother for ``MEMQKFTracker`` posterior sequences."""

    def __init__(self, shape_smoothing: str = "rts"):
        super().__init__(lag=0, shape_smoothing=shape_smoothing)

    def smooth(
        self,
        filtered_states: Sequence,
        predicted_states: Sequence | None = None,
        system_matrices=None,
        shape_system_matrices=None,
        lag: int | None = None,
    ) -> tuple[list[MEMQKFTrackerState], list[list[Any]]]:
        """Return full-interval smoothed MEM-QKF tracker states."""
        _ = lag
        lag_value = max(len(filtered_states) - 1, 0)
        return super().smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            system_matrices=system_matrices,
            shape_system_matrices=shape_system_matrices,
            lag=lag_value,
        )


FixedLagMemQkfSmoother = FixedLagMEMQKFSmoother
FixedLagFreeMEMQKFSmoother = FixedLagMEMQKFSmoother
FLMEMQKFSmoother = FixedLagMEMQKFSmoother
FixedIntervalMemQkfSmoother = FixedIntervalMEMQKFSmoother
FullIntervalMEMQKFSmoother = FixedIntervalMEMQKFSmoother
