"""Fixed-lag smoother for velocity-locked MEM-QKF extended object trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pyrecest.backend import arctan2, array, asarray, copy as backend_copy, diag, eye, linalg, maximum, where
from pyrecest.backend import abs as backend_abs
from pyrecest.filters.velocity_locked_mem_qkf_tracker import VelocityLockedMEMQKFTracker

from .abstract_smoother import AbstractSmoother


@dataclass
class VelocityLockedMEMQKFTrackerState:
    """Detached snapshot of a :class:`VelocityLockedMEMQKFTracker` state."""

    kinematic_state: object
    covariance: object
    shape_state: object
    shape_covariance: object
    measurement_matrix: object | None = None
    multiplicative_noise_cov: object | None = None
    covariance_regularization: float = 0.0
    default_meas_noise_cov: object | None = None
    update_mode: str = "sequential"
    minimum_axis_length: float = 1e-9
    minimum_covariance_eigenvalue: float = 0.0
    velocity_indices: tuple[int, int] = (2, 3)
    speed_threshold: float = 1e-9
    orientation_offset: float = 0.0
    sideslip_variance: float = 0.0
    minimum_orientation_variance: float = 1e-12

    @classmethod
    def from_tracker(cls, tracker: VelocityLockedMEMQKFTracker) -> "VelocityLockedMEMQKFTrackerState":
        """Create a detached snapshot from ``tracker``."""
        return cls(
            backend_copy(tracker.kinematic_state),
            backend_copy(tracker.covariance),
            backend_copy(tracker.shape_state),
            backend_copy(tracker.shape_covariance),
            None if tracker.measurement_matrix is None else backend_copy(tracker.measurement_matrix),
            backend_copy(tracker.multiplicative_noise_cov),
            float(tracker.covariance_regularization),
            None if tracker.default_meas_noise_cov is None else backend_copy(tracker.default_meas_noise_cov),
            str(tracker.update_mode),
            float(tracker.minimum_axis_length),
            float(tracker.minimum_covariance_eigenvalue),
            tuple(tracker.velocity_indices),
            float(tracker.speed_threshold),
            float(tracker.orientation_offset),
            float(tracker.sideslip_variance),
            float(tracker.minimum_orientation_variance),
        )

    def copy(self) -> "VelocityLockedMEMQKFTrackerState":
        """Return a detached copy of this state."""
        return VelocityLockedMEMQKFTrackerState(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.shape_state),
            backend_copy(self.shape_covariance),
            None if self.measurement_matrix is None else backend_copy(self.measurement_matrix),
            None if self.multiplicative_noise_cov is None else backend_copy(self.multiplicative_noise_cov),
            float(self.covariance_regularization),
            None if self.default_meas_noise_cov is None else backend_copy(self.default_meas_noise_cov),
            str(self.update_mode),
            float(self.minimum_axis_length),
            float(self.minimum_covariance_eigenvalue),
            tuple(self.velocity_indices),
            float(self.speed_threshold),
            float(self.orientation_offset),
            float(self.sideslip_variance),
            float(self.minimum_orientation_variance),
        )

    def to_tracker(self) -> VelocityLockedMEMQKFTracker:
        """Convert this snapshot back to a mutable tracker instance."""
        return VelocityLockedMEMQKFTracker(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.shape_state),
            backend_copy(self.shape_covariance),
            measurement_matrix=None if self.measurement_matrix is None else backend_copy(self.measurement_matrix),
            multiplicative_noise_cov=None if self.multiplicative_noise_cov is None else backend_copy(self.multiplicative_noise_cov),
            covariance_regularization=float(self.covariance_regularization),
            default_meas_noise_cov=None if self.default_meas_noise_cov is None else backend_copy(self.default_meas_noise_cov),
            update_mode=str(self.update_mode),
            minimum_axis_length=float(self.minimum_axis_length),
            minimum_covariance_eigenvalue=float(self.minimum_covariance_eigenvalue),
            velocity_indices=tuple(self.velocity_indices),
            speed_threshold=float(self.speed_threshold),
            orientation_offset=float(self.orientation_offset),
            sideslip_variance=float(self.sideslip_variance),
            minimum_orientation_variance=float(self.minimum_orientation_variance),
        )


@dataclass
class VelocityLockedMEMQKFSmootherGain:
    """Smoother gains for one VL-MEM-QKF backward recursion step."""

    kinematic: object
    shape: object | None = None


class FixedLagVelocityLockedMEMQKFSmoother(AbstractSmoother):
    """Fixed-lag smoother for ``VelocityLockedMEMQKFTracker`` posterior sequences.

    Kinematics are smoothed by a finite-window RTS recursion. The MEM-QKF shape
    state can either be passed through unchanged or smoothed by a separate RTS
    recursion in ``[orientation, semi_axis_1, semi_axis_2]`` space. Every output
    state is then projected back to the decoupled MEM-QKF covariance convention,
    and moving states are relocked to the smoothed kinematic velocity.
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
        self._filtered_buffer: list[VelocityLockedMEMQKFTrackerState] = []
        self._predicted_buffer: list[VelocityLockedMEMQKFTrackerState] = []
        self._system_matrix_buffer: list = []
        self._shape_system_matrix_buffer: list = []

    @staticmethod
    def _normalize_velocity_indices(velocity_indices, state_dim: int) -> tuple[int, int]:
        if len(velocity_indices) != 2:
            raise ValueError("velocity_indices must contain exactly two entries.")
        normalized = []
        for index in velocity_indices:
            index = int(index)
            if index < 0:
                index += state_dim
            if index < 0 or index >= state_dim:
                raise ValueError("velocity index out of bounds for kinematic state dimension.")
            normalized.append(index)
        if normalized[0] == normalized[1]:
            raise ValueError("velocity_indices must refer to two distinct states.")
        return tuple(normalized)

    @classmethod
    def _as_state(cls, state) -> VelocityLockedMEMQKFTrackerState:
        if isinstance(state, VelocityLockedMEMQKFTrackerState):
            return state.copy()
        if isinstance(state, VelocityLockedMEMQKFTracker):
            return VelocityLockedMEMQKFTrackerState.from_tracker(state)
        if isinstance(state, tuple) and len(state) in (4, 5):
            kwargs = {} if len(state) == 4 else dict(state[4])
            state_dim = asarray(state[0]).reshape(-1).shape[0]
            if "velocity_indices" in kwargs:
                kwargs["velocity_indices"] = cls._normalize_velocity_indices(kwargs["velocity_indices"], state_dim)
            return VelocityLockedMEMQKFTrackerState(
                asarray(state[0]).reshape(-1),
                asarray(state[1]),
                asarray(state[2]).reshape(3),
                asarray(state[3]),
                **kwargs,
            )
        raise ValueError(
            "State must be a VelocityLockedMEMQKFTracker, VelocityLockedMEMQKFTrackerState, "
            "or a tuple (kinematic_state, covariance, shape_state, shape_covariance[, kwargs])."
        )

    @classmethod
    def _normalize_state_sequence(cls, states: Sequence) -> list[VelocityLockedMEMQKFTrackerState]:
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
    def _canonicalize_shape(cls, state: VelocityLockedMEMQKFTrackerState, shape_state, shape_covariance):
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

    @classmethod
    def _heading_moments(cls, state: VelocityLockedMEMQKFTrackerState, kinematic_state, covariance):
        vx_index, vy_index = cls._normalize_velocity_indices(
            state.velocity_indices, asarray(kinematic_state).reshape(-1).shape[0]
        )
        velocity_x = kinematic_state[vx_index]
        velocity_y = kinematic_state[vy_index]
        speed_squared = velocity_x**2 + velocity_y**2
        if float(speed_squared) <= state.speed_threshold**2:
            return None
        orientation = arctan2(velocity_y, velocity_x) + state.orientation_offset
        jacobian_entries = [0.0] * asarray(kinematic_state).reshape(-1).shape[0]
        jacobian_entries[vx_index] = -velocity_y / speed_squared
        jacobian_entries[vy_index] = velocity_x / speed_squared
        heading_jacobian = array(jacobian_entries)
        orientation_variance = heading_jacobian @ covariance @ heading_jacobian.T + state.sideslip_variance
        orientation_variance = maximum(orientation_variance, state.minimum_orientation_variance)
        return orientation, orientation_variance

    def _postprocess_state(self, reference_state, kinematic_state, covariance, shape_state, shape_covariance):
        covariance = self._project_symmetric_covariance(covariance, reference_state.minimum_covariance_eigenvalue)
        shape_state, shape_covariance = self._canonicalize_shape(reference_state, shape_state, shape_covariance)
        heading_moments = self._heading_moments(reference_state, kinematic_state, covariance)
        if heading_moments is not None:
            orientation, orientation_variance = heading_moments
            shape_state = array([orientation, shape_state[1], shape_state[2]])
            shape_covariance = self._symmetrize(
                linalg.block_diag(array([[orientation_variance]]), shape_covariance[1:, 1:])
            )
        return VelocityLockedMEMQKFTrackerState(
            backend_copy(kinematic_state),
            covariance,
            shape_state,
            self._project_symmetric_covariance(shape_covariance, reference_state.minimum_covariance_eigenvalue),
            None if reference_state.measurement_matrix is None else backend_copy(reference_state.measurement_matrix),
            None if reference_state.multiplicative_noise_cov is None else backend_copy(reference_state.multiplicative_noise_cov),
            float(reference_state.covariance_regularization),
            None if reference_state.default_meas_noise_cov is None else backend_copy(reference_state.default_meas_noise_cov),
            str(reference_state.update_mode),
            float(reference_state.minimum_axis_length),
            float(reference_state.minimum_covariance_eigenvalue),
            tuple(reference_state.velocity_indices),
            float(reference_state.speed_threshold),
            float(reference_state.orientation_offset),
            float(reference_state.sideslip_variance),
            float(reference_state.minimum_orientation_variance),
        )

    @staticmethod
    def _shape_system_matrices(shape_system_matrices, length: int) -> list:
        return AbstractSmoother._normalize_matrix_sequence(
            shape_system_matrices, length, "shape_system_matrices", 3, default=eye(3)
        )

    def _smooth_shape(self, filtered_state, predicted_state, next_smoothed_state, shape_system_matrix):
        if self.shape_smoothing == "none":
            return backend_copy(filtered_state.shape_state), backend_copy(filtered_state.shape_covariance), None
        shape_gain = linalg.solve(
            predicted_state.shape_covariance.T,
            (filtered_state.shape_covariance @ shape_system_matrix.T).T,
        ).T
        shape_state = filtered_state.shape_state + shape_gain @ (
            next_smoothed_state.shape_state - predicted_state.shape_state
        )
        shape_covariance = filtered_state.shape_covariance + shape_gain @ (
            next_smoothed_state.shape_covariance - predicted_state.shape_covariance
        ) @ shape_gain.T
        return shape_state, self._symmetrize(shape_covariance), shape_gain

    def _smooth_window(self, filtered_states, predicted_states, system_matrices, shape_system_matrices):
        n_states = len(filtered_states)
        if n_states == 0:
            return [], []
        if len(predicted_states) != n_states - 1:
            raise ValueError("predicted_states must contain one entry fewer than filtered_states.")
        smoothed: list[VelocityLockedMEMQKFTrackerState | None] = [None] * n_states
        gains: list[VelocityLockedMEMQKFSmootherGain | None] = [None] * max(n_states - 1, 0)
        smoothed[-1] = self._postprocess_state(
            filtered_states[-1], filtered_states[-1].kinematic_state, filtered_states[-1].covariance,
            filtered_states[-1].shape_state, filtered_states[-1].shape_covariance
        )
        for time_idx in range(n_states - 2, -1, -1):
            filtered_state = filtered_states[time_idx]
            predicted_state = predicted_states[time_idx]
            next_smoothed = smoothed[time_idx + 1]
            system_matrix = system_matrices[time_idx]
            kinematic_gain = linalg.solve(
                predicted_state.covariance.T,
                (filtered_state.covariance @ system_matrix.T).T,
            ).T
            kinematic_state = filtered_state.kinematic_state + kinematic_gain @ (
                next_smoothed.kinematic_state - predicted_state.kinematic_state
            )
            covariance = filtered_state.covariance + kinematic_gain @ (
                next_smoothed.covariance - predicted_state.covariance
            ) @ kinematic_gain.T
            shape_state, shape_covariance, shape_gain = self._smooth_shape(
                filtered_state, predicted_state, next_smoothed, shape_system_matrices[time_idx]
            )
            smoothed[time_idx] = self._postprocess_state(
                filtered_state, kinematic_state, self._symmetrize(covariance), shape_state, shape_covariance
            )
            gains[time_idx] = VelocityLockedMEMQKFSmootherGain(kinematic_gain, shape_gain)
        return [state for state in smoothed if state is not None], gains

    def smooth(
        self,
        filtered_states: Sequence,
        predicted_states: Sequence | None = None,
        system_matrices=None,
        shape_system_matrices=None,
        lag: int | None = None,
    ) -> tuple[list[VelocityLockedMEMQKFTrackerState], list[list]]:
        """Return fixed-lag smoothed VL-MEM-QKF tracker states."""
        lag_value = self.lag if lag is None else int(lag)
        if lag_value < 0:
            raise ValueError("lag must be a non-negative integer.")
        filt_list = self._normalize_state_sequence(filtered_states)
        if len(filt_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if lag_value == 0 or len(filt_list) == 1:
            return [
                self._postprocess_state(s, s.kinematic_state, s.covariance, s.shape_state, s.shape_covariance)
                for s in filt_list
            ], [[] for _ in filt_list]
        if predicted_states is None:
            raise ValueError("predicted_states must be provided for non-zero lag smoothing.")
        pred_list = self._normalize_state_sequence(predicted_states)
        if len(pred_list) != len(filt_list) - 1:
            raise ValueError("predicted_states must contain one entry fewer than filtered_states.")
        state_dim = filt_list[0].kinematic_state.shape[0]
        sys_matrices_list = self._normalize_matrix_sequence(
            system_matrices, len(filt_list) - 1, "system_matrices", state_dim, default=eye(state_dim)
        )
        shape_matrices_list = self._shape_system_matrices(shape_system_matrices, len(filt_list) - 1)
        smoothed_states = []
        smoother_gains = []
        for time_idx in range(len(filt_list)):
            window_end = min(time_idx + lag_value, len(filt_list) - 1)
            if window_end == time_idx:
                smoothed_states.append(
                    self._postprocess_state(
                        filt_list[time_idx], filt_list[time_idx].kinematic_state, filt_list[time_idx].covariance,
                        filt_list[time_idx].shape_state, filt_list[time_idx].shape_covariance
                    )
                )
                smoother_gains.append([])
                continue
            window_smoothed, window_gains = self._smooth_window(
                filt_list[time_idx:window_end + 1], pred_list[time_idx:window_end],
                sys_matrices_list[time_idx:window_end], shape_matrices_list[time_idx:window_end]
            )
            smoothed_states.append(window_smoothed[0])
            smoother_gains.append(window_gains)
        return smoothed_states, smoother_gains

    def append(self, filtered_state, predicted_state=None, system_matrix=None, shape_system_matrix=None):
        """Append a filtered state and emit the oldest fixed-lag state if ready."""
        new_filtered_state = self._as_state(filtered_state)
        if self.lag == 0:
            return self._postprocess_state(
                new_filtered_state, new_filtered_state.kinematic_state, new_filtered_state.covariance,
                new_filtered_state.shape_state, new_filtered_state.shape_covariance
            )
        if self._filtered_buffer:
            if predicted_state is None:
                raise ValueError("predicted_state is required for the second and later filtered states.")
            self._predicted_buffer.append(self._as_state(predicted_state))
            state_dim = self._filtered_buffer[-1].kinematic_state.shape[0]
            self._system_matrix_buffer.append(eye(state_dim) if system_matrix is None else asarray(system_matrix))
            self._shape_system_matrix_buffer.append(eye(3) if shape_system_matrix is None else asarray(shape_system_matrix))
        elif predicted_state is not None:
            raise ValueError("predicted_state must not be provided for the first filtered state.")
        self._filtered_buffer.append(new_filtered_state)
        if len(self._filtered_buffer) <= self.lag:
            return None
        return self._emit_oldest()

    def _emit_oldest(self):
        smoothed_states, _ = self._smooth_window(
            self._filtered_buffer, self._predicted_buffer,
            self._system_matrix_buffer, self._shape_system_matrix_buffer
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

    def flush(self) -> list[VelocityLockedMEMQKFTrackerState]:
        """Return all still-buffered states with truncated look-ahead windows."""
        if self.lag == 0:
            return []
        remaining = []
        while self._filtered_buffer:
            if len(self._filtered_buffer) == 1:
                state = self._filtered_buffer.pop(0)
                remaining.append(
                    self._postprocess_state(state, state.kinematic_state, state.covariance, state.shape_state, state.shape_covariance)
                )
                self._predicted_buffer.clear()
                self._system_matrix_buffer.clear()
                self._shape_system_matrix_buffer.clear()
            else:
                remaining.append(self._emit_oldest())
        return remaining


FixedLagVLMEMQKFSmoother = FixedLagVelocityLockedMEMQKFSmoother
FLVLMEMQKFSmoother = FixedLagVelocityLockedMEMQKFSmoother
