"""Rauch--Tung--Striebel smoother for linear Gaussian state-space models."""

# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Sequence

from pyrecest.backend import eye, linalg, zeros
from pyrecest.distributions import GaussianDistribution

from .abstract_smoother import AbstractSmoother


class RauchTungStriebelSmoother(AbstractSmoother):
    """Rauch--Tung--Striebel smoother for linear Gaussian models.

    This class intentionally does not depend on ``pyrecest.filters.KalmanFilter``.
    The current filter classes do not retain the full forward-pass history that an
    RTS smoother requires, so this implementation provides its own forward pass for
    sequences of linear Gaussian models.

    The accepted model inputs can each either be a single matrix/vector reused for
    the whole sequence or a sequence with one entry per time step.

    Parameters
    ----------
    initial_state
        Initial prior state as ``GaussianDistribution`` or ``(mean, covariance)``.
    measurements
        Sequence of measurements. A one-dimensional array is interpreted as a
        sequence of scalar measurements. For vector measurements, pass a list/tuple
        of one-dimensional arrays or a two-dimensional array of shape ``(T, dim_z)``.
    measurement_matrices
        Measurement matrix ``H`` or sequence ``H_t``. Defaults to identity.
    meas_noise_covariances
        Measurement noise covariance ``R`` or sequence ``R_t``.
    system_matrices
        Transition matrix ``F`` or sequence ``F_t`` used between consecutive time
        steps. Defaults to identity.
    sys_noise_covariances
        Process noise covariance ``Q`` or sequence ``Q_t``. Defaults to zero.
    sys_inputs
        Optional control/input vector ``u`` or sequence ``u_t`` added in the
        prediction step. Defaults to zero.
    """

    def _backward_pass(
        self,
        filt_list: list[GaussianDistribution],
        pred_list: list[GaussianDistribution],
        sys_matrices_list: list,
    ) -> tuple[list[GaussianDistribution], list]:
        """RTS backward recursion given pre-computed filtered/predicted states."""
        n_states = len(filt_list)
        smoothed: list[GaussianDistribution | None] = [None] * n_states
        gains: list = [None] * len(pred_list)

        last = filt_list[-1]
        smoothed[-1] = GaussianDistribution(
            last.mu, self._symmetrize(last.C), check_validity=False
        )

        for t in range(n_states - 2, -1, -1):
            system_matrix = sys_matrices_list[t]
            smoother_gain = linalg.solve(
                pred_list[t].C.T,
                (filt_list[t].C @ system_matrix.T).T,
            ).T
            gains[t] = smoother_gain

            next_smoothed = smoothed[t + 1]
            assert next_smoothed is not None

            smoothed_mean = filt_list[t].mu + smoother_gain @ (
                next_smoothed.mu - pred_list[t].mu
            )
            smoothed_cov = (
                filt_list[t].C
                + smoother_gain @ (next_smoothed.C - pred_list[t].C) @ smoother_gain.T
            )
            smoothed[t] = GaussianDistribution(
                smoothed_mean, self._symmetrize(smoothed_cov), check_validity=False
            )

        return [s for s in smoothed if s is not None], gains

    def filter(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        initial_state: "GaussianDistribution | tuple",
        measurements,
        measurement_matrices=None,
        meas_noise_covariances=None,
        system_matrices=None,
        sys_noise_covariances=None,
        sys_inputs=None,
    ) -> tuple[list[GaussianDistribution], list[GaussianDistribution]]:
        """Run the forward Kalman filtering pass for a sequence.

        Returns
        -------
        filtered_states
            List of posterior states ``x_{t|t}``.
        predicted_states
            List of one-step predictions ``x_{t+1|t}``. Its length is one less than
            the number of measurements.
        """

        initial_state = self._as_gaussian(initial_state)
        measurement_list = self._normalize_measurements(measurements)

        if len(measurement_list) == 0:
            raise ValueError("At least one measurement is required.")

        state_dim = initial_state.dim
        identity_matrix = eye(state_dim)

        measurement_matrices_list = self._normalize_matrix_sequence(
            measurement_matrices,
            len(measurement_list),
            "measurement_matrices",
            state_dim,
            default=identity_matrix,
        )
        meas_noise_covariances_list = self._normalize_matrix_sequence(
            meas_noise_covariances,
            len(measurement_list),
            "meas_noise_covariances",
            state_dim,
        )
        system_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            max(len(measurement_list) - 1, 0),
            "system_matrices",
            state_dim,
            default=identity_matrix,
        )
        sys_noise_covariances_list = self._normalize_matrix_sequence(
            sys_noise_covariances,
            max(len(measurement_list) - 1, 0),
            "sys_noise_covariances",
            state_dim,
            default=zeros((state_dim, state_dim)),
        )
        sys_inputs_list = self._normalize_vector_sequence(
            sys_inputs,
            max(len(measurement_list) - 1, 0),
            "sys_inputs",
            state_dim,
        )

        filtered_states: list[GaussianDistribution] = []
        predicted_states: list[GaussianDistribution] = []

        predicted_mean = initial_state.mu
        predicted_covariance = initial_state.C

        for time_idx, measurement in enumerate(measurement_list):
            measurement_matrix = measurement_matrices_list[time_idx]
            meas_noise_covariance = meas_noise_covariances_list[time_idx]

            innovation = measurement - measurement_matrix @ predicted_mean
            innovation_covariance = (
                measurement_matrix @ predicted_covariance @ measurement_matrix.T
                + meas_noise_covariance
            )
            kalman_gain = linalg.solve(
                innovation_covariance.T,
                (predicted_covariance @ measurement_matrix.T).T,
            ).T

            filtered_mean = predicted_mean + kalman_gain @ innovation
            measurement_update = identity_matrix - kalman_gain @ measurement_matrix
            filtered_covariance = (
                measurement_update @ predicted_covariance @ measurement_update.T
                + kalman_gain @ meas_noise_covariance @ kalman_gain.T
            )
            filtered_covariance = self._symmetrize(filtered_covariance)

            filtered_states.append(
                GaussianDistribution(
                    filtered_mean,
                    filtered_covariance,
                    check_validity=False,
                )
            )

            if time_idx == len(measurement_list) - 1:
                continue

            system_matrix = system_matrices_list[time_idx]
            sys_noise_covariance = sys_noise_covariances_list[time_idx]
            sys_input = (
                zeros(state_dim)
                if sys_inputs_list[time_idx] is None
                else sys_inputs_list[time_idx]
            )

            predicted_mean = system_matrix @ filtered_mean + sys_input
            predicted_covariance = (
                system_matrix @ filtered_covariance @ system_matrix.T
                + sys_noise_covariance
            )
            predicted_covariance = self._symmetrize(predicted_covariance)

            predicted_states.append(
                GaussianDistribution(
                    predicted_mean,
                    predicted_covariance,
                    check_validity=False,
                )
            )

        return filtered_states, predicted_states

    def smooth(
        self,
        filtered_states: Sequence["GaussianDistribution | tuple"],
        predicted_states: Sequence["GaussianDistribution | tuple"],
        system_matrices=None,
    ) -> tuple[list[GaussianDistribution], list]:
        """Run the RTS backward pass.

        Parameters
        ----------
        filtered_states
            Sequence of posterior states ``x_{t|t}``.
        predicted_states
            Sequence of one-step predictions ``x_{t+1|t}``.
        system_matrices
            Transition matrix ``F`` or sequence ``F_t`` used in the forward pass.

        Returns
        -------
        smoothed_states
            List of smoothed posterior states ``x_{t|T}``.
        smoother_gains
            List of RTS smoother gains, one per backward recursion step.
        """

        filt_list = [self._as_gaussian(state) for state in filtered_states]
        pred_list = [self._as_gaussian(state) for state in predicted_states]

        if len(filt_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if len(pred_list) != max(len(filt_list) - 1, 0):
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )

        state_dim = filt_list[0].dim
        sys_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            max(len(filt_list) - 1, 0),
            "system_matrices",
            state_dim,
            default=eye(state_dim),
        )

        return self._backward_pass(filt_list, pred_list, sys_matrices_list)

    def filter_and_smooth(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        initial_state: "GaussianDistribution | tuple",
        measurements,
        measurement_matrices=None,
        meas_noise_covariances=None,
        system_matrices=None,
        sys_noise_covariances=None,
        sys_inputs=None,
    ) -> tuple[
        list[GaussianDistribution],
        list[GaussianDistribution],
        list[GaussianDistribution],
        list,
    ]:
        """Run a full forward-backward pass for a linear Gaussian sequence."""

        filtered_states, predicted_states = self.filter(
            initial_state=initial_state,
            measurements=measurements,
            measurement_matrices=measurement_matrices,
            meas_noise_covariances=meas_noise_covariances,
            system_matrices=system_matrices,
            sys_noise_covariances=sys_noise_covariances,
            sys_inputs=sys_inputs,
        )
        smoothed_states, smoother_gains = self.smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            system_matrices=system_matrices,
        )
        return filtered_states, predicted_states, smoothed_states, smoother_gains


RTSSmoother = RauchTungStriebelSmoother
