"""Unscented Rauch--Tung--Striebel smoother for nonlinear Gaussian models."""

# pylint: disable=duplicate-code

from __future__ import annotations

import inspect
from copy import copy
from typing import Callable, Sequence

# pylint: disable=no-member
import pyrecest.backend
from pyrecest.filters._ukf import MerweScaledSigmaPoints
from pyrecest.backend import asarray, linalg, ndim, outer, stack, zeros
from pyrecest.distributions import GaussianDistribution

from .abstract_smoother import AbstractSmoother


class UnscentedRauchTungStriebelSmoother(AbstractSmoother):
    """Unscented fixed-interval smoother for nonlinear Gaussian state-space models.

    This implements the unscented Rauch--Tung--Striebel (URTS) smoother for
    Euclidean state spaces. It mirrors the current :class:`UnscentedKalmanFilter`
    scope in PyRecEst and therefore only supports the NumPy backend.

    The smoother provides a complete forward pass (`filter`) and the backward pass
    (`smooth`). For the nonlinear case the smoother gain depends on the predicted
    cross-covariance between ``x_t`` and ``x_{t+1}``, so the forward pass stores
    those cross-covariances explicitly.

    Parameters
    ----------
    points
        Optional sigma-point object compatible with ``MerweScaledSigmaPoints``.
        If omitted, standard Merwe scaled sigma points are used.
    alpha, beta, kappa
        Default sigma-point parameters used when ``points`` is omitted.
    """

    def __init__(
        self, points=None, alpha: float = 0.001, beta: float = 2.0, kappa: float = 0.0
    ):
        self.points = points
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    @staticmethod
    def _assert_supported_backend():
        backend_name: str = getattr(
            pyrecest.backend, "__backend_name__", ""
        )  # pylint: disable=no-member
        assert backend_name not in (
            "pytorch",
            "jax",
        ), "Not supported on this backend"

    @staticmethod
    def _normalize_callable_sequence(
        functions, length: int, name: str, default=None
    ) -> list:
        if length == 0:
            return []

        if functions is None:
            if default is None:
                raise ValueError(f"{name} must be provided.")
            return [default] * length

        if callable(functions):
            return [functions] * length

        if isinstance(functions, (list, tuple)) and len(functions) == length:
            if not all(callable(function) for function in functions):
                raise ValueError(f"All entries in {name} must be callable.")
            return list(functions)

        raise ValueError(
            f"{name} must be a callable or a sequence of callables with length {length}."
        )

    @staticmethod
    def _normalize_scalar_sequence(
        values, length: int, name: str, default=None
    ) -> list:
        if length == 0:
            return []

        if values is None:
            return [default] * length

        values_arr = asarray(values)
        if ndim(values_arr) == 0:
            return [
                values_arr.item() if hasattr(values_arr, "item") else values_arr
            ] * length
        if ndim(values_arr) == 1 and values_arr.shape[0] == length:
            return [
                (
                    values_arr[idx].item()
                    if hasattr(values_arr[idx], "item")
                    else values_arr[idx]
                )
                for idx in range(length)
            ]

        if isinstance(values, (list, tuple)) and len(values) == length:
            return list(values)

        raise ValueError(f"{name} must be a scalar or a sequence with length {length}.")

    @staticmethod
    def _weighted_sum(values, weights):
        result = zeros(values[0].shape)
        for idx, weight in enumerate(weights):
            result = result + weight * values[idx]
        return result

    @staticmethod
    def _call_transition(function: Callable, sigma_point, time_step):
        if time_step is None:
            return asarray(function(sigma_point)).reshape(-1)

        try:
            signature = inspect.signature(function)
            positional_parameters = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_varargs = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if has_varargs or len(positional_parameters) >= 2:
                return asarray(function(sigma_point, time_step)).reshape(-1)
        except (TypeError, ValueError):
            pass

        return asarray(function(sigma_point)).reshape(-1)

    def _get_sigma_points(self, state_dim: int):
        if self.points is None:
            return MerweScaledSigmaPoints(
                state_dim,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa,
            )

        points_dim = getattr(self.points, "n", None)
        if points_dim is not None and points_dim != state_dim:
            raise ValueError(
                f"Sigma-point object has dimension {points_dim}, expected {state_dim}."
            )
        return self.points

    def _predict_state(
        self,
        filtered_state: "GaussianDistribution | tuple",
        transition_function: Callable,
        sys_noise_covariance,
        time_step,
    ) -> tuple[GaussianDistribution, object]:
        filtered_state = self._as_gaussian(filtered_state)
        sigma_points = self._get_sigma_points(filtered_state.dim)
        sigma_point_array = asarray(
            sigma_points.sigma_points(filtered_state.mu, filtered_state.C)
        )
        propagated_sigma_points = stack(
            [
                self._call_transition(transition_function, sigma_point, time_step)
                for sigma_point in sigma_point_array
            ],
            axis=0,
        )

        predicted_mean = self._weighted_sum(
            propagated_sigma_points,
            sigma_points.Wm,
        )
        predicted_covariance = copy(asarray(sys_noise_covariance))
        cross_covariance = zeros((filtered_state.dim, filtered_state.dim))

        for idx in range(propagated_sigma_points.shape[0]):
            sigma_point_deviation = sigma_point_array[idx] - filtered_state.mu
            propagated_deviation = propagated_sigma_points[idx] - predicted_mean
            predicted_covariance = predicted_covariance + sigma_points.Wc[idx] * outer(
                propagated_deviation,
                propagated_deviation,
            )
            cross_covariance = cross_covariance + sigma_points.Wc[idx] * outer(
                sigma_point_deviation,
                propagated_deviation,
            )

        predicted_covariance = self._symmetrize(predicted_covariance)

        return (
            GaussianDistribution(
                predicted_mean,
                predicted_covariance,
                check_validity=False,
            ),
            cross_covariance,
        )

    def _update_state(  # pylint: disable=too-many-locals
        self,
        predicted_state: "GaussianDistribution | tuple",
        measurement,
        measurement_function: Callable,
        meas_noise_covariance,
    ) -> GaussianDistribution:
        predicted_state = self._as_gaussian(predicted_state)
        measurement = asarray(measurement).reshape(-1)

        sigma_points = self._get_sigma_points(predicted_state.dim)
        sigma_point_array = asarray(
            sigma_points.sigma_points(predicted_state.mu, predicted_state.C)
        )
        measurement_sigma_points = stack(
            [
                asarray(measurement_function(sigma_point)).reshape(-1)
                for sigma_point in sigma_point_array
            ],
            axis=0,
        )

        predicted_measurement = self._weighted_sum(
            measurement_sigma_points,
            sigma_points.Wm,
        )
        innovation_covariance = copy(asarray(meas_noise_covariance))
        state_measurement_cross_covariance = zeros(
            (predicted_state.dim, predicted_measurement.shape[0])
        )

        for idx in range(measurement_sigma_points.shape[0]):
            sigma_point_deviation = sigma_point_array[idx] - predicted_state.mu
            measurement_deviation = (
                measurement_sigma_points[idx] - predicted_measurement
            )
            innovation_covariance = innovation_covariance + sigma_points.Wc[
                idx
            ] * outer(
                measurement_deviation,
                measurement_deviation,
            )
            state_measurement_cross_covariance = (
                state_measurement_cross_covariance
                + sigma_points.Wc[idx]
                * outer(sigma_point_deviation, measurement_deviation)
            )

        kalman_gain = linalg.solve(
            innovation_covariance.T,
            state_measurement_cross_covariance.T,
        ).T
        filtered_mean = predicted_state.mu + kalman_gain @ (
            measurement - predicted_measurement
        )
        filtered_covariance = (
            predicted_state.C - kalman_gain @ innovation_covariance @ kalman_gain.T
        )
        filtered_covariance = self._symmetrize(filtered_covariance)

        return GaussianDistribution(
            filtered_mean,
            filtered_covariance,
            check_validity=False,
        )

    def filter(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        initial_state: "GaussianDistribution | tuple",
        measurements,
        measurement_functions=None,
        meas_noise_covariances=None,
        transition_functions=None,
        sys_noise_covariances=None,
        time_steps=None,
    ) -> tuple[list[GaussianDistribution], list[GaussianDistribution], list]:
        """Run the forward unscented filtering pass for a full sequence.

        Returns
        -------
        filtered_states
            Posterior states ``x_{t|t}``.
        predicted_states
            One-step predictions ``x_{t+1|t}`` with length ``T - 1``.
        predicted_cross_covariances
            Cross-covariances ``Cov[x_t, x_{t+1}]`` needed by the URTS backward pass.
        """
        self._assert_supported_backend()
        initial_state = self._as_gaussian(initial_state)
        measurement_list = self._normalize_measurements(measurements)

        if len(measurement_list) == 0:
            raise ValueError("At least one measurement is required.")

        measurement_dim = measurement_list[0].shape[0]
        state_dim = initial_state.dim

        measurement_functions_list = self._normalize_callable_sequence(
            measurement_functions,
            len(measurement_list),
            "measurement_functions",
            default=lambda x: x,
        )
        meas_noise_covariances_list = self._normalize_matrix_sequence(
            meas_noise_covariances,
            len(measurement_list),
            "meas_noise_covariances",
            measurement_dim,
        )
        transition_functions_list = self._normalize_callable_sequence(
            transition_functions,
            max(len(measurement_list) - 1, 0),
            "transition_functions",
            default=lambda x, _dt=1.0: x,
        )
        sys_noise_covariances_list = self._normalize_matrix_sequence(
            sys_noise_covariances,
            max(len(measurement_list) - 1, 0),
            "sys_noise_covariances",
            state_dim,
            default=zeros((state_dim, state_dim)),
        )
        time_steps_list = self._normalize_scalar_sequence(
            time_steps,
            max(len(measurement_list) - 1, 0),
            "time_steps",
            default=1.0,
        )

        filtered_states: list[GaussianDistribution] = []
        predicted_states: list[GaussianDistribution] = []
        predicted_cross_covariances: list = []

        filtered_state = self._update_state(
            initial_state,
            measurement_list[0],
            measurement_functions_list[0],
            meas_noise_covariances_list[0],
        )
        filtered_states.append(filtered_state)

        for time_idx in range(len(measurement_list) - 1):
            predicted_state, predicted_cross_covariance = self._predict_state(
                filtered_states[-1],
                transition_functions_list[time_idx],
                sys_noise_covariances_list[time_idx],
                time_steps_list[time_idx],
            )
            predicted_states.append(predicted_state)
            predicted_cross_covariances.append(predicted_cross_covariance)

            filtered_state = self._update_state(
                predicted_state,
                measurement_list[time_idx + 1],
                measurement_functions_list[time_idx + 1],
                meas_noise_covariances_list[time_idx + 1],
            )
            filtered_states.append(filtered_state)

        return filtered_states, predicted_states, predicted_cross_covariances

    def smooth(  # pylint: disable=too-many-locals
        self,
        filtered_states: Sequence["GaussianDistribution | tuple"],
        predicted_states: Sequence["GaussianDistribution | tuple"],
        predicted_cross_covariances: Sequence,
    ) -> tuple[list[GaussianDistribution], list]:
        """Run the unscented RTS backward pass."""
        self._assert_supported_backend()

        filt_list = [self._as_gaussian(state) for state in filtered_states]
        pred_list = [self._as_gaussian(state) for state in predicted_states]

        if len(filt_list) == 0:
            return [], []
        if len(pred_list) != len(filt_list) - 1:
            raise ValueError(
                "predicted_states must have length len(filtered_states) - 1."
            )
        if len(predicted_cross_covariances) != len(pred_list):
            raise ValueError(
                "predicted_cross_covariances must have the same length as predicted_states."
            )

        smoothed: list[GaussianDistribution | None] = [None] * len(filt_list)
        gains: list = [None] * len(pred_list)
        smoothed[-1] = filt_list[-1]

        for time_idx in range(len(filt_list) - 2, -1, -1):
            cross_covariance = asarray(predicted_cross_covariances[time_idx])
            smoother_gain = linalg.solve(pred_list[time_idx].C.T, cross_covariance.T).T
            gains[time_idx] = smoother_gain

            next_smoothed = smoothed[time_idx + 1]
            assert next_smoothed is not None

            mean_update = next_smoothed.mu - pred_list[time_idx].mu
            smoothed_mean = filt_list[time_idx].mu + smoother_gain @ mean_update
            covariance_update = next_smoothed.C - pred_list[time_idx].C
            smoothed_covariance = (
                filt_list[time_idx].C
                + smoother_gain @ covariance_update @ smoother_gain.T
            )
            smoothed_covariance = self._symmetrize(smoothed_covariance)

            smoothed[time_idx] = GaussianDistribution(
                smoothed_mean,
                smoothed_covariance,
                check_validity=False,
            )

        return [s for s in smoothed if s is not None], gains

    def smooth_from_filtered(
        self,
        filtered_states: Sequence["GaussianDistribution | tuple"],
        transition_functions=None,
        sys_noise_covariances=None,
        time_steps=None,
    ) -> tuple[list[GaussianDistribution], list]:
        """Smooth from filtered states by recomputing nonlinear predictions."""
        self._assert_supported_backend()

        filt_list = [self._as_gaussian(state) for state in filtered_states]
        if len(filt_list) == 0:
            return [], []

        state_dim = filt_list[0].dim
        transition_functions_list = self._normalize_callable_sequence(
            transition_functions,
            max(len(filt_list) - 1, 0),
            "transition_functions",
            default=lambda x, _dt=1.0: x,
        )
        sys_noise_covariances_list = self._normalize_matrix_sequence(
            sys_noise_covariances,
            max(len(filt_list) - 1, 0),
            "sys_noise_covariances",
            state_dim,
            default=zeros((state_dim, state_dim)),
        )
        time_steps_list = self._normalize_scalar_sequence(
            time_steps,
            max(len(filt_list) - 1, 0),
            "time_steps",
            default=1.0,
        )

        predicted_states: list[GaussianDistribution] = []
        predicted_cross_covariances: list = []
        for time_idx in range(len(filt_list) - 1):
            predicted_state, predicted_cross_covariance = self._predict_state(
                filt_list[time_idx],
                transition_functions_list[time_idx],
                sys_noise_covariances_list[time_idx],
                time_steps_list[time_idx],
            )
            predicted_states.append(predicted_state)
            predicted_cross_covariances.append(predicted_cross_covariance)

        return self.smooth(
            filtered_states=filt_list,
            predicted_states=predicted_states,
            predicted_cross_covariances=predicted_cross_covariances,
        )

    def filter_and_smooth(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        initial_state: "GaussianDistribution | tuple",
        measurements,
        measurement_functions=None,
        meas_noise_covariances=None,
        transition_functions=None,
        sys_noise_covariances=None,
        time_steps=None,
    ) -> tuple[
        list[GaussianDistribution],
        list[GaussianDistribution],
        list[GaussianDistribution],
        list,
    ]:
        """Convenience wrapper that runs both the forward and backward passes."""
        filtered_states, predicted_states, predicted_cross_covariances = self.filter(
            initial_state=initial_state,
            measurements=measurements,
            measurement_functions=measurement_functions,
            meas_noise_covariances=meas_noise_covariances,
            transition_functions=transition_functions,
            sys_noise_covariances=sys_noise_covariances,
            time_steps=time_steps,
        )
        smoothed_states, smoother_gains = self.smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            predicted_cross_covariances=predicted_cross_covariances,
        )
        return filtered_states, predicted_states, smoothed_states, smoother_gains


URTSSmoother = UnscentedRauchTungStriebelSmoother
