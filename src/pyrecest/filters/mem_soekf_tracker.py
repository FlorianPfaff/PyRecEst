from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,duplicate-code,too-many-locals
from pyrecest.backend import (
    array,
    concatenate,
    cos,
    diag,
    eye,
    linalg,
    sin,
    trace,
    zeros,
)

from .mem_ekf_tracker import MEMEKFTracker


class MEMSOEKFTracker(MEMEKFTracker):
    """Second-order MEM-EKF tracker for one 2-D elliptical extended object.

    The tracker uses the same multiplicative-error-model state convention as
    :class:`MEMEKFTracker`, but performs the measurement update with the SOEKF
    pseudo-measurement ``[dx, dy, dx^2, dx * dy, dy^2]``. The state is locally
    shifted to the predicted measurement before each single-measurement update,
    matching the robust centering used by the reference MEM-SOEKF equations.
    """

    def __init__(self, *args, finite_difference_step=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.finite_difference_step = float(finite_difference_step)
        if self.finite_difference_step <= 0.0:
            raise ValueError("finite_difference_step must be positive")

    @staticmethod
    def _extent_transform_from_shape(shape_state):
        orientation, semi_axis_1, semi_axis_2 = shape_state
        rotation_matrix = array(
            [
                [cos(orientation), -sin(orientation)],
                [sin(orientation), cos(orientation)],
            ]
        )
        return rotation_matrix @ diag(array([semi_axis_1, semi_axis_2]))

    @staticmethod
    def _extent_row_jacobians_from_shape(shape_state):
        orientation, semi_axis_1, semi_axis_2 = shape_state
        first_row_jacobian = array(
            [
                [-semi_axis_1 * sin(orientation), cos(orientation), 0.0],
                [-semi_axis_2 * cos(orientation), 0.0, -sin(orientation)],
            ]
        )
        second_row_jacobian = array(
            [
                [semi_axis_1 * cos(orientation), sin(orientation), 0.0],
                [-semi_axis_2 * sin(orientation), 0.0, cos(orientation)],
            ]
        )
        return first_row_jacobian, second_row_jacobian

    def _shape_pseudo_jacobian_soekf(self, multiplicative_noise_cov):
        first_row_jacobian, second_row_jacobian = self._extent_row_jacobians_from_shape(
            self.shape_state
        )
        extent_transform = self._extent_transform()
        first_extent_row = extent_transform[0, :]
        second_extent_row = extent_transform[1, :]

        return array(
            [
                2.0 * first_extent_row @ multiplicative_noise_cov @ first_row_jacobian,
                first_extent_row @ multiplicative_noise_cov @ second_row_jacobian
                + second_extent_row @ multiplicative_noise_cov @ first_row_jacobian,
                2.0 * second_extent_row @ multiplicative_noise_cov @ second_row_jacobian,
            ]
        )

    def _shape_pseudo_hessians_soekf(self, multiplicative_noise_cov):
        def shape_pseudo_mean(shape_state):
            extent_transform = self._extent_transform_from_shape(shape_state)
            first_extent_row = extent_transform[0, :]
            second_extent_row = extent_transform[1, :]
            return array(
                [
                    first_extent_row @ multiplicative_noise_cov @ first_extent_row,
                    first_extent_row @ multiplicative_noise_cov @ second_extent_row,
                    second_extent_row @ multiplicative_noise_cov @ second_extent_row,
                ]
            )

        return self._finite_difference_hessians(shape_pseudo_mean, self.shape_state)

    def _evaluate_augmented_measurement(self, augmented_state, measurement_matrix):
        kinematic_dim = self.kinematic_state.shape[0]
        shape_start = kinematic_dim
        multiplicative_noise_start = shape_start + 3
        additive_noise_start = multiplicative_noise_start + self.measurement_dim

        kinematic_delta = augmented_state[:kinematic_dim]
        shape_state = augmented_state[shape_start:multiplicative_noise_start]
        multiplicative_noise = augmented_state[
            multiplicative_noise_start:additive_noise_start
        ]
        additive_noise = augmented_state[
            additive_noise_start : additive_noise_start + self.measurement_dim
        ]

        local_measurement = (
            measurement_matrix @ kinematic_delta
            + self._extent_transform_from_shape(shape_state) @ multiplicative_noise
            + additive_noise
        )
        return array(
            [
                local_measurement[0],
                local_measurement[1],
                local_measurement[0] ** 2,
                local_measurement[0] * local_measurement[1],
                local_measurement[1] ** 2,
            ]
        )

    def _finite_difference_jacobian(self, function, point):
        point = array(point)
        step = self.finite_difference_step
        perturbations = step * eye(point.shape[0])
        columns = []
        for dim_index in range(point.shape[0]):
            perturbation = perturbations[dim_index, :]
            columns.append(
                (function(point + perturbation) - function(point - perturbation))
                / (2.0 * step)
            )
        return array(columns).T

    def _finite_difference_hessians(self, function, point):
        point = array(point)
        step = self.finite_difference_step
        perturbations = step * eye(point.shape[0])
        output_dim = function(point).shape[0]
        hessians = []
        for output_index in range(output_dim):
            rows = []
            for row_index in range(point.shape[0]):
                row = []
                row_perturbation = perturbations[row_index, :]
                for col_index in range(point.shape[0]):
                    col_perturbation = perturbations[col_index, :]
                    second_difference = (
                        function(point + row_perturbation + col_perturbation)[
                            output_index
                        ]
                        - function(point + row_perturbation - col_perturbation)[
                            output_index
                        ]
                        - function(point - row_perturbation + col_perturbation)[
                            output_index
                        ]
                        + function(point - row_perturbation - col_perturbation)[
                            output_index
                        ]
                    ) / (4.0 * step**2)
                    row.append(second_difference)
                rows.append(row)
            hessians.append(rows)
        return array(hessians)

    @staticmethod
    def _replace_shape_pseudo_jacobian(jacobian, shape_jacobian, shape_start):
        rows = []
        shape_stop = shape_start + 3
        for row_index in range(jacobian.shape[0]):
            row = []
            for col_index in range(jacobian.shape[1]):
                if row_index >= 2 and shape_start <= col_index < shape_stop:
                    value = shape_jacobian[row_index - 2, col_index - shape_start]
                else:
                    value = jacobian[row_index, col_index]
                row.append(value)
            rows.append(row)
        return array(rows)

    @staticmethod
    def _add_shape_pseudo_hessians(hessians, shape_hessians, shape_start):
        corrected_hessians = []
        shape_stop = shape_start + 3
        for output_index in range(hessians.shape[0]):
            rows = []
            for row_index in range(hessians.shape[1]):
                row = []
                for col_index in range(hessians.shape[2]):
                    value = hessians[output_index, row_index, col_index]
                    if (
                        output_index >= 2
                        and shape_start <= row_index < shape_stop
                        and shape_start <= col_index < shape_stop
                    ):
                        value = value + shape_hessians[
                            output_index - 2,
                            row_index - shape_start,
                            col_index - shape_start,
                        ]
                    row.append(value)
                rows.append(row)
            corrected_hessians.append(rows)
        return array(corrected_hessians)

    def _second_order_moments(
        self,
        augmented_mean,
        augmented_covariance,
        measurement_matrix,
        multiplicative_noise_cov,
    ):
        def measurement_function(augmented_state):
            return self._evaluate_augmented_measurement(
                augmented_state,
                measurement_matrix,
            )

        shape_start = self.kinematic_state.shape[0]
        jacobian = self._finite_difference_jacobian(
            measurement_function,
            augmented_mean,
        )
        jacobian = self._replace_shape_pseudo_jacobian(
            jacobian,
            self._shape_pseudo_jacobian_soekf(multiplicative_noise_cov),
            shape_start,
        )

        hessians = self._finite_difference_hessians(
            measurement_function,
            augmented_mean,
        )
        hessians = self._add_shape_pseudo_hessians(
            hessians,
            self._shape_pseudo_hessians_soekf(multiplicative_noise_cov),
            shape_start,
        )

        nominal_measurement = measurement_function(augmented_mean)
        expected_measurement = []
        covariance_rows = []
        for row_index in range(nominal_measurement.shape[0]):
            expected_measurement.append(
                nominal_measurement[row_index]
                + 0.5 * trace(hessians[row_index] @ augmented_covariance)
            )
            covariance_row = []
            for col_index in range(nominal_measurement.shape[0]):
                covariance_row.append(
                    jacobian[row_index, :]
                    @ augmented_covariance
                    @ jacobian[col_index, :]
                    + 0.5
                    * trace(
                        hessians[row_index]
                        @ augmented_covariance
                        @ hessians[col_index]
                        @ augmented_covariance
                    )
                )
            covariance_rows.append(covariance_row)

        return (
            array(expected_measurement),
            self._symmetrize(array(covariance_rows)),
            jacobian,
        )

    def _update_single_measurement(
        self,
        measurement,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
    ):
        kinematic_dim = self.kinematic_state.shape[0]
        augmented_mean = concatenate(
            [
                zeros(kinematic_dim),
                self.shape_state,
                zeros(self.measurement_dim),
                zeros(self.measurement_dim),
            ]
        )
        augmented_covariance = linalg.block_diag(
            self.covariance,
            self.shape_covariance,
            multiplicative_noise_cov,
            meas_noise_cov,
        )
        (
            expected_measurement,
            measurement_covariance,
            jacobian,
        ) = self._second_order_moments(
            augmented_mean,
            augmented_covariance,
            measurement_matrix,
            multiplicative_noise_cov,
        )
        if self.covariance_regularization > 0.0:
            measurement_covariance = measurement_covariance + (
                self.covariance_regularization * eye(expected_measurement.shape[0])
            )

        predicted_measurement = measurement_matrix @ self.kinematic_state
        shifted_measurement = measurement - predicted_measurement
        pseudo_measurement = array(
            [
                shifted_measurement[0],
                shifted_measurement[1],
                shifted_measurement[0] ** 2,
                shifted_measurement[0] * shifted_measurement[1],
                shifted_measurement[1] ** 2,
            ]
        )

        cross_covariance = augmented_covariance @ jacobian.T
        gain = linalg.solve(measurement_covariance.T, cross_covariance.T).T
        updated_augmented_mean = augmented_mean + gain @ (
            pseudo_measurement - expected_measurement
        )
        updated_augmented_covariance = self._symmetrize(
            augmented_covariance - gain @ measurement_covariance @ gain.T
        )

        shape_start = kinematic_dim
        shape_stop = shape_start + 3
        self.kinematic_state = self.kinematic_state + updated_augmented_mean[:kinematic_dim]
        self.covariance = self._symmetrize(
            updated_augmented_covariance[:kinematic_dim, :kinematic_dim]
        )
        self.shape_state = updated_augmented_mean[shape_start:shape_stop]
        self.shape_covariance = self._symmetrize(
            updated_augmented_covariance[shape_start:shape_stop, shape_start:shape_stop]
        )


MemSoekfTracker = MEMSOEKFTracker
