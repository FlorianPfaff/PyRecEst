from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,duplicate-code
from pyrecest.backend import array, cos, eye, linalg, sin

from .mem_ekf_tracker import MEMEKFTracker


class MEMEKFStarTracker(MEMEKFTracker):
    """Moment-corrected MEM-EKF* tracker for one 2-D elliptical object.

    Compared with :class:`MEMEKFTracker`, this variant follows the MEM-EKF*
    moment approximation. It includes shape-parameter uncertainty in the
    measurement covariance and uses the covariance of the quadratic
    pseudo-measurement ``[dx^2, dy^2, dx * dy]``.
    """

    @staticmethod
    def _trace_3_by_3(matrix):
        return matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

    def _extent_row_jacobians(self):
        orientation, semi_axis_1, semi_axis_2 = self.shape_state
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

    def _shape_noise_covariance(self, multiplicative_noise_cov):
        first_row_jacobian, second_row_jacobian = self._extent_row_jacobians()

        def epsilon(row_jacobian_m, row_jacobian_n):
            return self._trace_3_by_3(
                self.shape_covariance
                @ row_jacobian_n.T
                @ multiplicative_noise_cov
                @ row_jacobian_m
            )

        shape_noise_covariance = array(
            [
                [
                    epsilon(first_row_jacobian, first_row_jacobian),
                    epsilon(first_row_jacobian, second_row_jacobian),
                ],
                [
                    epsilon(second_row_jacobian, first_row_jacobian),
                    epsilon(second_row_jacobian, second_row_jacobian),
                ],
            ]
        )
        return self._symmetrize(shape_noise_covariance)

    def _shape_pseudo_jacobian_star(self, multiplicative_noise_cov):
        first_row_jacobian, second_row_jacobian = self._extent_row_jacobians()
        extent_transform = self._extent_transform()
        first_extent_row = extent_transform[0, :]
        second_extent_row = extent_transform[1, :]

        return array(
            [
                2.0
                * first_extent_row
                @ multiplicative_noise_cov
                @ first_row_jacobian,
                2.0
                * second_extent_row
                @ multiplicative_noise_cov
                @ second_row_jacobian,
                first_extent_row @ multiplicative_noise_cov @ second_row_jacobian
                + second_extent_row @ multiplicative_noise_cov @ first_row_jacobian,
            ]
        )

    @staticmethod
    def _pseudo_measurement_covariance(innovation_covariance):
        sigma_11 = innovation_covariance[0, 0]
        sigma_12 = innovation_covariance[0, 1]
        sigma_22 = innovation_covariance[1, 1]
        return array(
            [
                [
                    2.0 * sigma_11**2,
                    2.0 * sigma_12**2,
                    2.0 * sigma_11 * sigma_12,
                ],
                [
                    2.0 * sigma_12**2,
                    2.0 * sigma_22**2,
                    2.0 * sigma_22 * sigma_12,
                ],
                [
                    2.0 * sigma_11 * sigma_12,
                    2.0 * sigma_22 * sigma_12,
                    sigma_11 * sigma_22 + sigma_12**2,
                ],
            ]
        )

    # pylint: disable=too-many-locals
    def _update_single_measurement(
        self,
        measurement,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
    ):
        extent_transform = self._extent_transform()
        shape_pseudo_jacobian = self._shape_pseudo_jacobian_star(
            multiplicative_noise_cov
        )
        shape_noise_covariance = self._shape_noise_covariance(multiplicative_noise_cov)

        predicted_measurement = measurement_matrix @ self.kinematic_state
        innovation = measurement - predicted_measurement
        innovation_covariance = self._symmetrize(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + extent_transform @ multiplicative_noise_cov @ extent_transform.T
            + shape_noise_covariance
            + meas_noise_cov
        )
        if self.covariance_regularization > 0.0:
            innovation_covariance = innovation_covariance + (
                self.covariance_regularization * eye(self.measurement_dim)
            )

        kinematic_cross_covariance = self.covariance @ measurement_matrix.T
        kinematic_gain = linalg.solve(
            innovation_covariance.T,
            kinematic_cross_covariance.T,
        ).T
        self.kinematic_state = self.kinematic_state + kinematic_gain @ innovation
        self.covariance = self._symmetrize(
            self.covariance - kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )

        pseudo_measurement = array(
            [innovation[0] ** 2, innovation[1] ** 2, innovation[0] * innovation[1]]
        )
        sigma_11 = innovation_covariance[0, 0]
        sigma_12 = innovation_covariance[0, 1]
        sigma_22 = innovation_covariance[1, 1]
        pseudo_mean = array([sigma_11, sigma_22, sigma_12])
        pseudo_covariance = self._pseudo_measurement_covariance(
            innovation_covariance
        )
        if self.covariance_regularization > 0.0:
            pseudo_covariance = pseudo_covariance + (
                self.covariance_regularization * eye(3)
            )

        shape_cross_covariance = self.shape_covariance @ shape_pseudo_jacobian.T
        shape_gain = linalg.solve(pseudo_covariance.T, shape_cross_covariance.T).T
        self.shape_state = self.shape_state + shape_gain @ (
            pseudo_measurement - pseudo_mean
        )
        self.shape_covariance = self._symmetrize(
            self.shape_covariance - shape_gain @ pseudo_covariance @ shape_gain.T
        )


MemEkfStarTracker = MEMEKFStarTracker
