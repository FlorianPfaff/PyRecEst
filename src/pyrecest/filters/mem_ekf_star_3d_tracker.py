"""MEM-EKF* moment approximation for a spatial ellipsoid with SO(3) orientation."""

from __future__ import annotations

from math import isfinite
from numbers import Integral

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs as backend_abs,
    all as backend_all,
    array,
    concatenate,
    cos,
    diag,
    eye,
    isfinite as backend_isfinite,
    linalg,
    max as backend_max,
    maximum,
    sin,
    sqrt,
    stack,
    trace,
    zeros,
)
from pyrecest.numerics import assert_covariance_matrix
from pyrecest.exceptions import ShapeError

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class MEMEKFStar3DTracker(AbstractExtendedObjectTracker):
    """Sequential MEM-EKF* tracker for one fully oriented 3-D ellipsoid.

    Measurements obey ``z = H x + R diag(a) h + v``. ``orientation`` is a
    proper 3-by-3 rotation from the object's principal-axis frame to the world
    frame, and ``semi_axes`` contains three positive semi-axis lengths. The
    6-by-6 ``shape_covariance`` describes ``[delta_theta, delta_a]``, with
    right/body rotation errors: ``R_true = R exp(skew(delta_theta))``.
    Rotation/axis cross-covariances are retained. Orientation corrections use
    the SO(3) exponential and reset the covariance into the new tangent frame.

    This is a 3-D generalization of the MEM-EKF* moment approximation, not an
    exact Bayesian update. The initial orientation should be locally close to
    the desired ellipsoid frame. Axes retain their labels and are never sorted.
    Rotation about equal axes is unobservable for isotropic multiplicative
    noise. The default ``multiplicative_noise_cov = I / 5`` matches the second
    moment of uniform sources throughout a filled 3-D ellipsoid; fourth moments
    in the update use the MEM-EKF* Gaussian approximation.

    Covariances accept a scalar, a diagonal vector, or a square matrix.
    ``measurement_matrix`` defaults to selecting the first three kinematic
    coordinates. ``minimum_axis_length`` clips corrected axes to a positive
    lower bound. ``covariance_regularization`` adds a nonnegative diagonal term
    to the measurement and quadratic innovation covariances.
    """

    measurement_dim = 3
    shape_dim = 6
    _quadratic_pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def __init__(
        self,
        kinematic_state,
        covariance,
        orientation,
        semi_axes,
        shape_covariance,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        covariance_regularization=0.0,
        minimum_axis_length=1e-9,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.kinematic_state = array(kinematic_state)
        if self.kinematic_state.ndim != 1 or self.kinematic_state.shape[0] < 3:
            raise ShapeError(
                "kinematic_state must be a vector containing a 3-D position"
            )
        self._require_finite(self.kinematic_state, "kinematic_state")
        self.covariance = self._as_covariance_matrix(
            covariance, self.kinematic_state.shape[0], "covariance"
        )
        self.orientation = self._as_rotation(orientation, "orientation")
        self.semi_axes = self._as_vector(semi_axes, 3, "semi_axes")
        if not bool(backend_all(self.semi_axes > 0.0)):
            raise ValueError("semi_axes must be positive")
        self.shape_covariance = self._as_covariance_matrix(
            shape_covariance, 6, "shape_covariance"
        )
        self.measurement_matrix = (
            None
            if measurement_matrix is None
            else self._get_measurement_matrix(measurement_matrix)
        )
        self.multiplicative_noise_cov = self._as_multiplicative_noise(
            0.2 * eye(3)
            if multiplicative_noise_cov is None
            else multiplicative_noise_cov
        )
        self.covariance_regularization = float(covariance_regularization)
        if (
            not isfinite(self.covariance_regularization)
            or self.covariance_regularization < 0.0
        ):
            raise ValueError(
                "covariance_regularization must be finite and non-negative"
            )
        self.minimum_axis_length = float(minimum_axis_length)
        if not isfinite(self.minimum_axis_length) or self.minimum_axis_length <= 0.0:
            raise ValueError("minimum_axis_length must be finite and positive")

    @staticmethod
    def _require_finite(value, name):
        try:
            finite = bool(backend_all(backend_isfinite(value)))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must contain only finite numeric values"
            ) from error
        if not finite:
            raise ValueError(f"{name} must contain only finite values")

    @classmethod
    def _as_vector(cls, value, dim, name):
        vector = array(value)
        if vector.shape != (dim,):
            raise ShapeError(f"{name} must have shape ({dim},)")
        cls._require_finite(vector, name)
        return vector

    @classmethod
    def _as_covariance_matrix(cls, value, dim, name):
        matrix = array(value)
        if matrix.ndim == 0:
            matrix = matrix * eye(dim)
        elif matrix.ndim == 1 and matrix.shape == (dim,):
            matrix = diag(matrix)
        if matrix.shape != (dim, dim):
            raise ShapeError(f"{name} must have shape ({dim}, {dim})")
        return assert_covariance_matrix(matrix, name=name, dim=dim)

    @classmethod
    def _as_rotation(cls, value, name):
        rotation = array(value)
        if rotation.shape != (3, 3):
            raise ShapeError(f"{name} must have shape (3, 3)")
        cls._require_finite(rotation, name)
        if (
            float(backend_max(backend_abs(rotation.T @ rotation - eye(3)))) > 1e-7
            or abs(float(linalg.det(rotation)) - 1.0) > 1e-7
        ):
            raise ValueError(f"{name} must be a proper orthogonal rotation matrix")
        return rotation

    @classmethod
    def _as_multiplicative_noise(cls, value):
        covariance = cls._as_covariance_matrix(value, 3, "multiplicative_noise_cov")
        if float(linalg.eigvalsh(covariance)[0]) <= 0.0:
            raise ValueError("multiplicative_noise_cov must be positive definite")
        return covariance

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _project_covariance(cls, covariance):
        covariance = cls._symmetrize(covariance)
        values, vectors = linalg.eigh(covariance)
        if float(values[0]) < 0.0:
            covariance = cls._symmetrize((vectors * maximum(values, 0.0)) @ vectors.T)
        return covariance

    @staticmethod
    def _skew(vector):
        x, y, z = vector
        return array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    @classmethod
    def _rotation_exp(cls, rotation_vector):
        """SO(3) exponential, with continuous coefficients at zero angle."""
        rotation_vector = cls._as_vector(rotation_vector, 3, "rotation_vector")
        angle_squared = rotation_vector @ rotation_vector
        skew = cls._skew(rotation_vector)
        if float(angle_squared) < 1e-8:
            sine_coefficient = 1.0 - angle_squared / 6.0 + angle_squared**2 / 120.0
            cosine_coefficient = 0.5 - angle_squared / 24.0 + angle_squared**2 / 720.0
        else:
            angle = sqrt(angle_squared)
            sine_coefficient = sin(angle) / angle
            cosine_coefficient = (1.0 - cos(angle)) / angle_squared
        return eye(3) + sine_coefficient * skew + cosine_coefficient * (skew @ skew)

    @classmethod
    def _right_jacobian(cls, rotation_vector):
        """Transport an additive rotation-vector error after right retraction."""
        rotation_vector = cls._as_vector(rotation_vector, 3, "rotation_vector")
        angle_squared = rotation_vector @ rotation_vector
        skew = cls._skew(rotation_vector)
        if float(angle_squared) < 1e-8:
            first = 0.5 - angle_squared / 24.0 + angle_squared**2 / 720.0
            second = 1.0 / 6.0 - angle_squared / 120.0 + angle_squared**2 / 5040.0
        else:
            angle = sqrt(angle_squared)
            first = (1.0 - cos(angle)) / angle_squared
            second = (angle - sin(angle)) / (angle_squared * angle)
        return eye(3) - first * skew + second * (skew @ skew)

    def _get_measurement_matrix(self, measurement_matrix=None):
        if measurement_matrix is None:
            stored_matrix = getattr(self, "measurement_matrix", None)
            return (
                eye(3, self.kinematic_state.shape[0])
                if stored_matrix is None
                else stored_matrix
            )
        matrix = array(measurement_matrix)
        if matrix.shape != (3, self.kinematic_state.shape[0]):
            raise ShapeError("measurement_matrix must have shape (3, kinematic_dim)")
        self._require_finite(matrix, "measurement_matrix")
        return matrix

    def _normalize_measurements(self, measurements):
        measurements = array(measurements)
        self._require_finite(measurements, "measurements")
        if measurements.ndim == 1:
            if measurements.shape == (0,):
                return measurements.reshape((3, 0))
            if measurements.shape == (3,):
                return measurements.reshape((3, 1))
        elif measurements.ndim == 2:
            if measurements.shape[0] == 3:
                return measurements
            if measurements.shape[1] == 3:
                return measurements.T
        raise ShapeError("measurements must have shape (3,), (3, n), or (n, 3)")

    def _extent_transform(self):
        return self.orientation @ diag(self.semi_axes)

    def _extent_row_jacobians(self):
        axes = diag(self.semi_axes)
        return stack(
            [
                concatenate([axes @ self._skew(row), diag(row)], axis=1)
                for row in self.orientation
            ]
        )

    def _shape_noise_covariance(self, multiplicative_noise_cov):
        jacobians = self._extent_row_jacobians()
        return self._symmetrize(
            array(
                [
                    [
                        trace(
                            self.shape_covariance
                            @ jacobians[j].T
                            @ multiplicative_noise_cov
                            @ jacobians[i]
                        )
                        for j in range(3)
                    ]
                    for i in range(3)
                ]
            )
        )

    def _shape_pseudo_jacobian(self, multiplicative_noise_cov):
        extent = self._extent_transform()
        jacobians = self._extent_row_jacobians()
        return stack(
            [
                extent[i] @ multiplicative_noise_cov @ jacobians[j]
                + extent[j] @ multiplicative_noise_cov @ jacobians[i]
                for i, j in self._quadratic_pairs
            ]
        )

    @classmethod
    def _pseudo_measurement_covariance(cls, innovation_covariance):
        return array(
            [
                [
                    innovation_covariance[i, k] * innovation_covariance[j, ell]
                    + innovation_covariance[i, ell] * innovation_covariance[j, k]
                    for k, ell in cls._quadratic_pairs
                ]
                for i, j in cls._quadratic_pairs
            ]
        )

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        """Process target-originated points sequentially; a 3-by-3 set uses columns."""
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return
        measurement_matrix = self._get_measurement_matrix(meas_mat)
        measurement_noise = (
            zeros((3, 3))
            if meas_noise_cov is None
            else self._as_covariance_matrix(meas_noise_cov, 3, "meas_noise_cov")
        )
        multiplicative_noise = (
            self.multiplicative_noise_cov
            if multiplicative_noise_cov is None
            else self._as_multiplicative_noise(multiplicative_noise_cov)
        )
        for index in range(measurements.shape[1]):
            self._update_single_measurement(
                measurements[:, index],
                measurement_matrix,
                measurement_noise,
                multiplicative_noise,
            )
        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _update_single_measurement(
        self, measurement, measurement_matrix, measurement_noise, multiplicative_noise
    ):
        extent = self._extent_transform()
        pseudo_jacobian = self._shape_pseudo_jacobian(multiplicative_noise)
        innovation = measurement - measurement_matrix @ self.kinematic_state
        innovation_covariance = self._symmetrize(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + extent @ multiplicative_noise @ extent.T
            + self._shape_noise_covariance(multiplicative_noise)
            + measurement_noise
            + self.covariance_regularization * eye(3)
        )
        kinematic_cross_covariance = self.covariance @ measurement_matrix.T
        kinematic_gain = linalg.solve(
            innovation_covariance, kinematic_cross_covariance.T
        ).T
        kinematic_state = self.kinematic_state + kinematic_gain @ innovation
        covariance = self._project_covariance(
            self.covariance - kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )

        pseudo_measurement = array(
            [innovation[i] * innovation[j] for i, j in self._quadratic_pairs]
        )
        pseudo_mean = array(
            [innovation_covariance[i, j] for i, j in self._quadratic_pairs]
        )
        pseudo_covariance = self._pseudo_measurement_covariance(innovation_covariance)
        pseudo_covariance = pseudo_covariance + self.covariance_regularization * eye(6)
        shape_cross_covariance = self.shape_covariance @ pseudo_jacobian.T
        shape_gain = linalg.solve(pseudo_covariance, shape_cross_covariance.T).T
        correction = shape_gain @ (pseudo_measurement - pseudo_mean)
        shape_covariance = self._project_covariance(
            self.shape_covariance - shape_gain @ pseudo_covariance @ shape_gain.T
        )
        reset = linalg.block_diag(self._right_jacobian(correction[:3]), eye(3))
        orientation = self.orientation @ self._rotation_exp(correction[:3])
        semi_axes = maximum(self.semi_axes + correction[3:], self.minimum_axis_length)
        shape_covariance = self._project_covariance(reset @ shape_covariance @ reset.T)
        for value in (
            kinematic_state,
            covariance,
            orientation,
            semi_axes,
            shape_covariance,
        ):
            self._require_finite(value, "posterior state")
        self.kinematic_state = kinematic_state
        self.covariance = covariance
        self.orientation = orientation
        self.semi_axes = semi_axes
        self.shape_covariance = shape_covariance

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_sys_noise=None,
        rotation_increment=None,
    ):
        """Predict kinematics and a constant-axis shape with optional body rotation.

        ``rotation_increment`` is a deterministic rotation vector applied on the
        right. Existing body error covariance is transported by the inverse
        increment's adjoint. ``shape_sys_noise`` is then added in the predicted
        body frame. Omitted increments/noises give constant-shape prediction.
        All inputs are validated before the tracker state changes.
        """
        state_dim = self.kinematic_state.shape[0]
        system_matrix = array(system_matrix)
        if system_matrix.shape != (state_dim, state_dim):
            raise ShapeError(
                "system_matrix must have shape (kinematic_dim, kinematic_dim)"
            )
        self._require_finite(system_matrix, "system_matrix")
        process_noise = (
            zeros((state_dim, state_dim))
            if sys_noise is None
            else self._as_covariance_matrix(sys_noise, state_dim, "sys_noise")
        )
        control = (
            zeros(state_dim)
            if inputs is None
            else self._as_vector(inputs, state_dim, "inputs")
        )
        shape_noise = (
            zeros((6, 6))
            if shape_sys_noise is None
            else self._as_covariance_matrix(shape_sys_noise, 6, "shape_sys_noise")
        )
        increment = (
            eye(3)
            if rotation_increment is None
            else self._rotation_exp(rotation_increment)
        )
        transport = linalg.block_diag(increment.T, eye(3))
        kinematic_state = system_matrix @ self.kinematic_state + control
        covariance = self._project_covariance(
            system_matrix @ self.covariance @ system_matrix.T + process_noise
        )
        orientation = self.orientation @ increment
        shape_covariance = self._project_covariance(
            transport @ self.shape_covariance @ transport.T + shape_noise
        )
        for value in (kinematic_state, covariance, orientation, shape_covariance):
            self._require_finite(value, "predicted state")
        self.kinematic_state = kinematic_state
        self.covariance = covariance
        self.orientation = orientation
        self.shape_covariance = shape_covariance
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict(self, *args, **kwargs):
        """Alias for :meth:`predict_linear`."""
        self.predict_linear(*args, **kwargs)

    @property
    def extent(self):
        return self.get_point_estimate_extent()

    def get_point_estimate(self):
        """Return the kinematic state followed by the flattened 3-by-3 extent."""
        return concatenate([self.kinematic_state, self.get_point_estimate_extent(True)])

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        """Return ``R diag(semi_axes**2) R.T``, without multiplicative scaling."""
        extent = self._extent_transform()
        extent = self._symmetrize(extent @ extent.T)
        return extent.flatten() if flatten_matrix else extent

    def get_contour_points(self, n, scaling_factor=1.0):
        """Return ``n`` surface points (n-by-3) using a Fibonacci sphere."""
        if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
            raise ValueError("n must be a positive integer")
        scaling_factor = float(scaling_factor)
        if not isfinite(scaling_factor) or scaling_factor <= 0.0:
            raise ValueError("scaling_factor must be finite and positive")
        golden_angle = 2.399963229728653
        points = []
        for index in range(n):
            height = 1.0 - 2.0 * (index + 0.5) / n
            radius = sqrt(1.0 - height**2)
            angle = golden_angle * index
            points.append(array([radius * cos(angle), radius * sin(angle), height]))
        unit_sphere = stack(points, axis=1)
        center = self._get_measurement_matrix() @ self.kinematic_state
        return (
            center[:, None] + scaling_factor * self._extent_transform() @ unit_sphere
        ).T

    def plot_point_estimate(self, axes=None, n=200):
        """Draw the ellipsoid surface points on a Matplotlib 3-D axes."""
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        if axes is None:
            axes = plt.figure().add_subplot(projection="3d")
        points = self.get_contour_points(n)
        axes.scatter(points[:, 0], points[:, 1], points[:, 2])
        return axes


MemEkfStar3DTracker = MEMEKFStar3DTracker
