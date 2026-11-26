"""Shared geometry and Gaussian moments for spatial multiplicative-error trackers."""

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from numbers import Integral

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs as backend_abs,
    all as backend_all,
    array,
    cast,
    concatenate,
    cos,
    diag,
    einsum,
    eye,
    float32,
    is_complex,
    isfinite as backend_isfinite,
    linalg,
    max as backend_max,
    maximum,
    sin,
    sqrt,
    stack,
)
from pyrecest.exceptions import ShapeError
from pyrecest.numerics import assert_covariance_matrix

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


def axis_squared_moments(axes, axis_covariance, source_covariance, additive_covariance):
    """Moments of squared body-frame measurements for uncertain Gaussian axes.

    For independent Gaussian a, h and v, y = diag(a) h + v. The source h and
    additive noise v are centered. Return E[y²], Cov(y²), and Cov(a, y²),
    where the square is componentwise. Leading batch dimensions broadcast.
    """
    axis_variance = einsum("...ii->...i", axis_covariance)
    source_variance = einsum("...ii->...i", source_covariance)
    mean_outer = axes[..., :, None] * axes[..., None, :]
    expected_products = mean_outer + axis_covariance
    expected_spread = additive_covariance + source_covariance * expected_products
    expected_squares = einsum("...ii->...i", expected_spread)

    # Conditional on a, Cov(y_i², y_j² | a) = 2 Cov(y_i, y_j | a)².
    # The total covariance also contains Cov(E[y_i² | a], E[y_j² | a]).
    squared_axes = axes**2
    variance_products = (
        squared_axes[..., :, None] * axis_variance[..., None, :]
        + axis_variance[..., :, None] * squared_axes[..., None, :]
        + 2 * mean_outer * axis_covariance
        + axis_variance[..., :, None] * axis_variance[..., None, :]
        + axis_covariance**2
    )
    covariance_of_squares = 4 * mean_outer * axis_covariance + 2 * axis_covariance**2
    pseudo_covariance = (
        2 * (expected_spread**2 + source_covariance**2 * variance_products)
        + source_variance[..., :, None]
        * source_variance[..., None, :]
        * covariance_of_squares
    )
    cross_covariance = 2 * axis_covariance * (axes * source_variance)[..., None, :]
    return expected_squares, pseudo_covariance, cross_covariance


class _SpatialMEMTracker(AbstractExtendedObjectTracker):
    """Common inputs and ellipsoid point estimates for 3D MEM trackers."""

    measurement_dim = 3
    shape_dim = 6

    @abstractmethod
    def _extent_transform(self):
        """Return a factor S such that the geometric extent equals S S.T."""

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
    def _as_real_array(cls, value, name):
        """Normalize real inputs to the active backend's floating precision."""
        value = array(value)
        if is_complex(value):
            raise ValueError(f"{name} must contain only real numeric values")
        cls._require_finite(value, name)
        dtype = array(0.0).dtype
        if value.dtype != dtype:
            value = cast(value, dtype)
            cls._require_finite(value, name)
        return value

    @classmethod
    def _as_vector(cls, value, dim, name):
        vector = cls._as_real_array(value, name)
        if vector.shape != (dim,):
            raise ShapeError(f"{name} must have shape ({dim},)")
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
        matrix = assert_covariance_matrix(matrix, name=name, dim=dim)
        return cls._as_real_array(matrix, name)

    @classmethod
    def _as_rotation(cls, value, name):
        rotation = cls._as_real_array(value, name)
        if rotation.shape != (3, 3):
            raise ShapeError(f"{name} must have shape (3, 3)")
        tolerance = 5e-7 if rotation.dtype == float32 else 1e-7
        if (
            float(backend_max(backend_abs(rotation.T @ rotation - eye(3)))) > tolerance
            or abs(float(linalg.det(rotation)) - 1.0) > tolerance
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
        matrix = self._as_real_array(measurement_matrix, "measurement_matrix")
        if matrix.shape != (3, self.kinematic_state.shape[0]):
            raise ShapeError("measurement_matrix must have shape (3, kinematic_dim)")
        return matrix

    def _normalize_measurements(self, measurements):
        measurements = self._as_real_array(measurements, "measurements")
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
