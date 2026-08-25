from math import lgamma, log as scalar_log, pi as scalar_pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    allclose,
    array,
    diagonal,
    exp,
    is_complex,
    isfinite,
    linalg,
    log,
    pi,
    sqrt,
    transpose,
)
from pyrecest.exceptions import ShapeError, ValidationError

from .abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)


class AbstractEllipsoidalBallDistribution(AbstractBoundedNonPeriodicDistribution):
    """
    This class represents distributions on ellipsoidal balls.
    """

    def __init__(self, center, shape_matrix):
        """
        Initialize the class with a center and shape matrix.

        :param center: The center of the ellipsoidal ball.
        :param shape_matrix: The shape matrix of the ellipsoidal ball.
        """
        center = array(center)
        shape_matrix = array(shape_matrix)

        if center.ndim != 1:
            raise ShapeError("center", center.shape, expected="(dim,)")
        AbstractBoundedNonPeriodicDistribution.__init__(self, center.shape[-1])
        if shape_matrix.ndim != 2:
            raise ShapeError(
                "shape_matrix",
                shape_matrix.shape,
                expected=f"({self.dim}, {self.dim})",
            )
        if shape_matrix.shape != (self.dim, self.dim):
            raise ShapeError(
                "shape_matrix",
                shape_matrix.shape,
                expected=f"({self.dim}, {self.dim})",
                reason="shape_matrix must match the center dimension",
            )
        if is_complex(center):
            raise ValidationError("center must be real-valued")
        if is_complex(shape_matrix):
            raise ValidationError("shape_matrix must be real-valued")
        if not bool(backend_all(isfinite(center))):
            raise ValidationError("center must contain only finite values")
        if not bool(backend_all(isfinite(shape_matrix))):
            raise ValidationError("shape_matrix must contain only finite values")
        if not bool(allclose(shape_matrix, transpose(shape_matrix))):
            raise ValidationError("shape_matrix must be symmetric")

        shape_diagonal = diagonal(shape_matrix)
        if not bool(backend_all(shape_diagonal > 0.0)):
            raise ValidationError("shape_matrix must be positive definite")
        diagonal_scale = sqrt(shape_diagonal)
        scaled_shape_matrix = (
            shape_matrix / diagonal_scale[:, None] / diagonal_scale[None, :]
        )
        if not bool(backend_all(isfinite(scaled_shape_matrix))) or not bool(
            backend_all(linalg.eigvalsh(scaled_shape_matrix) > 0.0)
        ):
            raise ValidationError("shape_matrix must be positive definite")

        self.center = center
        self.shape_matrix = shape_matrix

    def get_manifold_size(self):
        """
        Calculate the size of the manifold.

        :returns: The size of the manifold.
        """
        # Handle cases with dimensions up to 4 directly
        if self.dim == 0:
            return 1

        if self.dim == 1:
            c = 2
        elif self.dim == 2:
            c = pi
        elif self.dim == 3:
            c = 4 / 3 * pi
        elif self.dim == 4:
            c = 0.5 * pi**2
        else:
            half_dim = self.dim / 2
            # Keep the unit-ball volume in log space. gamma(half_dim + 1)
            # overflows starting around dim=342 even while the volume itself
            # is still representable in float64.
            log_unit_ball_volume = half_dim * scalar_log(scalar_pi) - lgamma(
                half_dim + 1
            )

        if self.dim <= 4:
            log_unit_ball_volume = log(array(c))

        cholesky_factor = linalg.cholesky(self.shape_matrix)
        log_volume = log_unit_ball_volume + log(diagonal(cholesky_factor)).sum()
        return exp(log_volume)
