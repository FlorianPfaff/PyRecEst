from numbers import Integral
from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    int32,
    int64,
    is_complex,
    linalg,
    random,
    reshape,
    where,
    zeros,
)
from pyrecest.exceptions import ShapeError, ValidationError

from .abstract_ellipsoidal_ball_distribution import AbstractEllipsoidalBallDistribution
from .abstract_uniform_distribution import AbstractUniformDistribution


class EllipsoidalBallUniformDistribution(
    AbstractEllipsoidalBallDistribution, AbstractUniformDistribution
):
    """A class representing a uniform distribution on an ellipsoidal ball."""

    def __init__(self, center, shape_matrix):
        """
        Initialize EllipsoidalBallUniformDistribution.

        :param center: Center of the ellipsoidal ball.
        :param shape_matrix: Shape matrix defining the ellipsoidal ball.
        """
        AbstractUniformDistribution.__init__(self)
        AbstractEllipsoidalBallDistribution.__init__(self, center, shape_matrix)

    @property
    def input_dim(self) -> int:
        """Returns the size of the input vector for evaluation of the pdf."""
        return self.dim

    def mean(self):
        """Return the mean of the uniform distribution on the ellipsoidal ball."""
        return self.center

    def covariance(self):
        """Return the covariance matrix of the uniform ellipsoidal ball.

        For a uniform distribution on {center + L u : ||u|| <= 1}, with
        L @ L.T = shape_matrix and dimension d, Cov[u] = I / (d + 2), hence
        Cov[center + L u] = shape_matrix / (d + 2).
        """
        return self.shape_matrix / (self.dim + 2)

    def pdf(self, xs):
        """
        Compute the probability density function at given points.

        :param xs: Points at which to compute the PDF.
        :returns: PDF values at given points.
        """
        xs, single = self._coerce_points(xs)

        reciprocal_volume = 1 / self.get_manifold_size()

        # (n, dim)
        diff = xs - self.center[None, :]

        # Solve S * y = diff^T  -> y^T = diff^T * S^{-1}
        # S: (dim, dim), diff.T: (dim, n)  => solved.T: (n, dim)
        solved = linalg.solve(self.shape_matrix, diff.T).T

        # Quadratic form per row: sum_i diff_i * solved_i
        quad = (diff * solved).sum(axis=1)

        # Optional tiny tolerance near the boundary:
        inside = quad <= 1.0

        pdf_values = where(inside, reciprocal_volume, zeros(quad.shape[0]))

        return pdf_values[0] if single else pdf_values

    def _coerce_points(self, xs):
        xs = array(xs)
        if is_complex(xs):
            raise ValidationError("xs must be real-valued")
        if xs.ndim == 0:
            if self.dim != 1:
                raise ShapeError(
                    "xs",
                    xs.shape,
                    expected=f"({self.dim},) or (n, {self.dim})",
                    reason="scalar points are only valid for dim == 1",
                )
            return reshape(xs, (1, 1)), True
        if xs.ndim == 1:
            if self.dim == 1:
                return reshape(xs, (-1, 1)), False
            if xs.shape[0] == self.dim:
                return reshape(xs, (1, self.dim)), True
            raise ShapeError(
                "xs",
                xs.shape,
                expected=f"({self.dim},) or (n, {self.dim})",
            )
        if xs.ndim != 2 or xs.shape[1] != self.dim:
            raise ShapeError(
                "xs",
                xs.shape,
                expected=f"({self.dim},) or (n, {self.dim})",
            )
        return xs, False

    def sample(self, n: Union[int, int32, int64]):
        """
        Generate samples from the distribution.

        :param n: Number of samples to generate.
        :returns: Generated samples.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        n = int(n)
        random_points = random.normal(size=(n, self.dim))
        random_points /= linalg.norm(random_points, axis=1).reshape(-1, 1)

        random_radii = random.uniform(size=(n, 1))  # So that broadcasting works below
        random_radii = random_radii ** (
            1 / self.dim
        )  # Consider that the ellipsoid surfaces with higher radii are larger

        # Scale random points by the radii
        random_points *= random_radii

        # Rotate the points according to the shape matrix
        L = linalg.cholesky(self.shape_matrix)
        # For points (d, n), this would be L @ random_points
        transformed_points = random_points @ L.T + self.center.reshape(1, -1)

        return transformed_points
