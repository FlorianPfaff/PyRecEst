import copy
import warnings

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    atleast_2d,
    full,
    hstack,
    imag,
    isnan,
    pi,
    real,
    sqrt,
    vstack,
    zeros,
)
from scipy.linalg import norm

from ..abstract_orthogonal_basis_distribution import AbstractOrthogonalBasisDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution


class AbstractSphericalHarmonicsDistribution(
    AbstractSphericalDistribution, AbstractOrthogonalBasisDistribution
):
    def __init__(self, coeff_mat, transformation="identity"):
        AbstractSphericalDistribution.__init__(self)
        coeff_mat = atleast_2d(coeff_mat)
        assert (
            coeff_mat.shape[1] == coeff_mat.shape[0] * 2 - 1
        ), "CoefficientMatrix:Size, Dimensions of coefficient Matrix are incompatible."

        # Ignore irrelevant entries of coeff_mat and set to NaN
        n = coeff_mat.shape[0]

        for i in range(n):
            if pyrecest.backend.__backend_name__ in (
                "numpy",
                "pytorch",
            ):  # Set the irrelevant elements to nan
                coeff_mat[i, 2 * i + 1 :] = float("NaN")  # noqa: E203
            else:
                assert coeff_mat[i, 2 * i + 1 :].shape[0] == 0 or all(  # noqa: E203
                    isnan(coeff_mat[i, 2 * i + 1 :])  # noqa: E203
                    | (coeff_mat[i, 2 * i + 1 :] == 0)  # noqa: E203
                )
        AbstractOrthogonalBasisDistribution.__init__(self, coeff_mat, transformation)

    def pdf(self, xs):
        return AbstractOrthogonalBasisDistribution.pdf(self, xs)

    def normalize_in_place(self):
        int_val = self.integrate()
        if int_val < 0:
            warnings.warn(
                "Warning: Normalization:negative - Coefficient for first degree is negative. "
                "This can either be caused by a user error or due to negativity caused by "
                "non-square rooted version"
            )
        elif abs(int_val) < 1e-12:
            raise ValueError(
                "Normalization:almostZero - Coefficient for first degree is too close to zero, "
                "this usually points to a user error"
            )
        elif abs(int_val - 1) > 1e-5:
            warnings.warn(
                "Warning: Normalization:notNormalized - Coefficients apparently do not belong "
                "to normalized density. Normalizing..."
            )
        else:
            return

        if self.transformation == "identity":
            self.coeff_mat = self.coeff_mat / int_val
        elif self.transformation == "sqrt":
            self.coeff_mat = self.coeff_mat / sqrt(int_val)
        else:
            warnings.warn("Warning: Currently cannot normalize")

    def integrate(self):
        if self.transformation == "identity":
            int_val = self.coeff_mat[0, 0] * sqrt(4.0 * pi)
        elif self.transformation == "sqrt":
            int_val = norm(self.coeff_mat[~isnan(self.coeff_mat)]) ** 2
        else:
            raise ValueError("No analytical formula for normalization available")

        assert abs(imag(int_val)) < 1e-8
        return real(int_val)

    def truncate(self, degree):
        trun_result = copy.deepcopy(self)

        current_max_degree = self.coeff_mat.shape[0] - 1

        if current_max_degree > degree:
            # Truncate the matrix for a lower max_degree
            new_coeff_mat = self.coeff_mat[: degree + 1, : 2 * degree + 1]
        elif current_max_degree < degree:
            # Pad the matrix for a higher max_degree
            warnings.warn("Less coefficients than desired, filling up with zeros")

            # Create the blocks
            upper_left_block = self.coeff_mat
            upper_right_block = full(
                (self.coeff_mat.shape[0], 2 * degree + 1 - self.coeff_mat.shape[1]),
                float("nan"),
            )
            lower_block_shape = (degree + 1 - self.coeff_mat.shape[0], 2 * degree + 1)
            lower_block = zeros(lower_block_shape, dtype=self.coeff_mat.dtype)

            # Compose the new matrix using block matrices
            new_coeff_mat = vstack(
                [hstack([upper_left_block, upper_right_block]), lower_block]
            )
        else:
            new_coeff_mat = self.coeff_mat

        trun_result.coeff_mat = new_coeff_mat

        return trun_result
