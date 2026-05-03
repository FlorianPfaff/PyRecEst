"""Conversion registrations for SO(3) distribution representations."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, matmul, reshape, sum, transpose

from .conversion import register_conversion
from .so3_dirac_distribution import SO3DiracDistribution
from .so3_product_dirac_distribution import SO3ProductDiracDistribution
from .so3_product_tangent_gaussian_distribution import (
    SO3ProductTangentGaussianDistribution,
)
from .so3_tangent_gaussian_distribution import SO3TangentGaussianDistribution


def _sample_so3_product_dirac(distribution, n_particles):
    """Create an SO(3)^K Dirac approximation by sampling a source distribution."""
    if not isinstance(n_particles, int) or n_particles <= 0:
        raise ValueError("n_particles must be a positive integer")
    return SO3ProductDiracDistribution(distribution.sample(n_particles))


def _so3_tangent_gaussian_from_dirac(
    distribution,
    n_particles=None,
    covariance_regularization=0.0,
    check_validity=False,
):
    """Create an SO(3) tangent Gaussian approximation.

    Dirac inputs are moment-matched in the local tangent chart. Non-Dirac inputs
    can be converted by sampling first when ``n_particles`` is supplied.
    """
    if isinstance(distribution, SO3TangentGaussianDistribution):
        return SO3TangentGaussianDistribution(
            distribution.mean(),
            distribution.covariance(),
            check_validity=check_validity,
        )

    if not hasattr(distribution, "as_quaternions") or not hasattr(distribution, "w"):
        if n_particles is None:
            raise ValueError(
                "SO3 tangent Gaussian conversion requires a weighted SO(3) "
                "Dirac representation or n_particles for sampling-based "
                "approximation."
            )
        distribution = SO3DiracDistribution.from_distribution(distribution, n_particles)

    rotations = distribution.as_quaternions()
    weights = reshape(distribution.w, (-1, 1))
    initial_mean = distribution.mean()

    tangent_vectors = SO3TangentGaussianDistribution.log_map(
        rotations, base=initial_mean
    )
    tangent_mean = sum(weights * tangent_vectors, axis=0)
    refined_mean = SO3TangentGaussianDistribution.exp_map(
        tangent_mean, base=initial_mean
    )[0]

    residuals = SO3TangentGaussianDistribution.log_map(
        rotations, base=refined_mean
    )
    residual_mean = sum(weights * residuals, axis=0)
    centered = residuals - residual_mean
    covariance = matmul(transpose(weights * centered), centered)

    if covariance_regularization != 0.0:
        covariance = covariance + covariance_regularization * diag(
            array([1.0, 1.0, 1.0])
        )

    return SO3TangentGaussianDistribution(
        refined_mean, covariance, check_validity=check_validity
    )


def _so3_product_tangent_gaussian_from_dirac(
    distribution,
    n_particles=None,
    covariance_regularization=0.0,
    check_validity=False,
):
    """Create an SO(3)^K tangent Gaussian approximation.

    Dirac inputs are moment-matched in the local tangent chart. Non-Dirac inputs
    can be converted by sampling first when ``n_particles`` is supplied.
    """
    if isinstance(distribution, SO3ProductTangentGaussianDistribution):
        return SO3ProductTangentGaussianDistribution(
            distribution.mean(),
            distribution.covariance(),
            num_rotations=distribution.num_rotations,
            check_validity=check_validity,
        )

    if not hasattr(distribution, "as_quaternions") or not hasattr(distribution, "w"):
        if n_particles is None:
            raise ValueError(
                "SO3 product tangent Gaussian conversion requires a weighted "
                "SO(3)^K Dirac representation or n_particles for "
                "sampling-based approximation."
            )
        distribution = _sample_so3_product_dirac(distribution, n_particles)

    if not hasattr(distribution, "num_rotations"):
        raise ValueError(
            "SO(3)^K tangent Gaussian moment matching requires a product "
            "Dirac distribution with num_rotations."
        )

    rotations = distribution.as_quaternions()
    weights = reshape(distribution.w, (-1, 1))
    num_rotations = distribution.num_rotations
    initial_mean = distribution.mean()

    tangent_vectors = SO3ProductTangentGaussianDistribution.log_map(
        rotations, base=initial_mean, num_rotations=num_rotations
    )
    tangent_mean = sum(weights * tangent_vectors, axis=0)
    refined_mean = SO3ProductTangentGaussianDistribution.exp_map(
        tangent_mean, base=initial_mean, num_rotations=num_rotations
    )[0]

    residuals = SO3ProductTangentGaussianDistribution.log_map(
        rotations, base=refined_mean, num_rotations=num_rotations
    )
    residual_mean = sum(weights * residuals, axis=0)
    centered = residuals - residual_mean
    covariance = matmul(transpose(weights * centered), centered)

    if covariance_regularization != 0.0:
        covariance = covariance + covariance_regularization * diag(
            array([1.0] * (3 * num_rotations))
        )

    return SO3ProductTangentGaussianDistribution(
        refined_mean,
        covariance,
        num_rotations=num_rotations,
        check_validity=check_validity,
    )


register_conversion(
    SO3TangentGaussianDistribution,
    SO3DiracDistribution,
    SO3DiracDistribution.from_distribution,
    method="SO3DiracDistribution.from_distribution",
)
register_conversion(
    SO3ProductTangentGaussianDistribution,
    SO3ProductDiracDistribution,
    _sample_so3_product_dirac,
    method="SO3ProductDiracDistribution.from_distribution",
)
register_conversion(
    SO3DiracDistribution,
    SO3TangentGaussianDistribution,
    _so3_tangent_gaussian_from_dirac,
    method="SO3TangentGaussianDistribution.from_distribution",
)
register_conversion(
    SO3ProductDiracDistribution,
    SO3ProductTangentGaussianDistribution,
    _so3_product_tangent_gaussian_from_dirac,
    method="SO3ProductTangentGaussianDistribution.from_distribution",
)


__all__ = []
