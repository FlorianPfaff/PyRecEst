# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import (
    SO3BinghamDistribution,
    SO3TangentGaussianDistribution,
)


class _SO3BinghamSubclass(SO3BinghamDistribution):
    pass


class _SO3TangentGaussianSubclass(SO3TangentGaussianDistribution):
    pass


def _identity_quaternion():
    return array([0.0, 0.0, 0.0, 1.0])


def test_so3_bingham_multiply_preserves_left_subclass():
    left = _SO3BinghamSubclass.from_mode_and_concentration(
        _identity_quaternion(), 2.0
    )
    right = SO3BinghamDistribution.from_mode_and_concentration(
        _identity_quaternion(), 3.0
    )

    product = left.multiply(right)

    assert isinstance(product, _SO3BinghamSubclass)


def test_so3_bingham_compose_preserves_left_subclass():
    left = _SO3BinghamSubclass.from_mode_and_concentration(
        _identity_quaternion(), 2.0
    )
    right = SO3BinghamDistribution.from_mode_and_concentration(
        _identity_quaternion(), 3.0
    )

    composed = left.compose(right)

    assert isinstance(composed, _SO3BinghamSubclass)


def test_so3_tangent_gaussian_diagonal_factory_preserves_requested_subclass():
    distribution = _SO3TangentGaussianSubclass.from_covariance_diagonal(
        _identity_quaternion(), array([0.1, 0.2, 0.3])
    )

    assert isinstance(distribution, _SO3TangentGaussianSubclass)
