# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)


class _VonMisesFisherSubclass(VonMisesFisherDistribution):
    pass


def test_conversion_factory_preserves_requested_subclass():
    source = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 2.0)

    converted = convert_distribution(source, _VonMisesFisherSubclass)

    assert isinstance(converted, _VonMisesFisherSubclass)


def test_mean_resultant_factory_preserves_requested_subclass():
    converted = _VonMisesFisherSubclass.from_mean_resultant_vector(
        array([0.2, 0.0, 0.0])
    )

    assert isinstance(converted, _VonMisesFisherSubclass)


def test_multiply_preserves_left_subclass_for_regular_and_uniform_results():
    left = _VonMisesFisherSubclass(array([1.0, 0.0, 0.0]), 1.0)
    regular = VonMisesFisherDistribution(array([0.0, 1.0, 0.0]), 2.0)
    cancelling = VonMisesFisherDistribution(array([-1.0, 0.0, 0.0]), 1.0)

    assert isinstance(left.multiply(regular), _VonMisesFisherSubclass)
    assert isinstance(left.multiply(cancelling), _VonMisesFisherSubclass)


def test_convolve_preserves_left_subclass_for_regular_and_uniform_results():
    left = _VonMisesFisherSubclass(array([1.0, 0.0, 0.0]), 2.0)
    zonal = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)
    uniform = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 0.0)

    assert isinstance(left.convolve(zonal), _VonMisesFisherSubclass)
    assert isinstance(left.convolve(uniform), _VonMisesFisherSubclass)
