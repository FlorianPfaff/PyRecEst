# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution


class _VonMisesSubclass(VonMisesDistribution):
    pass


def test_from_moment_preserves_requested_subclass_for_regular_and_uniform_results():
    source = VonMisesDistribution(array(0.3), array(2.0))

    regular = _VonMisesSubclass.from_moment(source.trigonometric_moment(1))
    uniform = _VonMisesSubclass.from_moment(array(0.0 + 0.0j))

    assert isinstance(regular, _VonMisesSubclass)
    assert isinstance(uniform, _VonMisesSubclass)


def test_multiply_preserves_left_subclass():
    left = _VonMisesSubclass(array(0.2), array(1.5))
    right = VonMisesDistribution(array(1.1), array(0.7))

    multiplied = left.multiply(right)

    assert isinstance(multiplied, _VonMisesSubclass)


def test_convolve_preserves_left_subclass_for_regular_and_uniform_results():
    left = _VonMisesSubclass(array(0.2), array(1.5))
    regular = VonMisesDistribution(array(1.1), array(0.7))
    uniform = VonMisesDistribution(array(1.1), array(0.0))

    assert isinstance(left.convolve(regular), _VonMisesSubclass)
    assert isinstance(left.convolve(uniform), _VonMisesSubclass)
