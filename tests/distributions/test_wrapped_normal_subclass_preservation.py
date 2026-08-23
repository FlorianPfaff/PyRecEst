# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution


class _WrappedNormalSubclass(WrappedNormalDistribution):
    pass


def test_from_moment_preserves_requested_subclass():
    source = WrappedNormalDistribution(array(0.3), array(0.8))

    reconstructed = _WrappedNormalSubclass.from_moment(source.trigonometric_moment(1))

    assert isinstance(reconstructed, _WrappedNormalSubclass)


def test_shift_preserves_runtime_subclass():
    dist = _WrappedNormalSubclass(array(0.3), array(0.8))

    shifted = dist.shift(0.2)

    assert isinstance(shifted, _WrappedNormalSubclass)


def test_convolve_preserves_left_subclass():
    left = _WrappedNormalSubclass(array(0.3), array(0.8))
    right = WrappedNormalDistribution(array(0.4), array(0.6))

    convolved = left.convolve(right)

    assert isinstance(convolved, _WrappedNormalSubclass)


def test_multiply_paths_preserve_left_subclass():
    left = _WrappedNormalSubclass(array(0.3), array(0.8))
    right = WrappedNormalDistribution(array(0.4), array(0.6))

    assert isinstance(left.multiply(right), _WrappedNormalSubclass)
    assert isinstance(left.multiply_vm_approximation(right), _WrappedNormalSubclass)
    assert isinstance(left.multiply_vm(right), _WrappedNormalSubclass)
