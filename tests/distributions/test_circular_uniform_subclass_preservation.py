from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)


class _CircularUniformSubclass(CircularUniformDistribution):
    pass


def test_shift_preserves_runtime_subclass_and_state():
    dist = _CircularUniformSubclass()
    dist.marker = {"values": [1, 2, 3]}

    shifted = dist.shift(0.25)

    assert isinstance(shifted, _CircularUniformSubclass)
    assert shifted is not dist
    assert shifted.marker == dist.marker
    assert shifted.marker is not dist.marker
