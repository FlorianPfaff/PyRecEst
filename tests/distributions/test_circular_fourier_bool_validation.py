import numpy as np
import pytest

from pyrecest.backend import array
from pyrecest.distributions import CircularFourierDistribution, VonMisesDistribution


@pytest.mark.parametrize("invalid_flag", ["False", "True", 0, 1, None])
def test_constructor_rejects_non_boolean_multiplied_by_n(invalid_flag):
    with pytest.raises(TypeError, match="multiplied_by_n must be a boolean"):
        CircularFourierDistribution(
            c=array([1.0]),
            n=1,
            transformation="identity",
            multiplied_by_n=invalid_flag,
        )


@pytest.mark.parametrize("invalid_flag", ["False", "True", 0, 1, None])
def test_function_value_factory_rejects_non_boolean_scaling_flag(invalid_flag):
    with pytest.raises(
        TypeError, match="store_values_multiplied_by_n must be a boolean"
    ):
        CircularFourierDistribution.from_function_values(
            array([1.0, 1.0, 1.0]),
            transformation="identity",
            store_values_multiplied_by_n=invalid_flag,
        )


def test_distribution_factory_rejects_truthy_string_scaling_flag():
    distribution = VonMisesDistribution(array(0.0), array(1.0))

    with pytest.raises(
        TypeError, match="store_values_multiplied_by_n must be a boolean"
    ):
        CircularFourierDistribution.from_distribution(
            distribution,
            n=3,
            transformation="identity",
            store_values_multiplied_by_n="False",
        )


def test_numpy_boolean_scaling_flags_are_accepted():
    direct = CircularFourierDistribution(
        c=array([1.0]),
        n=1,
        transformation="identity",
        multiplied_by_n=np.bool_(False),
    )
    from_values = CircularFourierDistribution.from_function_values(
        array([1.0, 1.0, 1.0]),
        transformation="identity",
        store_values_multiplied_by_n=np.bool_(False),
    )

    assert direct.multiplied_by_n is False
    assert from_values.multiplied_by_n is False
