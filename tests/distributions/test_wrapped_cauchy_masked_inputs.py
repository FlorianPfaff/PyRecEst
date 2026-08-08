import numpy as np
import pytest

from pyrecest.distributions import WrappedCauchyDistribution


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("mu", np.ma.array(0.3, mask=True)),
        ("gamma", np.ma.array(0.8, mask=True)),
        ("mu", np.ma.masked),
        ("gamma", np.ma.masked),
    ),
)
def test_wrapped_cauchy_rejects_masked_parameters(parameter, value) -> None:
    kwargs = {"mu": 0.3, "gamma": 0.8}
    kwargs[parameter] = value

    with pytest.raises(ValueError, match=parameter):
        WrappedCauchyDistribution(**kwargs)


def test_wrapped_cauchy_rejects_masked_evaluation_inputs() -> None:
    distribution = WrappedCauchyDistribution(mu=0.3, gamma=0.8)

    with pytest.raises(ValueError, match="xs"):
        distribution.pdf(np.ma.array([0.2, 0.4], mask=[False, True]))
    with pytest.raises(ValueError, match="xs"):
        distribution.pdf([np.ma.array(0.2, mask=True)])
    with pytest.raises(ValueError, match="starting_point"):
        distribution.cdf([0.4], starting_point=np.ma.array(0.0, mask=True))


def test_wrapped_cauchy_accepts_clear_mask_wrappers() -> None:
    distribution = WrappedCauchyDistribution(
        mu=np.ma.array(0.3, mask=False),
        gamma=np.ma.array(0.8, mask=False),
    )

    density = distribution.pdf(np.ma.array([0.2, 0.4], mask=False))
    cdf = distribution.cdf(
        np.ma.array([0.4], mask=False),
        starting_point=np.ma.array(0.0, mask=False),
    )

    assert np.all(np.isfinite(np.asarray(density)))
    assert np.all(np.isfinite(np.asarray(cdf)))
