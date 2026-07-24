import pytest
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
    GSSVMDistribution,
)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda kappa: GeneralizedKSineSkewedVonMisesDistribution(
            mu=0.0,
            kappa=kappa,
            lambda_=0.25,
            k=1,
            m=1,
        ),
        lambda kappa: GSSVMDistribution(
            mu=0.0,
            kappa=kappa,
            lambda_=0.25,
            n=1,
        ),
    ],
)
def test_generalized_sine_skewed_von_mises_rejects_invalid_kappa(constructor):
    for kappa, message in (
        (-0.1, "nonnegative"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        ([1.0, 2.0], "scalar"),
    ):
        with pytest.raises(ValueError, match=message):
            constructor(kappa)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GeneralizedKSineSkewedVonMisesDistribution(
            mu=0.0,
            kappa=0.0,
            lambda_=0.25,
            k=1,
            m=1,
        ),
        lambda: GSSVMDistribution(
            mu=0.0,
            kappa=0.0,
            lambda_=0.25,
            n=1,
        ),
    ],
)
def test_generalized_sine_skewed_von_mises_accepts_zero_kappa(constructor):
    assert float(constructor().kappa) == 0.0
