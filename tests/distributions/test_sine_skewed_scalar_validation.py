import pytest
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
    GeneralizedKSineSkewedWrappedCauchyDistribution,
    GSSVMDistribution,
    SineSkewedWrappedCauchyDistribution,
    SineSkewedWrappedNormalDistribution,
)


@pytest.fixture(
    params=(
        pytest.param(
            lambda mu: GeneralizedKSineSkewedVonMisesDistribution(
                mu=mu,
                kappa=1.0,
                lambda_=0.25,
                k=1,
                m=1,
            ),
            id="generalized-von-mises",
        ),
        pytest.param(
            lambda mu: GeneralizedKSineSkewedWrappedCauchyDistribution(
                mu=mu,
                gamma=0.5,
                lambda_=0.25,
                k=1,
                m=1,
            ),
            id="generalized-wrapped-cauchy",
        ),
        pytest.param(
            lambda mu: SineSkewedWrappedNormalDistribution(
                mu=mu,
                sigma=0.5,
                lambda_=0.25,
            ),
            id="wrapped-normal",
        ),
        pytest.param(
            lambda mu: SineSkewedWrappedCauchyDistribution(
                mu=mu,
                gamma=0.5,
                lambda_=0.25,
            ),
            id="wrapped-cauchy",
        ),
    )
)
def sine_skewed_constructor(request):
    return request.param


def test_sine_skewed_distributions_reject_vector_mu(sine_skewed_constructor):
    with pytest.raises(ValueError, match="mu must be a scalar"):
        sine_skewed_constructor([0.0, 0.5])


@pytest.mark.parametrize("mu", [float("nan"), float("inf"), float("-inf")])
def test_sine_skewed_distributions_reject_nonfinite_mu(
    sine_skewed_constructor,
    mu,
):
    with pytest.raises(ValueError, match="mu must be finite"):
        sine_skewed_constructor(mu)


def test_generalized_sine_skewed_von_mises_rejects_vector_lambda():
    with pytest.raises(ValueError, match="lambda_ must be a scalar"):
        GeneralizedKSineSkewedVonMisesDistribution(
            mu=0.0,
            kappa=1.0,
            lambda_=[0.25, 0.5],
            k=1,
            m=1,
        )


def test_gssvm_rejects_vector_lambda():
    with pytest.raises(ValueError, match="lambda_ must be a scalar"):
        GSSVMDistribution(mu=0.0, kappa=1.0, lambda_=[0.25, 0.5], n=1)


def test_generalized_sine_skewed_wrapped_cauchy_rejects_vector_parameters():
    with pytest.raises(ValueError, match="gamma must be a scalar"):
        GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=0.0,
            gamma=[0.25, 0.5],
            lambda_=0.25,
            k=1,
            m=1,
        )

    with pytest.raises(ValueError, match="lambda_ must be a scalar"):
        GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=0.0,
            gamma=0.5,
            lambda_=[0.25, 0.5],
            k=1,
            m=1,
        )


@pytest.mark.parametrize("gamma", [float("nan"), float("inf")])
def test_generalized_sine_skewed_wrapped_cauchy_rejects_nonfinite_gamma(gamma):
    with pytest.raises(ValueError, match="gamma must be finite"):
        GeneralizedKSineSkewedWrappedCauchyDistribution(
            mu=0.0,
            gamma=gamma,
            lambda_=0.25,
            k=1,
            m=1,
        )
