import pytest
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
    GeneralizedKSineSkewedWrappedCauchyDistribution,
    GSSVMDistribution,
)


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
