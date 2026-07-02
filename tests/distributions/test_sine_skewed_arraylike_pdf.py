import numpy as np
import pyrecest.backend
from pyrecest.backend import pi
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedVonMisesDistribution,
    GeneralizedKSineSkewedWrappedCauchyDistribution,
    SineSkewedWrappedNormalDistribution,
)


def _to_numpy(values):
    return np.asarray(pyrecest.backend.to_numpy(values))


def _assert_pdf_pair(values):
    values = _to_numpy(values)
    assert values.shape == (2,)
    assert np.all(values >= 0.0)


def test_generalized_k_sine_skewed_von_mises_pdf_accepts_list_input():
    dist = GeneralizedKSineSkewedVonMisesDistribution(
        mu=pi, kappa=1.0, lambda_=0.5, k=1, m=1
    )

    _assert_pdf_pair(dist.pdf([0.0, float(pi / 2)]))


def test_sine_skewed_wrapped_normal_pdf_accepts_list_input():
    dist = SineSkewedWrappedNormalDistribution(mu=0.0, sigma=1.0, lambda_=0.5)

    _assert_pdf_pair(dist.pdf([0.0, float(pi / 2)]))


def test_generalized_k_sine_skewed_wrapped_cauchy_pdf_accepts_list_input():
    dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
        mu=pi, gamma=0.5, lambda_=0.5, k=1, m=1
    )

    _assert_pdf_pair(dist.pdf([0.0, float(pi / 2)]))
