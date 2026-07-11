import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, pi, sin
from pyrecest.distributions.circle.sine_skewed_distributions import (
    GeneralizedKSineSkewedWrappedCauchyDistribution,
)


def test_pdf_remains_finite_for_large_gamma():
    mu = 0.3
    skewness = 0.4
    xs = array([0.0, 0.3, 1.0, pi, 2.0 * pi - 1e-6])
    dist = GeneralizedKSineSkewedWrappedCauchyDistribution(
        mu=mu,
        gamma=1000.0,
        lambda_=skewness,
        k=1,
        m=1,
    )

    pdf_values = dist.pdf(xs)
    expected = (1.0 + skewness * sin(xs - mu)) / (2.0 * pi)

    assert np.all(np.isfinite(np.asarray(pdf_values)))
    npt.assert_allclose(pdf_values, expected, rtol=1e-12, atol=0.0)
