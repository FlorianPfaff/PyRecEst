import numpy as np
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


def test_zero_order_pdf_centers_truncation_on_wrapped_residual():
    sigma = 0.2
    mean = 5.5
    distribution = HypertoroidalWrappedNormalDistribution(
        array([mean]),
        array([[sigma**2]]),
    )

    values = distribution.pdf(
        array([[mean], [mean - 2.0 * np.pi]]),
        m=0,
    )

    expected_peak = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    npt.assert_allclose(
        to_numpy(values),
        [expected_peak, expected_peak],
        rtol=1.0e-6,
        atol=1.0e-6,
    )
