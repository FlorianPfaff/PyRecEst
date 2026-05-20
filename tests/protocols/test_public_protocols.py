from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.protocols import (
    SupportsFilterState,
    SupportsLinearPrediction,
    SupportsLinearUpdate,
    SupportsLogPdf,
    SupportsMeanAndCovariance,
    SupportsPdf,
    SupportsPointEstimate,
    SupportsSampling,
)


def test_gaussian_distribution_satisfies_distribution_protocols():
    distribution = GaussianDistribution(array([0.0]), array([[1.0]]))

    assert isinstance(distribution, SupportsPdf)
    assert isinstance(distribution, SupportsLogPdf)
    assert isinstance(distribution, SupportsSampling)
    assert isinstance(distribution, SupportsMeanAndCovariance)


def test_kalman_filter_satisfies_filter_protocols():
    filter_ = KalmanFilter((array([0.0]), array([[1.0]])))

    assert isinstance(filter_, SupportsFilterState)
    assert isinstance(filter_, SupportsPointEstimate)
    assert isinstance(filter_, SupportsLinearPrediction)
    assert isinstance(filter_, SupportsLinearUpdate)
