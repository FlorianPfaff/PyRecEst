import numpy as np
import pytest
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


@pytest.mark.parametrize(
    "points",
    [
        JulierSigmaPoints(n=2, kappa=1.0),
        MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
    ],
    ids=["julier", "merwe"],
)
def test_sigma_points_reject_asymmetric_covariance(points):
    covariance = np.array([[1.0, 9.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="P must be symmetric"):
        points.sigma_points(np.zeros(2), covariance)


@pytest.mark.parametrize(
    "points",
    [
        JulierSigmaPoints(n=2, kappa=1.0),
        MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
    ],
    ids=["julier", "merwe"],
)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf], ids=["nan", "infinity"])
def test_sigma_points_reject_nonfinite_covariance(points, invalid_value):
    covariance = np.diag([1.0, invalid_value])

    with pytest.raises(ValueError, match="P must contain only finite values"):
        points.sigma_points(np.zeros(2), covariance)


@pytest.mark.parametrize(
    "points",
    [
        JulierSigmaPoints(n=2, kappa=1.0),
        MerweScaledSigmaPoints(n=2, alpha=0.5, beta=2.0, kappa=0.0),
    ],
    ids=["julier", "merwe"],
)
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf], ids=["nan", "infinity"])
def test_sigma_points_reject_nonfinite_mean(points, invalid_value):
    mean = np.array([0.0, invalid_value])

    with pytest.raises(ValueError, match="x must contain only finite values"):
        points.sigma_points(mean, np.eye(2))
