import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.gaussian_mixture_phd_filter import GaussianMixturePHDFilter

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Currently only supported for the numpy backend",
)


@pytest.mark.parametrize(
    "measurement_covariance",
    [
        [[0.1, 0.0], [0.0, 0.1]],
        [[[0.1], [0.0]], [[0.0], [0.1]]],
    ],
    ids=["shared-covariance", "per-measurement-covariance"],
)
def test_update_linear_accepts_nested_list_model_inputs(measurement_covariance):
    tracker = GaussianMixturePHDFilter(
        initial_components=[GaussianDistribution(array([0.0, 0.0]), eye(2))],
        initial_weights=array([0.9]),
        detection_probability=0.95,
        clutter_intensity=1e-4,
        extraction_threshold=0.3,
        merging_threshold=0.01,
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )

    tracker.update_linear(
        [[0.2], [-0.1]],
        [[1.0, 0.0], [0.0, 1.0]],
        measurement_covariance,
    )

    npt.assert_allclose(
        tracker.get_point_estimate().reshape((-1,)),
        np.array([0.18181818, -0.09090909]),
        atol=5e-3,
    )
