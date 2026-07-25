import numpy.testing as npt
from pyrecest.backend import array, stack, to_numpy
from pyrecest.distributions import GaussianMixture


def test_gaussian_mixture_moment_match_is_invariant_to_weight_scale():
    means = array([[0.0], [2.0]])
    covariance_matrices = stack(
        [
            array([[1.0]]),
            array([[1.0]]),
        ],
        axis=2,
    )

    normalized_mean, normalized_covariance = (
        GaussianMixture.mixture_parameters_to_gaussian_parameters(
            means,
            covariance_matrices,
            [0.25, 0.75],
        )
    )
    scaled_mean, scaled_covariance = (
        GaussianMixture.mixture_parameters_to_gaussian_parameters(
            means,
            covariance_matrices,
            [1.0, 3.0],
        )
    )

    npt.assert_allclose(to_numpy(normalized_mean), [1.5])
    npt.assert_allclose(to_numpy(normalized_covariance), [[1.75]])
    npt.assert_allclose(to_numpy(scaled_mean), to_numpy(normalized_mean))
    npt.assert_allclose(to_numpy(scaled_covariance), to_numpy(normalized_covariance))
