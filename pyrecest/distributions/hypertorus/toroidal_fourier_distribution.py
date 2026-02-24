from .hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from .abstract_toroidal_distribution import AbstractToroidalDistribution

class ToroidalFourierDistribution(HypertoroidalFourierDistribution, AbstractToroidalDistribution):
    @staticmethod
    def from_function(fun, n_coefficients, dim=2, desired_transformation='sqrt'):
        # Dim only used for compatibility with HypertoroidalFourierDistribution
        assert dim == 2
        return HypertoroidalFourierDistribution.from_function(fun, n_coefficients, dim=2, desired_transformation=desired_transformation)