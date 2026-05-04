import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import cov, ones, reshape
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import zeros

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_linear_distribution import AbstractLinearDistribution


class LinearDiracDistribution(AbstractDiracDistribution, AbstractLinearDistribution):
    def __init__(self, d, w=None):
        dim = d.shape[1] if d.ndim > 1 else 1
        AbstractLinearDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, d, w)

    def mean(self):
        # Like np.average(self.d, weights=self.w, axis=0) but for all backends
        return self.w @ self.d

    def set_mean(self, new_mean):
        mean_offset = new_mean - self.mean()
        if self.d.ndim == 1:
            self.d += mean_offset
        else:
            self.d += reshape(mean_offset, (1, -1))

    def covariance(self):
        _, C = LinearDiracDistribution.weighted_samples_to_mean_and_cov(self.d, self.w)
        return C

    def plot(self, *args, **kwargs):
        if pyrecest.backend.__backend_name__ == "numpy":
            sample_locs = self.d
            sample_weights = self.w
        elif pyrecest.backend.__backend_name__ == "pytorch":
            sample_locs = self.d.numpy()
            sample_weights = self.w.numpy()
        else:
            raise ValueError("Plotting not supported for this backend")

        if self.dim == 1:
            plt.stem(sample_locs.squeeze(), sample_weights, *args, **kwargs)
        elif self.dim == 2:
            plt.scatter(
                sample_locs[:, 0],
                sample_locs[:, 1],
                sample_weights / max(sample_weights) * 100,
                *args,
                **kwargs
            )
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            # You can adjust 's' for marker size as needed
            ax.scatter(
                sample_locs[:, 0],
                sample_locs[:, 1],
                sample_locs[:, 2],
                s=(sample_weights / max(sample_weights) * 100),
                *args,
                **kwargs
            )
        else:
            raise ValueError("Plotting not supported for this dimension")

        plt.show()

    @staticmethod
    def from_distribution(distribution, n_particles=None, n_samples=None, n=None):
        particle_count = LinearDiracDistribution._resolve_particle_count(
            n_particles=n_particles,
            n_samples=n_samples,
            n=n,
        )
        samples = distribution.sample(particle_count)
        return LinearDiracDistribution(samples, ones(particle_count) / particle_count)

    @staticmethod
    def _resolve_particle_count(n_particles=None, n_samples=None, n=None):
        from ..conversion import ConversionError

        specified_counts = [
            value for value in (n_particles, n_samples, n) if value is not None
        ]
        if not specified_counts:
            raise ConversionError(
                "LinearDiracDistribution.from_distribution requires "
                "n_particles, n_samples, or n."
            )

        particle_counts = [int(value) for value in specified_counts]
        if len(set(particle_counts)) != 1:
            raise ConversionError(
                "n_particles, n_samples, and n must agree when more than one "
                "particle-count alias is supplied."
            )

        particle_count = particle_counts[0]
        if particle_count <= 0:
            raise ConversionError("Number of particles must be positive.")
        return particle_count

    @staticmethod
    def weighted_samples_to_mean_and_cov(samples, weights=None):
        sample_matrix = reshape(samples, (-1, 1)) if samples.ndim == 1 else samples

        if weights is None:
            weights = ones(sample_matrix.shape[0]) / sample_matrix.shape[0]
        else:
            weights = weights / backend_sum(weights)

        mean = weights @ sample_matrix
        deviation = sample_matrix - mean
        if sample_matrix.shape[0] == 1:
            covariance = zeros((sample_matrix.shape[1], sample_matrix.shape[1]))
        else:
            covariance = cov(deviation.T, aweights=weights, bias=True)
            if sample_matrix.shape[1] == 1:
                covariance = reshape(covariance, (1, 1))

        return mean, covariance
