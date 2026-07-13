from numbers import Integral

import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, cov, ones, reshape, to_numpy, zeros

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_linear_distribution import AbstractLinearDistribution


class LinearDiracDistribution(AbstractDiracDistribution, AbstractLinearDistribution):
    def __init__(self, d, w=None):
        d = asarray(d)
        if d.ndim == 0:
            d = reshape(d, (1,))
        dim = d.shape[1] if d.ndim > 1 else 1
        AbstractLinearDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, d, w)

    def mean(self):
        # Like np.average(self.d, weights=self.w, axis=0) but for all backends
        return self.w @ self.d

    def set_mean(self, new_mean):
        new_mean = asarray(new_mean)
        if new_mean.ndim == 0:
            if self.dim != 1:
                raise ValueError(
                    f"new_mean must have shape ({self.dim},), got {new_mean.shape}."
                )
            new_mean = reshape(new_mean, (1,))
        elif new_mean.ndim != 1 or new_mean.shape[0] != self.dim:
            raise ValueError(
                f"new_mean must have shape ({self.dim},), got {new_mean.shape}."
            )

        mean_offset = new_mean - self.mean()
        if self.d.ndim == 1:
            self.d += mean_offset
        else:
            self.d += reshape(mean_offset, (1, -1))

    def covariance(self):
        _, C = LinearDiracDistribution.weighted_samples_to_mean_and_cov(self.d, self.w)
        return C

    def plot(self, *args, **kwargs):
        if pyrecest.backend.__backend_name__ in {"numpy", "pytorch"}:
            sample_locs = to_numpy(self.d)
            sample_weights = to_numpy(self.w)
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

        particle_counts = [
            LinearDiracDistribution._validate_particle_count(value)
            for value in specified_counts
        ]
        if len(set(particle_counts)) != 1:
            raise ConversionError(
                "n_particles, n_samples, and n must agree when more than one "
                "particle-count alias is supplied."
            )

        return particle_counts[0]

    @staticmethod
    def _validate_particle_count(value):
        from ..conversion import ConversionError

        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
        ):
            raise ConversionError("Number of particles must be a positive integer.")
        return int(value)

    @staticmethod
    def weighted_samples_to_mean_and_cov(samples, weights=None):
        samples = asarray(samples)
        sample_matrix = reshape(samples, (-1, 1)) if samples.ndim <= 1 else samples
        if sample_matrix.ndim != 2:
            raise ValueError("samples must be a scalar, 1D array, or 2D array")
        if sample_matrix.shape[0] == 0:
            raise ValueError("samples must contain at least one sample")

        if weights is None:
            weights = ones(sample_matrix.shape[0]) / sample_matrix.shape[0]
        else:
            weights = reshape(asarray(weights), (-1,))
            if weights.shape[0] != sample_matrix.shape[0]:
                raise ValueError("Number of weights and samples must match")
            weight_scale, scaled_total_weight = (
                AbstractDiracDistribution._validate_weights(weights)
            )
            weights = (weights / weight_scale) / scaled_total_weight

        mean = weights @ sample_matrix
        deviation = sample_matrix - mean
        if sample_matrix.shape[0] == 1:
            covariance = zeros((sample_matrix.shape[1], sample_matrix.shape[1]))
        else:
            covariance = cov(deviation.T, aweights=weights, bias=True)
            if sample_matrix.shape[1] == 1:
                covariance = reshape(covariance, (1, 1))

        return mean, covariance
