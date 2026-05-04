"""Demonstrate representation conversion for common PyRecEst distributions.

This example exercises the generic representation-conversion gateway on two
small workflows:

1. approximate a Gaussian distribution by weighted Dirac particles and moment
   match those particles back to a Gaussian;
2. approximate a circular von Mises distribution by grid and Fourier
   representations.

Run from the repository root with:

``python examples/basic/representation_conversion.py``
"""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, sum
from pyrecest.distributions import GaussianDistribution, VonMisesDistribution
from pyrecest.distributions.conversion import convert_distribution


def _format_vector(vector):
    """Format a one-dimensional backend vector for concise terminal output."""
    return "[" + ", ".join(f"{float(value): .6f}" for value in vector) + "]"


def _format_matrix(matrix):
    """Format a small backend matrix for concise terminal output."""
    return "[" + ", ".join(_format_vector(row) for row in matrix) + "]"


def run_linear_example(n_particles=500):
    """Convert a Gaussian distribution to particles and back to a Gaussian.

    The particle approximation is stochastic, so the moment-matched Gaussian is
    not expected to equal the original distribution exactly. The example prints
    the resulting moments and the conversion methods reported by the gateway.
    """
    prior = GaussianDistribution(
        array([1.0, -0.5]),
        diag(array([0.25, 1.0])),
    )

    particle_result = convert_distribution(
        prior,
        "particles",
        n_particles=n_particles,
        return_info=True,
    )
    particles = particle_result.distribution

    gaussian_result = convert_distribution(
        particles,
        "gaussian",
        return_info=True,
    )
    moment_matched = gaussian_result.distribution

    return prior, particle_result, gaussian_result, moment_matched


def run_circular_example(no_of_gridpoints=64, n_fourier=64):
    """Convert a von Mises distribution to grid and Fourier representations."""
    distribution = VonMisesDistribution(array(1.25), 4.0)

    grid_result = convert_distribution(
        distribution,
        "grid",
        no_of_gridpoints=no_of_gridpoints,
        return_info=True,
    )
    fourier_result = convert_distribution(
        distribution,
        "fourier",
        n=n_fourier,
        return_info=True,
    )

    evaluation_points = array([0.0, 1.25, 3.141592653589793])
    return distribution, grid_result, fourier_result, evaluation_points


def print_linear_example():
    """Print the Gaussian/particle conversion workflow."""
    prior, particle_result, gaussian_result, moment_matched = run_linear_example()
    particles = particle_result.distribution

    print("Gaussian -> particles -> Gaussian")
    print("\nsource Gaussian:")
    print(f"  mean       = {_format_vector(prior.mu)}")
    print(f"  covariance = {_format_matrix(prior.C)}")

    print("\nparticle approximation:")
    print(f"  method       = {particle_result.method}")
    print(f"  exact        = {particle_result.exact}")
    print(f"  shape        = {particles.d.shape}")
    print(f"  weight sum   = {float(sum(particles.w)):.6f}")

    print("\nmoment-matched Gaussian from particles:")
    print(f"  method     = {gaussian_result.method}")
    print(f"  exact      = {gaussian_result.exact}")
    print(f"  mean       = {_format_vector(moment_matched.mu)}")
    print(f"  covariance = {_format_matrix(moment_matched.C)}")


def print_circular_example():
    """Print the circular grid/Fourier conversion workflow."""
    distribution, grid_result, fourier_result, evaluation_points = (
        run_circular_example()
    )
    grid = grid_result.distribution
    fourier = fourier_result.distribution

    print("\nvon Mises -> grid / Fourier")
    print("\nsource von Mises distribution:")
    print(f"  mu    = {float(distribution.mu):.6f}")
    print(f"  kappa = {float(distribution.kappa):.6f}")

    print("\ngrid approximation:")
    print(f"  method      = {grid_result.method}")
    print(f"  exact       = {grid_result.exact}")
    print(f"  grid points = {grid.grid_values.shape[0]}")

    print("\nFourier approximation:")
    print(f"  method       = {fourier_result.method}")
    print(f"  exact        = {fourier_result.exact}")
    print(f"  coefficients = {fourier.get_c().shape[0]}")

    print("\npointwise density comparison:")
    print("      x      original        grid      Fourier")
    for x, original, grid_value, fourier_value in zip(
        evaluation_points,
        distribution.pdf(evaluation_points),
        grid.pdf(evaluation_points),
        fourier.pdf(evaluation_points),
    ):
        print(
            f"  {float(x): .3f}  {float(original):10.6f}"
            f"  {float(grid_value):10.6f}  {float(fourier_value):10.6f}"
        )


def main():
    """Run both representation-conversion examples."""
    print_linear_example()
    print_circular_example()


if __name__ == "__main__":
    main()
