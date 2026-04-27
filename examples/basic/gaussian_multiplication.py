"""Example for multiplying multiple Gaussian distributions.

The product of normalized Gaussian densities is proportional to another
Gaussian. This script creates three two-dimensional Gaussian factors,
multiplies them with :meth:`GaussianDistribution.multiply`, and verifies the
result against the closed-form information representation.
"""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, diag, linalg, zeros
from pyrecest.distributions import GaussianDistribution


def multiply_all(distributions):
    """Multiply all Gaussian distributions in sequence."""
    product = distributions[0]
    for distribution in distributions[1:]:
        product = product.multiply(distribution)
    return product


def multiply_all_information_form(distributions):
    """Compute the normalized Gaussian product in information form."""
    precision_sum = zeros(distributions[0].C.shape)
    weighted_mean_sum = zeros(distributions[0].mu.shape)

    for distribution in distributions:
        precision = linalg.inv(distribution.C)
        precision_sum = precision_sum + precision
        weighted_mean_sum = weighted_mean_sum + precision @ distribution.mu

    covariance = linalg.inv(precision_sum)
    mean = covariance @ weighted_mean_sum
    return GaussianDistribution(mean, covariance, check_validity=False)


def create_gaussian_factors():
    """Create Gaussian factors to be multiplied."""
    return [
        GaussianDistribution(array([0.0, 1.0]), diag(array([4.0, 1.0]))),
        GaussianDistribution(array([1.0, -0.5]), diag(array([1.0, 2.25]))),
        GaussianDistribution(array([-0.75, 0.25]), diag(array([0.5, 0.75]))),
    ]


def run_example():
    """Run the Gaussian multiplication example."""
    distributions = create_gaussian_factors()
    product = multiply_all(distributions)
    reference_product = multiply_all_information_form(distributions)
    return distributions, product, reference_product


def main():
    """Print the multiplied Gaussian and a closed-form consistency check."""
    distributions, product, reference_product = run_example()

    print("Gaussian factors:")
    for index, distribution in enumerate(distributions, start=1):
        print(f"  factor {index}")
        print(f"    mean = {distribution.mu}")
        print(f"    covariance =\n{distribution.C}")

    print("\nProduct computed with GaussianDistribution.multiply():")
    print(f"  mean = {product.mu}")
    print(f"  covariance =\n{product.C}")

    print("\nClosed-form information result:")
    print(f"  mean = {reference_product.mu}")
    print(f"  covariance =\n{reference_product.C}")

    print("\nMatches closed form:")
    print(
        bool(
            allclose(product.mu, reference_product.mu)
            and allclose(product.C, reference_product.C)
        )
    )


if __name__ == "__main__":
    main()
