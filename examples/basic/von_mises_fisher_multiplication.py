"""Multiply two von Mises-Fisher distributions on the unit sphere."""

# pylint: disable=no-name-in-module,no-member

from pyrecest.backend import allclose, array, linalg
from pyrecest.distributions import VonMisesFisherDistribution


def _normalize(vector):
    """Return a unit-norm copy of ``vector``."""
    return vector / linalg.norm(vector)


def _format_vector(vector):
    """Format backend vectors for concise terminal output."""
    return "[" + ", ".join(f"{float(value): .6f}" for value in vector) + "]"


def _expected_product_parameters(first, second):
    """Compute the analytic vMF product parameters.

    A vMF density has the form ``C(kappa) * exp(kappa * mu.T @ x)``.
    Multiplying two vMF densities therefore adds their natural parameters
    ``kappa * mu``.  The normalized product is again vMF with
    ``kappa = ||kappa_1 * mu_1 + kappa_2 * mu_2||``.
    """
    natural_parameter = first.kappa * first.mu + second.kappa * second.mu
    kappa = linalg.norm(natural_parameter)
    mu = natural_parameter / kappa
    return mu, kappa


def run_example():
    """Multiply two vMF distributions and verify the product relation."""
    first = VonMisesFisherDistribution(_normalize(array([1.0, 0.2, 0.1])), 8.0)
    second = VonMisesFisherDistribution(_normalize(array([0.2, 1.0, 0.3])), 5.0)

    product = first.multiply(second)
    expected_mu, expected_kappa = _expected_product_parameters(first, second)

    xs_unnormalized = array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 0.5, 0.25],
        ]
    )
    xs = xs_unnormalized / linalg.norm(xs_unnormalized, axis=1).reshape(-1, 1)

    # The product density is proportional to the pointwise product of the
    # original densities.  This ratio is constant over the hypersphere.
    normalization_ratio = product.C / (first.C * second.C)
    scaled_pointwise_product = normalization_ratio * first.pdf(xs) * second.pdf(xs)

    assert allclose(product.mu, expected_mu, atol=1e-12)
    assert allclose(product.kappa, expected_kappa, atol=1e-12)
    assert allclose(product.pdf(xs), scaled_pointwise_product, atol=1e-12)

    return first, second, product, normalization_ratio, xs, scaled_pointwise_product


def main():
    first, second, product, normalization_ratio, xs, scaled_product = run_example()

    print("Multiplying von Mises-Fisher distributions on S^2")
    print("\nfirst distribution:")
    print(f"  mu    = {_format_vector(first.mu)}")
    print(f"  kappa = {float(first.kappa):.6f}")
    print("\nsecond distribution:")
    print(f"  mu    = {_format_vector(second.mu)}")
    print(f"  kappa = {float(second.kappa):.6f}")
    print("\nnormalized product distribution:")
    print(f"  mu    = {_format_vector(product.mu)}")
    print(f"  kappa = {float(product.kappa):.6f}")
    print(
        f"\nconstant ratio product.pdf(x) / (first.pdf(x) * second.pdf(x)) = {float(normalization_ratio):.6f}"
    )

    print("\npointwise check:")
    print("index  product.pdf(x)  ratio * first.pdf(x) * second.pdf(x)")
    for index, (x, scaled_value) in enumerate(zip(xs, scaled_product), start=1):
        print(
            f"{index:>5}  {float(product.pdf(x)):>14.8f}"
            f"  {float(scaled_value):>37.8f}"
        )


if __name__ == "__main__":
    main()
