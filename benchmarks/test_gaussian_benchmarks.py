import pytest
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution

pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark
def test_gaussian_pdf_vector_benchmark(benchmark):
    distribution = GaussianDistribution(array([0.0, 0.0]), array([[1.0, 0.0], [0.0, 1.0]]))
    points = array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5], [2.0, -1.0]])

    benchmark(lambda: distribution.pdf(points))
