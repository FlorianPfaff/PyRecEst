from pyrecest.backend import pi, zeros
from scipy.integrate import quad
from .abstract_circular_distribution import AbstractCircularDistribution
from beartype import beartype
from typing import Callable

class PieceWiseConstantDistribution(AbstractCircularDistribution):
    def __init__(self, weights):
        self.weights = weights
        self.n = len(weights)
        normalization_constant = 1 / (2 * pi * sum(weights) / self.n)
        self.weights = normalization_constant * weights
    @staticmethod
    @beartype
    def calculate_parameters_numerically(pdf: Callable, n):
        assert n >= 1
        w = zeros(n)
        for j in range(n):
            left = PieceWiseConstantDistribution.left_border(j, n)
            right = PieceWiseConstantDistribution.right_border(j, n)
            w[j] = quad(pdf, left, right)[0]
        return w

    @staticmethod
    def left_border(m, n):
        assert 1 <= m <= n
        return 2 * pi / n * (m - 1)

    @staticmethod
    def right_border(m, n):
        assert 1 <= m <= n
        return 2 * pi / n * m

    @staticmethod
    def interval_center(m, n):
        assert 1 <= m <= n
        return 2 * pi / n * (m - 0.5)