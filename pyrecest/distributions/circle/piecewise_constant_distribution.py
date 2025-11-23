import numpy as np
from scipy.integrate import quad
from .abstract_circular_distribution import AbstractCircularDistribution
from beartype import beartype
from typing import Callable

class PieceWiseConstantDistribution(AbstractCircularDistribution):
    @staticmethod
    @beartype
    def calculate_parameters_numerically(pdf: Callable, n):
        assert n >= 1
        w = np.zeros(n)
        for j in range(n):
            left = PieceWiseConstantDistribution.left_border(j, n)
            right = PieceWiseConstantDistribution.right_border(j, n)
            w[j] = quad(pdf, left, right)[0]
        return w

    @staticmethod
    def left_border(m, n):
        assert 1 <= m <= n
        return 2 * np.pi / n * (m - 1)

    @staticmethod
    def right_border(m, n):
        assert 1 <= m <= n
        return 2 * np.pi / n * m

    @staticmethod
    def interval_center(m, n):
        assert 1 <= m <= n
        return 2 * np.pi / n * (m - 0.5)