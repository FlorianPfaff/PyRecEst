import numpy as np
from .abstract_hyperspherical_filter import AbstractHypersphericalFilter

class SphericalHarmonicsFilter(AbstractHypersphericalFilter):
    def __init__(self, degree, transformation='identity'):
        coeff_mat = np.zeros((degree + 1, 2 * degree + 1))
        if transformation == 'identity':
            coeff_mat[0, 0] = 1 / np.sqrt(4 * np.pi)
        elif transformation == 'sqrt':
            coeff_mat[0, 0] = 1
        # TODO