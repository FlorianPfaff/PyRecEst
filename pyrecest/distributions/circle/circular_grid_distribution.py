import numpy as np
from scipy.special import sinc
from .circular_fourier_distribution import CircularFourierDistribution
from .abstract_circular_distribution import AbstractCircularDistribution
from ..hypertorus.hypertoroidal_grid_distribution import HypertoroidalGridDistribution

class CircularGridDistribution(HypertoroidalGridDistribution, AbstractCircularDistribution):
    def __init__(self, grid_values, enforce_pdf_nonnegative=True):
        if isinstance(grid_values, AbstractCircularDistribution):
            raise ValueError("You gave a distribution as the first argument. To convert distributions to a distribution in grid representation, use .from_distribution")
        AbstractCircularDistribution.__init__(self)
        HypertoroidalGridDistribution.__init__(self, grid_values, enforce_pdf_nonnegative=enforce_pdf_nonnegative)

    def pdf(self, xs, use_sinc=False, sinc_repetitions=5):
        if use_sinc:
            assert sinc_repetitions % 2 == 1
            step_size = 2 * np.pi / len(self.grid_values)
            _range = np.arange(-np.floor(sinc_repetitions / 2) * len(self.grid_values), np.ceil(sinc_repetitions / 2) * len(self.grid_values))
            sinc_vals = sinc((xs / step_size).reshape(-1, 1) - _range)
            if self.enforce_pdf_nonnegative:
                p = np.sum(np.sqrt(self.grid_values).reshape(-1, 1) * sinc_vals, axis=1) ** 2
            else:
                p = np.sum(self.grid_values.reshape(-1, 1) * sinc_vals, axis=1)
        else:
            no_coeffs = len(self.grid_values)
            if len(self.grid_values) % 2 == 0:
                no_coeffs += 1
            if self.enforce_pdf_nonnegative:
                fd = CircularFourierDistribution.from_function_values(np.sqrt(self.grid_values), no_coeffs)
            else:
                fd = CircularFourierDistribution.from_function_values(self.grid_values, no_coeffs)
            p = fd.pdf(xs)
            
        return p

    def pdf_on_grid(self, no_of_desired_gridpoints):
        x_grid = np.linspace(0, 2 * np.pi, len(self.grid_values), endpoint=False)
        step = len(self.grid_values) // no_of_desired_gridpoints
        assert step == int(step), "Number of function values has to be a multiple of no_of_gridpoints"

        vals = self.grid_values[::step]
        return vals, x_grid

    def trigonometric_moment(self, n):
        no_coeffs = len(self.grid_values)
        if len(self.grid_values) % 2 == 0:
            no_coeffs += 1
        fd = CircularFourierDistribution.from_function_values(self.grid_values, no_coeffs)
        return fd.trigonometric_moment(n)