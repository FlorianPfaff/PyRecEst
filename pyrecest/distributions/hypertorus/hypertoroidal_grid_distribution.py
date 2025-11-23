import numpy as np
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from ..abstract_grid_distribution import AbstractGridDistribution
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution

class HypertoroidalGridDistribution(AbstractGridDistribution, AbstractHypertoroidalDistribution):
    def __init__(self, grid_values, grid_type = "custom", grid = None, enforce_pdf_nonnegative=True, dim=None):
        # Constructor
        if dim is None and grid is None or grid.ndim<=1:
            dim = 1
        elif dim is None:
            dim = grid.shape[1]

        AbstractHypertoroidalDistribution.__init__(self, dim)
        AbstractGridDistribution.__init__(self, grid_values, grid_type, grid, dim)
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # Check if normalized. If not: Normalize
        self.normalize_in_place()
        
    def get_closest_point(self, xs):
        min_error = np.inf # Initialize with infinity
        closest_point = None
        
        for point in self.grid:
            error = AbstractHypertoroidalDistribution.angular_error(x, point)
            if error < min_error:
                min_error = error
                closest_point = point
        
        return closest_point

    def get_manifold_size(self):
        return AbstractHypertoroidalDistribution.get_manifold_size(self)

    @staticmethod
    def generate_cartesian_product_grid(n_grid_points):
        grid_individual_axis = [np.linspace(0, 2 * np.pi - 2 * np.pi / n, n) for n in n_grid_points]
        meshgrids = np.meshgrid(*grid_individual_axis, indexing='ij')
        grid = np.column_stack([grid.ravel() for grid in meshgrids])
        return grid

    def multiply(self, other):
        assert np.all(self.grid == other.grid), 'Multiply:IncompatibleGrid: Can only multiply for equal grids.'
        return super().multiply(other)

    def pdf(self, xs):
        assert self.grid_type == 'CartesianProd', 'pdf is not defined. Can only interpolate for certain grid types.'
        print('Warning: PDF:UseInterpolated: pdf is not defined. Using interpolation with Fourier series.')

        transformation = 'sqrt' if self.enforce_pdf_nonnegative else 'identity'

        print('Warning: Cannot interpolate if number of grid points are not specified. Assuming equidistant grid')
        raise NotImplementedError('Pdf currently not implemented')
        fd = HypertoroidalFourierDistribution.from_function_values(
            np.reshape(self.grid_values, self.grid_density_description["n_grid_values"]),
            self.grid_density_description["n_grid_values"] + (self.grid_density_description["n_grid_values"] % 2 == 0),
            transformation
        )
        return fd.pdf(xs)

    def get_grid(self):
        if self.grid.size > 0:
            return self.grid
        elif self.grid_type == 'CartesianProd':
            print('Warning: Grid:GenerateDuringRunTime: Generating grid anew on call to .getGrid(). If you require the grid frequently, store it in the class.')
            return np.squeeze(self.generate_cartesian_product_grid(self.n_grid_points))
        else:
            raise ValueError('Grid:UnknownGrid: Grid was not provided and is thus unavailable')

    def pdf_unnormalized(self, xs):
        assert self.grid_type == 'CartesianProd', 'pdf is not defined. Can only interpolate for certain grid types.'
        p = self.integrate() * self.pdf(xs)
        return p
    
    def shift(self, shift_by):
        if np.linalg.norm(shift_by) == 0:
            return self

        assert self.grid_type == 'CartesianProd'

        # Initialize with some uniform distribution and then replace coefficient matrix
        coeff_mat_tmp = np.zeros(tuple([3] * self.dim))
        coeff_mat_tmp[0] = 1 / (2 * np.pi) ** self.dim
        raise NotImplementedError('Shift currently not implemented')
        hfd = HypertoroidalFourierDistribution(fftshift(coeff_mat_tmp), 'identity')
        hfd.C = fftshift(fftn(np.reshape(self.grid_values, self.n_grid_points)))
        hfd_shifted = hfd.shift(shift_by)

        shifted_distribution = self.__class__(self.grid, self.grid_values)
        shifted_distribution.grid_values = np.reshape(ifftn(ifftshift(hfd_shifted.C), overwrite_x=True), self.grid_values.shape)
        return shifted_distribution
    
    def value_of_closest(self, xa):
        p = np.empty(xa.shape[1])
        for i in range(xa.shape[1]):
            dists = np.sum(np.minimum((self.grid - xa[:, i]) ** 2, (2 * np.pi - (self.grid - xa[:, i])) ** 2), axis=0)
            min_ind = np.argmin(dists)
            p[i] = self.grid_values[min_ind]
        return p

    def convolve(self, other):
        assert self.grid_type == 'CartesianProd' and other.grid_type == 'CartesianProd'
        assert np.all(self.n_grid_points == other.no_of_grid_points)
        assert np.all(self.grid == other.grid)

        this_tensor = np.reshape(self.grid_values, self.n_grid_points)
        other_tensor = np.reshape(other.grid_values, other.no_of_grid_points)

        res_tensor = (2 * np.pi) ** self.dim / self.grid_values.size * ifftn(fftn(this_tensor) * fftn(other_tensor))
        convolved_distribution = self.__class__(self.grid, np.reshape(res_tensor, self.grid_values.shape))
        return convolved_distribution

    @staticmethod
    def from_distribution(distribution, n_grid_points, grid_type='CartesianProd', enforce_pdf_nonnegative = True):
        assert isinstance(grid_type, str)
        """
        if grid_type == 'CartesianProd' and isinstance(distribution, HypertoroidalFourierDistribution) and \
                (distribution.C.shape == n_grid_points or (np.isscalar(n_grid_points) and n_grid_points == distribution.C.shape[0])):
            grid_values = distribution.pdf_on_grid()
            grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(n_grid_points)
            hgd = HypertoroidalGridDistribution(grid, grid_values.flatten())
            hgd.grid_type = grid_type"
        else:
        """
        
        hgd = HypertoroidalGridDistribution.from_function(distribution.pdf, n_grid_points, distribution.dim, grid_type=grid_type, enforce_pdf_nonnegative=enforce_pdf_nonnegative)
        return hgd

    @staticmethod
    def from_function(fun, n_gridpoints, dim, grid_type='CartesianProd', enforce_pdf_nonnegative = True):
        assert isinstance(grid_type, str)
        if grid_type == 'CartesianProd':
            if np.isscalar(n_gridpoints):
                n_gridpoints = np.repeat(n_gridpoints, dim)
            else:
                assert len(n_gridpoints) == dim

            grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(n_gridpoints)
        else:
            raise ValueError("Grid scheme not recognized")

        grid_values = np.apply_along_axis(fun, 1, grid)
        sgd = HypertoroidalGridDistribution(grid_values, grid=grid, enforce_pdf_nonnegative=enforce_pdf_nonnegative, grid_type=grid_type)
        return sgd

    def plot(self, *args, **kwargs):
        hdd = HypertoroidalDiracDistribution(self.grid, (1 / len(self.grid_values)) * np.ones(len(self.grid_values)))
        hdd.w = self.grid_values.T
        h = hdd.plot(*args, **kwargs)
        return h

    def plot_interpolated(self, *args, **kwargs):
        if self.dim > 2:
            raise ValueError("Can currently only plot for T1 and T2 torus.")
        if self.enforce_pdf_nonnegative:
            transformation = 'sqrt'
        else:
            transformation = 'identity'
        sizes = np.repeat(len(self.grid_values) ** (1 / self.dim), self.dim)
        raise NotImplementedError('Plot currently not implemented')
        fd = HypertoroidalFourierDistribution.from_function_values(np.reshape(self.grid_values, [sizes, 1]), sizes, transformation)
        h = fd.plot(*args, **kwargs)
        return h

    def trigonometric_moment(self, n):
        hwd = HypertoroidalDiracDistribution(self.grid, self.grid_values.T / sum(self.grid_values))
        m = hwd.trigonometric_moment(n)
        return m

    def slice_at(self, dims, val, use_fftn=True):
        raise NotImplementedError("Not yet supported.")
        assert self.grid_type == 'CartesianProd', "This operation is only supported for grids generated based on a Cartesian product."
        assert all([dim <= self.dim for dim in dims]), "Cannot perform this operation for a dimension that is higher than the dimensionality of the distribution."

        fvals_on_grid = np.reshape(self.grid_values, self.n_grid_points)
        if np.all(val == np.zeros(len(val))):
            grid_shifted_full = fvals_on_grid
        elif use_fftn:
            # Use HypertoroidalFourierDistribution, which uses FFTN
            shift_vec = np.zeros(self.dim)
            shift_vec[dims] = -val
            hfd_sqrt = HypertoroidalFourierDistribution.from_function_values(fvals_on_grid, self.n_grid_points, 'sqrt')
            hfd_sqrt_shifted = hfd_sqrt.shift(shift_vec)
            grid_shifted_full = hfd_sqrt_shifted.pdf_on_grid()
        else:

            raise NotImplementedError("This part of the code requires further adaptation to work in Python.")
        return grid_shifted_full