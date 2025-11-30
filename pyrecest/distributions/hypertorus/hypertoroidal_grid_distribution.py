import warnings
from pyrecest.backend import pi, atleast_1d, fft, unique, prod, sum, abs, repeat, isclose, min, any, all, equal, meshgrid, stack, linspace, zeros, ones, argmin, real, imag, linalg

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from ..abstract_grid_distribution import AbstractGridDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class HypertoroidalGridDistribution(AbstractGridDistribution, AbstractHypertoroidalDistribution):
    """Grid-based distribution on a hypertorus.

    Python convention:
        * grid: array of shape (n_grid_points, dim)
        * grid_values: array of shape (n_grid_points,)
        * pdf(x): x has shape (n_eval, dim)

    """

    def __init__(
        self,
        grid_values,
        grid_type: str = "custom",
        grid = None,
        enforce_pdf_nonnegative: bool = True,
        dim: int | None = None,
        n_grid_points = None,
    ):
        # --- infer dimension -------------------------------------------------
        grid_array = None
        if grid is not None:
            grid_array = grid
            if grid_array.ndim != 2:
                raise ValueError("grid must be of shape (n_samples, dim)")
            n_samples, inferred_dim = grid_array.shape

            if dim is None:
                dim = inferred_dim
            elif dim != inferred_dim:
                raise ValueError(
                    f"Inconsistent dimension: dim={dim} but grid has dim={inferred_dim}"
                )
        else:
            if dim is None:
                # Fallback: 1D torus if no grid and no dim are given
                dim = 1

        # Initialize hypertoroidal base class
        AbstractHypertoroidalDistribution.__init__(self, dim)

        # Flatten grid_values to 1D
        grid_values = grid_values.reshape(-1)

        # Initialize grid base class
        AbstractGridDistribution.__init__(self, grid_values, grid_type, grid_array, dim)

        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative

        # --- number of grid points along each axis ---------------------------
        # For Cartesian product grids, try to infer per-dimension grid sizes.
        if grid_type == "cart_prod":
            if n_grid_points is not None:
                n_grid_points = atleast_1d(n_grid_points)
            elif grid_array is not None:
                # Infer from unique coordinates along each dimension
                per_dim = []
                for d in range(dim):
                    unique_vals = unique(grid_array[:, d])
                    per_dim.append(unique_vals.size)
                n_grid_points = per_dim
                if prod(n_grid_points) != grid_array.shape[0]:
                    raise ValueError(
                        "Grid does not look like a full Cartesian product: "
                        f"product(n_grid_points)={prod(n_grid_points)} "
                        f"but grid has {grid_array.shape[0]} rows."
                    )
            else:
                # No grid given; fall back to equal resolution in each dimension
                total_points = grid_values.size
                per_dim_float = total_points ** (1.0 / dim)
                if not isclose(per_dim_float, round(per_dim_float)):
                    raise ValueError(
                        "Cannot infer per-dimension grid size from the number of grid values."
                    )
                n_grid_points = repeat(int(round(per_dim_float)), dim)

            self.n_grid_points = n_grid_points
        else:
            self.n_grid_points = None

        # Check if normalized. If not: normalize in place.
        self.normalize_in_place()

    # ------------------------------------------------------------------ utils
    def get_closest_point(self, x):
        """Return the closest grid point (toroidal distance) to x.

        Parameters
        ----------
        x : array_like, shape (dim,) or (1, dim)
        """
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected point of dimension {self.dim}, got {x.shape[1]}")
        if self.grid is None or self.grid.size == 0:
            raise ValueError("Grid is empty; cannot find closest point.")

        # Toroidal squared distance: min(Δ^2, (2π - |Δ|)^2) summed over dimensions
        delta = self.grid[None, :, :] - x[:, None, :]  # (1, n_grid, dim)
        abs_delta = abs(delta)
        dists = sum(min((abs_delta ** 2, (2 * pi - abs_delta) ** 2)), axis=-1)
        min_index = int(argmin(dists[0]))
        return self.grid[min_index]

    def get_manifold_size(self):
        return AbstractHypertoroidalDistribution.get_manifold_size(self)

    # ---------------------------------------------------------- grid helpers
    @staticmethod
    def generate_cartesian_product_grid(n_grid_points):
        """Generate a Cartesian product grid on [0, 2π)^dim.

        Parameters
        ----------
        n_grid_points : int or sequence of int
            Number of grid points along each dimension.
        """
        n_grid_points = atleast_1d(n_grid_points)
        if any(n_grid_points <= 0):
            raise ValueError("n_grid_points must contain positive integers")

        axes = [
            linspace(0.0, 2.0 * pi - 2.0 * pi / n, n)
            for n in n_grid_points
        ]
        mesh = meshgrid(*axes, indexing="ij")
        grid = stack([m.ravel() for m in mesh], axis=-1)  # (n_samples, dim)
        return grid

    # ---------------------------------------------------------------- combine
    def multiply(self, other):
        assert np.all(self.grid == other.grid), (
            "Multiply:IncompatibleGrid: Can only multiply for equal grids."
        )
        return super().multiply(other)

    # --------------------------------------------------------------------- pdf
    def pdf(self, xs):
        """Interpolate the pdf using a Fourier-series representation.

        xs : array_like, shape (n_eval, dim) or (dim,) for a single point
        """
        if self.grid_type != "CartesianProd":
            raise ValueError(
                "pdf is only defined (via interpolation) for 'CartesianProd' grids."
            )

        if xs.ndim == 1:
            xs = xs.reshape(1, -1)
        if xs.shape[1] != self.dim:
            raise ValueError(
                f"Expected xs with shape (n_eval, {self.dim}), got {xs.shape}"
            )

        warnings.warn(
            "pdf is not available in closed form; using Fourier-series interpolation.",
            RuntimeWarning,
        )

        transformation = "sqrt" if self.enforce_pdf_nonnegative else "identity"

        if self.n_grid_points is None:
            if self.grid is None:
                raise ValueError(
                    "Cannot interpolate pdf: grid and n_grid_points are not specified."
                )
            warnings.warn(
                "Cannot interpolate if number of grid points is not specified. "
                "Inferring from the grid assuming a Cartesian product.",
                RuntimeWarning,
            )
            per_dim = []
            for d in range(self.dim):
                unique_vals = unique(self.grid[:, d])
                per_dim.append(unique_vals.size)
            self.n_grid_points = per_dim

        n_grid_points = self.n_grid_points
        if prod(n_grid_points) != self.grid_values.size:
            raise ValueError(
                "n_grid_points is inconsistent with number of grid values."
            )

        # reshape values to tensor with trailing singleton dim for 1D case
        values_tensor = self.grid_values.reshape(*n_grid_points, 1)

        sizes_for_fd = n_grid_points + (n_grid_points % 2 == 0).astype(int)
        """
        fd = HypertoroidalFourierDistribution.from_function_values(
            values_tensor, sizes_for_fd, transformation
        )
        return fd.pdf(xs)
        """
        raise NotImplementedError(
            "pdf interpolation is not yet implemented in the Python version."
        )

    def get_grid(self):
        if self.grid is not None and self.grid.size > 0:
            return self.grid
        if self.grid_type == "cart_prod":
            warnings.warn(
                "Grid:GenerateDuringRunTime: Generating grid anew on call to "
                "get_grid(). If you require the grid frequently, store it in the class.",
                RuntimeWarning,
            )
            if self.n_grid_points is None:
                raise ValueError(
                    "Cannot generate grid: n_grid_points is not defined."
                )
            return self.generate_cartesian_product_grid(self.n_grid_points)
        raise ValueError("Grid:UnknownGrid: Grid was not provided and is thus unavailable")

    def pdf_unnormalized(self, xs):
        if self.grid_type != "CartesianProd":
            raise ValueError(
                "pdf_unnormalized is only defined for 'CartesianProd' grids."
            )
        p = self.integrate() * self.pdf(xs)
        return p

    # ------------------------------------------------------------------- shift
    def shift(self, shift_by):
        """Return a shifted copy of the distribution.

        Parameters
        ----------
        shift_by : array_like, shape (dim,)
            Angular shift on the hypertorus.
        """
        shift_by = shift_by.reshape(-1)
        if shift_by.size != self.dim:
            raise ValueError(
                f"shift_by must have length {self.dim}, got length {shift_by.size}"
            )

        if linalg.norm(shift_by) == 0.0:
            return self

        if self.grid_type != "CartesianProd":
            raise ValueError("Shift is only implemented for 'CartesianProd' grids.")

        if self.n_grid_points is None:
            if self.grid is None:
                raise ValueError(
                    "Cannot shift distribution: grid / n_grid_points are not defined."
                )
            per_dim = []
            for d in range(self.dim):
                unique_vals = unique(self.grid[:, d])
                per_dim.append(unique_vals.size)
            self.n_grid_points = per_dim

        # Initialize a temporary uniform Fourier distribution and overwrite its
        # coefficient matrix with the FFT of the grid values.
        coeff_shape = (3,) * self.dim
        coeff_mat_tmp = zeros(coeff_shape, dtype=complex)
        coeff_mat_tmp[(0,) * self.dim] = 1.0 / (2.0 * pi) ** self.dim
        raise NotImplementedError(
            "Shift is not yet implemented in the Python version."
        )
        """
        hfd = HypertoroidalFourierDistribution(fftshift(coeff_mat_tmp), "identity")
        tensor = self.grid_values.reshape(*self.n_grid_points)
        hfd.C = fftshift(fftn(tensor))
        hfd_shifted = hfd.shift(shift_by)

        tmp = ifftn(ifftshift(hfd_shifted.C))
        # Check imaginary leakage and discard imag part
        imag_max = max(abs(imag(tmp)))
        if imag_max > 1e-8:
            warnings.warn(
                f"Shift: discarding non-negligible imaginary part (max={imag_max:.2e}).",
                RuntimeWarning,
            )
        shifted_values = real(tmp).reshape(self.grid_values.shape)

        shifted_distribution = self.__class__(
            grid_values=shifted_values,
            grid_type=self.grid_type,
            grid=self.grid,
            enforce_pdf_nonnegative=self.enforce_pdf_nonnegative,
            dim=self.dim,
            n_grid_points=self.n_grid_points,
        )
        return shifted_distribution
        """
    def value_of_closest(self, xa):
        """Return pdf values of the closest grid point for each query.

        Parameters
        ----------
        xa : array_like, shape (n_eval, dim) or (dim,)
        """
        if xa.ndim == 1:
            xa = xa.reshape(1, -1)
        if xa.shape[1] != self.dim:
            raise ValueError(
                f"Expected xa with shape (n_eval, {self.dim}), got {xa.shape}"
            )

        if self.grid is None or self.grid.size == 0:
            raise ValueError("Grid is empty; cannot evaluate value_of_closest.")

        # Broadcast differences: (n_eval, n_grid, dim)
        delta = self.grid[None, :, :] - xa[:, None, :]
        abs_delta = abs(delta)
        dists = sum(min((abs_delta ** 2, (2 * pi - abs_delta) ** 2)), axis=-1)
        min_inds = argmin(dists, axis=1)
        return self.grid_values[min_inds]

    def convolve(self, other):
        if not (
            self.grid_type == "CartesianProd" and other.grid_type == "CartesianProd"
        ):
            raise ValueError("Convolution is only supported for 'CartesianProd' grids.")

        if self.n_grid_points is None or other.n_grid_points is None:
            raise ValueError("Both distributions must have n_grid_points defined.")

        if not all(self.n_grid_points == other.n_grid_points):
            raise ValueError("Convolution requires identical grid resolutions.")

        if not equal(self.grid, other.grid):
            raise ValueError("Convolution requires identical grids.")

        this_tensor = self.grid_values.reshape(*self.n_grid_points)
        other_tensor = other.grid_values.reshape(*other.n_grid_points)

        res_tensor = (
            (2.0 * pi) ** self.dim
            / self.grid_values.size
            * fft.ifftn(fft.fftn(this_tensor) * fft.fftn(other_tensor))
        )

        tmp = (
            (2.0 * pi) ** self.dim
            / self.grid_values.size
            * fft.ifftn(fft.fftn(this_tensor) * fft.fftn(other_tensor))
        )
        imag_max = max(abs(imag(tmp)))
        if imag_max > 1e-8:
            warnings.warn(
                f"Convolve: discarding non-negligible imaginary part (max={imag_max:.2e}).",
                RuntimeWarning,
            )
        convolved_values = real(tmp).reshape(self.grid_values.shape)

        convolved_distribution = self.__class__(
            grid_values=convolved_values,
            grid_type=self.grid_type,
            grid=self.grid,
            enforce_pdf_nonnegative=self.enforce_pdf_nonnegative,
            dim=self.dim,
            n_grid_points=self.n_grid_points,
        )
        return convolved_distribution

    # -------------------------------------------------------- construction API
    @staticmethod
    def from_distribution(
        distribution,
        n_grid_points,
        grid_type: str = "CartesianProd",
        enforce_pdf_nonnegative: bool = True,
    ):
        """Create a grid distribution from another hypertoroidal distribution."""
        if not isinstance(grid_type, str):
            raise TypeError("grid_type must be a string")

        n_grid_points = atleast_1d(n_grid_points)

        """
        HypertoroidalFourierDistribution not yet implemented, so handling not required.
        # Special case: build directly from a HypertoroidalFourierDistribution
        if (
            grid_type == "cart_prod"
            and isinstance(distribution, HypertoroidalFourierDistribution)
            and (
                tuple(distribution.C.shape) == tuple(n_grid_points)
                or (
                    n_grid_points.size == 1
                    and distribution.C.shape[0] == int(n_grid_points[0])
                )
            )
        ):
            grid_values = distribution.pdf_on_grid()
            grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(
                n_grid_points
            )
            hgd = HypertoroidalGridDistribution(
                grid_values=grid_values.reshape(-1),
                grid_type=grid_type,
                grid=grid,
                enforce_pdf_nonnegative=enforce_pdf_nonnegative,
                dim=distribution.dim,
                n_grid_points=n_grid_points,
            )
            return hgd
        """
        # Generic case: sample pdf of the given distribution on a grid.
        hgd = HypertoroidalGridDistribution.from_function(
            distribution.pdf,
            n_grid_points,
            dim=distribution.dim,
            grid_type=grid_type,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )
        return hgd

    @staticmethod
    def from_function(
        fun,
        n_grid_points,
        dim,
        grid_type: str = "CartesianProd",
        enforce_pdf_nonnegative: bool = True,
    ):
        """Construct a grid distribution by sampling a function on a grid.

        Parameters
        ----------
        fun : callable
            Function handle representing a (possibly unnormalized) pdf.
            Must accept an array of shape (n_eval, dim) and return shape (n_eval,).
        n_grid_points : int or sequence of int
            Number of grid points along each dimension.
        dim : int
            Dimensionality of the torus.
        """
        if not isinstance(grid_type, str):
            raise TypeError("grid_type must be a string")

        n_grid_points = atleast_1d(n_grid_points)
        if n_grid_points.size == 1 and dim > 1:
            n_grid_points = repeat(n_grid_points[0], dim)
        elif n_grid_points.size != dim:
            raise ValueError(
                f"n_grid_points must be a scalar or have length dim={dim}, got {n_grid_points}"
            )

        if grid_type == "CartesianProd":
            grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(
                n_grid_points
            )
        else:
            raise ValueError("Grid scheme not recognized")

        # fun expects points as (n_eval, dim)
        grid_values = fun(grid).reshape(-1)

        sgd = HypertoroidalGridDistribution(
            grid_values=grid_values,
            grid_type=grid_type,
            grid=grid,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
            dim=dim,
            n_grid_points=n_grid_points,
        )
        return sgd

    def plot(self, *args, **kwargs):
        # Initially equally weighted and then overwrite to prevent
        # normalization in the constructor of the Dirac distribution
        hdd = HypertoroidalDiracDistribution(
            self.grid,
            (1.0 / len(self.grid_values)) * ones(len(self.grid_values)),
        )
        # Overwrite with actual (possibly unnormalized) weights
        hdd.w = self.grid_values.copy()
        return hdd.plot(*args, **kwargs)

    def plot_interpolated(self, *args, **kwargs):
        if self.dim > 2:
            raise ValueError("Can currently only plot for T1 and T2 torus.")

        transformation = "sqrt" if self.enforce_pdf_nonnegative else "identity"

        n_per_dim_float = self.grid_values.size ** (1.0 / self.dim)
        if not isclose(n_per_dim_float, round(n_per_dim_float)):
            raise ValueError(
                "Number of grid values is not a perfect power of dim; cannot infer grid resolution."
            )
        n_per_dim = int(round(n_per_dim_float))
        sizes = repeat(n_per_dim, self.dim)

        values_tensor = self.grid_values.reshape(*sizes, 1)
        """
        fd = HypertoroidalFourierDistribution.from_function_values(
            values_tensor, sizes, transformation
        )
        return fd.plot(*args, **kwargs)
        """

    def trigonometric_moment(self, n):
        hwd = HypertoroidalDiracDistribution(
            self.grid, self.grid_values / sum(self.grid_values)
        )
        m = hwd.trigonometric_moment(n)
        return m

    def slice_at(self, dims, val, use_fftn=True):
        raise NotImplementedError(
            "slice_at is not yet implemented in the Python version."
        )
