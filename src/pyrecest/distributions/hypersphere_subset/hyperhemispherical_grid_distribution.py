import warnings

from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    allclose,
    argmax,
    argmin,
    array,
    concatenate,
    linalg,
    minimum,
    vstack,
)

from ...sampling.hyperspherical_sampler import get_grid_hyperhemisphere
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
from .abstract_hypersphere_subset_grid_distribution import (
    AbstractHypersphereSubsetGridDistribution,
)
from .custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from .hyperspherical_dirac_distribution import HypersphericalDiracDistribution


class HyperhemisphericalGridDistribution(
    AbstractHypersphereSubsetGridDistribution, AbstractHyperhemisphericalDistribution
):
    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        if not bool(all(abs(grid) <= 1 + 1e-12)):
            raise ValueError("Grid points must not lie outside the unit cube.")
        if not bool(all(grid[:, -1] >= -1e-12)):
            raise ValueError("Always using upper hemisphere along the last dimension.")

        super().__init__(grid, grid_values, enforce_pdf_nonnegative)

    def mean_direction(self):
        warnings.warn(
            "For hyperhemispheres, `mean_direction` returns the mode and not the true mean direction.",
            UserWarning,
        )
        index_max = argmax(self.grid_values)
        return self.grid[index_max, :]

    def to_full_sphere(self, method="antipodal"):
        if method != "antipodal":
            raise ValueError(
                "Currently only 'antipodal' method is supported for converting hyperhemispherical grid distributions to full-sphere grid distributions."
            )
        from .hyperspherical_grid_distribution import HypersphericalGridDistribution

        grid_full = vstack((self.grid, -self.grid))
        grid_values_ = 0.5 * concatenate((self.grid_values, self.grid_values))
        return HypersphericalGridDistribution(grid_full, grid_values_)

    def plot(self):
        hdd = HypersphericalDiracDistribution(self.grid, self.grid_values.T)
        return hdd.plot()

    def plot_interpolated(self):
        hdgd = self.to_full_sphere()

        def doubled_pdf(point):
            return 2 * hdgd.pdf(point)

        hhgd_interp = CustomHyperhemisphericalDistribution(doubled_pdf, self.dim)
        return hhgd_interp.plot()

    def get_closest_point(self, xs):
        xs = array(xs)

        if xs.ndim == 1:
            if xs.shape[0] != self.input_dim:
                raise ValueError(
                    f"xs must have length {self.input_dim}, got {xs.shape[0]}."
                )
            xs = xs[None, :]
        elif xs.ndim == 2:
            if xs.shape[1] == self.input_dim:
                pass
            elif xs.shape[0] == self.input_dim:
                xs = xs.T
            else:
                raise ValueError(
                    f"xs must have shape (n, input_dim) with input_dim={self.input_dim}."
                )
        else:
            raise ValueError("xs must be a 1D or 2D array.")

        diff1 = xs[:, None, :] - self.grid[None, :, :]
        diff2 = xs[:, None, :] + self.grid[None, :, :]
        all_distances = minimum(linalg.norm(diff1, axis=2), linalg.norm(diff2, axis=2))
        indices = argmin(all_distances, axis=1)
        points = self.get_grid_point(indices)

        if points.ndim == 2 and points.shape[0] == 1:
            points = points[0, :]
            indices = indices[0]

        return points, indices

    def get_manifold_size(self):
        return (
            0.5
            * AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.dim
            )
        )

    @beartype
    def multiply(
        self: "HyperhemisphericalGridDistribution",
        other: "HyperhemisphericalGridDistribution",
    ) -> "HyperhemisphericalGridDistribution":
        if (
            self.dim != other.dim
            or self.get_grid().shape != other.get_grid().shape
            or not allclose(self.get_grid(), other.get_grid())
        ):
            raise ValueError("Multiply:IncompatibleGrid")

        hgd_filtered = self.to_full_sphere().multiply(other.to_full_sphere())
        n_hemi = self.grid.shape[0]
        hemi_grid = hgd_filtered.grid[:n_hemi]
        hemi_values = 2.0 * hgd_filtered.grid_values[:n_hemi]
        return HyperhemisphericalGridDistribution(
            hemi_grid, hemi_values, enforce_pdf_nonnegative=True
        )

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        dim=2,
        grid_type="leopardi_symm",
        enforce_pdf_nonnegative=True,
    ):
        if grid_type not in ("leopardi_symm", "healpix"):
            raise ValueError(
                "For hyperhemispheres, use one of the symmetric grid types 'leopardi_symm' or 'healpix'."
            )
        grid, _ = get_grid_hyperhemisphere(grid_type, no_of_grid_points, dim=dim)
        grid_values = fun(grid)
        sgd = HyperhemisphericalGridDistribution(
            grid, grid_values, enforce_pdf_nonnegative=enforce_pdf_nonnegative
        )
        sgd.grid_type = grid_type
        return sgd
