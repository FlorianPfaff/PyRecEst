import itertools
from abc import abstractmethod

# pylint: disable=no-name-in-module,no-member
from pyrecest import backend
from pyrecest.backend import (
    arange,
    arccos,
    arctan2,
    array,
    column_stack,
    cos,
    deg2rad,
    empty,
    flip,
    linspace,
    pi,
    sin,
    sqrt,
    stack,
    vstack,
)
from pyrecest.distributions import (
    AbstractSphericalDistribution,
    HyperhemisphericalUniformDistribution,
    HypersphericalUniformDistribution,
)

from .abstract_sampler import AbstractSampler
from .hypertoroidal_sampler import CircularUniformSampler
from .leopardi_sampler import get_partition_points_cartesian


def get_grid_hypersphere(method: str, grid_density_parameter: int, dim: int):
    if method == "healpix":
        samples, grid_specific_description = HealpixSampler().get_grid(
            grid_density_parameter, dim=dim
        )
    elif method == "driscoll_healy":
        samples, grid_specific_description = DriscollHealySampler().get_grid(
            grid_density_parameter, dim=dim
        )
    elif method in ("fibonacci", "spherical_fibonacci"):
        samples, grid_specific_description = SphericalFibonacciSampler().get_grid(
            grid_density_parameter, dim=dim
        )
    elif method == "healpix_hopf":
        samples, grid_specific_description = HealpixHopfSampler().get_grid(
            grid_density_parameter, dim=dim
        )
    else:
        raise ValueError(f"Unknown method {method}")

    return samples, grid_specific_description


def get_grid_sphere(method: str, grid_density_parameter: int):
    return get_grid_hypersphere(method, grid_density_parameter, dim=2)


def get_grid_hyperhemisphere(method: str, grid_density_parameter: int, dim: int):
    if method == "leopardi":
        ls = SymmetricLeopardiSampler(
            original_code_column_order=True, delete_half=True, symmetry_type="plane"
        )
        samples, _ = ls.get_grid(grid_density_parameter * 2, dim)
        # To have upper half along last dim instead of first
        grid_specific_description = {
            "scheme": "leopardi_hemisphere",
            "n_side": grid_density_parameter,
        }
    else:
        raise ValueError(f"Unknown method {method}")

    return samples, grid_specific_description


class AbstractHypersphericalUniformSampler(AbstractSampler):
    def sample_stochastic(self, n_samples: int, dim: int):
        return HypersphericalUniformDistribution(dim).sample(n_samples)

    @abstractmethod
    def get_grid(self, grid_density_parameter, dim: int):
        raise NotImplementedError()


class AbstractHyperhemisphericalUniformSampler(AbstractSampler):
    def sample_stochastic(self, n_samples: int, dim: int):
        return HyperhemisphericalUniformDistribution(dim).sample(n_samples)

    @abstractmethod
    def get_grid(self, grid_density_parameter, dim: int):
        raise NotImplementedError()


class AbstractSphericalUniformSampler(AbstractHypersphericalUniformSampler):
    def sample_stochastic(
        self, n_samples: int, dim: int = 2
    ):  # Only having dim there for interface compatibility
        assert dim == 2
        return HypersphericalUniformDistribution(2).sample(n_samples)


class AbstractSphericalCoordinatesBasedSampler(AbstractSphericalUniformSampler):
    @abstractmethod
    def get_grid_spherical_coordinates(
        self,
        grid_density_parameter: int,
    ):
        raise NotImplementedError()

    def get_grid(self, grid_density_parameter, dim: int = 2):
        assert (
            dim == 2
        ), "AbstractSphericalCoordinatesBasedSampler is supposed to be used for the sphere, i.e. dim=2"
        phi, theta, grid_specific_description = self.get_grid_spherical_coordinates(
            grid_density_parameter
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        grid = column_stack((x, y, z))

        return grid, grid_specific_description


class SphericalCoordinatesBasedFixedResolutionSampler(
    AbstractSphericalCoordinatesBasedSampler
):
    def get_grid_spherical_coordinates(self, grid_density_parameter):
        res_lon, res_lat = grid_density_parameter
        assert grid_density_parameter.shape[0] == 2
        phi = linspace(0.0, 2 * pi, num=res_lon, endpoint=False)
        theta = linspace(pi / (res_lat + 1), pi, num=res_lat, endpoint=False)
        return phi, theta, {"res_lat": res_lat, "res_lon": res_lon}


class HealpixSampler(AbstractHypersphericalUniformSampler):
    def get_grid(self, grid_density_parameter, dim: int = 2):
        import healpy as hp

        assert (
            dim == 2
        ), "HealpixSampler is supposed to be used for the sphere, i.e. dim=2"

        n_side = grid_density_parameter
        n_areas = hp.nside2npix(n_side)
        x, y, z = hp.pix2vec(n_side, arange(n_areas))
        grid = column_stack((x, y, z))

        grid_specific_description = {
            "scheme": "healpix",
            "n_side": grid_density_parameter,
        }

        return grid, grid_specific_description


class LeopardiSampler(AbstractHypersphericalUniformSampler):
    def __init__(self, original_code_column_order=True):
        self.original_code_column_order = original_code_column_order
        assert backend.__backend_name__ != "jax", "Backend unsupported"

    def get_grid(self, grid_density_parameter, dim: int):
        # Use [::-1] due to different convention
        grid_eucl = get_partition_points_cartesian(
            dim, grid_density_parameter, delete_half=False, symmetry_type="asymm"
        )

        if self.original_code_column_order:
            grid_eucl = flip(grid_eucl, axis=1)
            grid_eucl[:, [0, 1]] = grid_eucl[:, [1, 0]]

        grid_specific_description = {
            "scheme": "leopardi",
            "n_side": grid_density_parameter,
        }
        return grid_eucl, grid_specific_description


class SymmetricLeopardiSampler(AbstractHypersphericalUniformSampler):
    def __init__(
        self, original_code_column_order=True, delete_half=False, symmetry_type="plane"
    ):
        self.original_code_column_order = original_code_column_order
        self.delete_half = delete_half
        self.symmetry_type = symmetry_type
        assert backend.__backend_name__ != "jax", "Backend unsupported"

    def get_grid(self, grid_density_parameter, dim: int):
        # Use [::-1] due to different convention
        grid_eucl = get_partition_points_cartesian(
            dim,
            grid_density_parameter,
            delete_half=self.delete_half,
            symmetry_type=self.symmetry_type,
        )

        if self.original_code_column_order:
            grid_eucl = flip(grid_eucl, axis=1)
            grid_eucl[:, [0, 1]] = grid_eucl[:, [1, 0]]

        grid_specific_description = {
            "scheme": "leopardi_symm",
            "n_side": grid_density_parameter,
            "delete_half": self.delete_half,
            "symmetry_type": self.symmetry_type,
        }
        return grid_eucl, grid_specific_description


class DriscollHealySampler(AbstractSphericalCoordinatesBasedSampler):
    def get_grid_spherical_coordinates(self, grid_density_parameter: int):
        import pyshtools as pysh

        grid = pysh.SHGrid.from_zeros(grid_density_parameter)

        # Get the longitudes (phi) and latitudes (theta) directly from the grid
        phi_deg_mat = grid.lons()
        theta_deg_mat = grid.lats()

        phi_theta_stacked_deg = array(
            list(itertools.product(phi_deg_mat, theta_deg_mat))
        )
        phi_theta_stacked_rad = deg2rad(phi_theta_stacked_deg)

        phi = phi_theta_stacked_rad[:, 0]
        theta = phi_theta_stacked_rad[:, 1]

        grid_specific_description = {
            "scheme": "driscoll_healy",
            "l_max": grid_density_parameter,
            "n_lat": grid.nlat,
            "n_lon": grid.nlon,
        }

        return phi, theta, grid_specific_description


class SphericalFibonacciSampler(AbstractSphericalCoordinatesBasedSampler):
    def get_grid_spherical_coordinates(self, grid_density_parameter: int):
        indices = arange(0, grid_density_parameter, dtype=float) + 0.5
        phi = pi * (1 + 5**0.5) * indices
        theta = arccos(1 - 2 * indices / grid_density_parameter)
        grid_specific_description = {
            "scheme": "spherical_fibonacci",
            "n_samples": grid_density_parameter,
        }
        return phi, theta, grid_specific_description


class AbstractHopfBasedS3Sampler(AbstractHypersphericalUniformSampler):
    @staticmethod
    def hopf_coordinates_to_quaterion_yershova(θ, ϕ, ψ):
        """
        One possible way to index the S3-sphere via the hopf fibration.
        Using the convention from
        "Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration"
        by
        Anna Yershova, Swati Jain, Steven M. LaValle, Julie C. Mitchell
        As in appendix (or in Eq 4 if one reorders it).
        """

        quaternions = stack(
            [
                cos(θ / 2) * cos(ψ / 2),
                cos(θ / 2) * sin(ψ / 2),
                sin(θ / 2) * cos(ϕ + ψ / 2),
                sin(θ / 2) * sin(ϕ + ψ / 2),
            ],
            axis=1,
        )

        return quaternions

    @staticmethod
    def quaternion_to_hopf_yershova(q):
        θ = 2 * arccos(sqrt(q[:, 0] ** 2 + q[:, 1] ** 2))
        ϕ = arctan2(q[:, 3], q[:, 2]) - arctan2(q[:, 1], q[:, 0])
        ψ = 2 * arctan2(q[:, 1], q[:, 0])
        return θ, ϕ, ψ


# pylint: disable=too-many-locals
class HealpixHopfSampler(AbstractHopfBasedS3Sampler):
    def get_grid(self, grid_density_parameter, dim: int = 3):
        """
        Hopf coordinates are (θ, ϕ, ψ) where θ and ϕ are the angles for the sphere and ψ is the angle on the circle
        First parameter is the number of points on the sphere, second parameter is the number of points on the circle.
        """
        import healpy as hp

        assert (
            dim == 3
        ), "HealpixHopfSampler is supposed to be used for the 3-sphere, i.e. dim=3"

        if isinstance(grid_density_parameter, int):
            grid_density_parameter = [grid_density_parameter]

        s3_points_list = []

        for i in range(grid_density_parameter[0] + 1):
            if len(grid_density_parameter) == 2:
                n_sample_circle = grid_density_parameter[1]
            else:
                n_sample_circle = 2**i * 6

            psi_points = CircularUniformSampler().get_grid(n_sample_circle)

            assert len(psi_points) != 0

            nside = 2**i
            numpixels = hp.nside2npix(nside)

            healpix_points = empty((numpixels, 2))
            for j in range(numpixels):
                theta, phi = hp.pix2ang(nside, j, nest=True)
                healpix_points[j] = array([theta, phi])

            for j in range(len(healpix_points)):
                for k in range(len(psi_points)):
                    temp = array(
                        [healpix_points[j, 0], healpix_points[j, 1], psi_points[k]]
                    )
                    s3_points_list.append(temp)

        s3_points = vstack(s3_points_list)  # Need to stack like this and unpack
        grid = AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(
            s3_points[:, 0], s3_points[:, 1], s3_points[:, 2]
        )

        grid_specific_description = {
            "scheme": "healpix_hopf",
            "layer-parameter": grid_density_parameter,
        }
        return grid, grid_specific_description


class FibonacciHopfSampler(AbstractHopfBasedS3Sampler):
    def get_grid(self, grid_density_parameter):
        """
        Hopf coordinates are (θ, ϕ, ψ) where θ and ϕ are the angles for the sphere and ψ is the angle on the circle
        First parameter is the number of points on the sphere, second parameter is the number of points on the circle.
        """
        if isinstance(grid_density_parameter, int):
            grid_density_parameter = [grid_density_parameter]

        s3_points_list = []

        # Step 1: Discretize the sphere using the Fibonacci grid
        spherical_sampler = SphericalFibonacciSampler()
        phi, theta, _ = spherical_sampler.get_grid_spherical_coordinates(
            grid_density_parameter[0]
        )
        spherical_points = column_stack((theta, phi))  # stack to match expected shape

        # Step 2: Discretize the unit circle using the circular grid
        circular_sampler = CircularUniformSampler()
        if len(grid_density_parameter) == 2:
            n_sample_circle = grid_density_parameter[1]
        else:
            n_sample_circle = sqrt(grid_density_parameter[0])
        psi_points = circular_sampler.get_grid(n_sample_circle)

        # Step 3: Combine the two grids to generate a grid for S3
        for spherical_point in spherical_points:
            for psi in psi_points:
                s3_point = array([spherical_point[0], spherical_point[1], psi])
                s3_points_list.append(s3_point)

        s3_points = vstack(s3_points_list)
        grid = AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(
            s3_points[:, 0], s3_points[:, 1], s3_points[:, 2]
        )

        grid_specific_description = {
            "scheme": "fibonacci_hopf",
            "layer-parameter": grid_density_parameter,
        }
        return grid, grid_specific_description
