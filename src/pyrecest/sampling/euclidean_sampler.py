# pylint: disable=no-name-in-module,no-member
import warnings

import numpy as np
from pyrecest.backend import eye, zeros
from scipy.special import erfinv as scipy_erfinv
from scipy.stats import qmc

from ..distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from .abstract_sampler import AbstractSampler


class AbstractEuclideanSampler(AbstractSampler):
    pass


class GaussianSampler(AbstractEuclideanSampler):
    def sample_stochastic(self, n_samples: int, dim: int):
        return GaussianDistribution(zeros(dim), eye(dim)).sample(n_samples)


class FibonacciRejectionSampler(AbstractEuclideanSampler):
    """Deterministic rejection sampler using Fibonacci proposal grids."""

    def sample_stochastic(self, n_samples: int, dim: int):
        raise NotImplementedError("Use sample_rejection with a density function.")

    def sample_rejection(
        self,
        pdf,
        n_candidates: int,
        dim: int,
        max_density: float,
        *,
        bounding_box=None,
    ):
        """Sample an arbitrary bounded density with deterministic Fibonacci proposals.

        Parameters
        ----------
        pdf : callable
            Density function accepting an array of shape ``(n_candidates, dim)``.
        n_candidates : int
            Number of deterministic proposal points before rejection.
        dim : int
            Dimension of the Euclidean domain.
        max_density : float
            Upper bound on ``pdf`` inside the bounding box.
        bounding_box : np.ndarray of shape (dim, 2), optional
            Lower and upper bounds for each dimension. Defaults to the unit hypercube.

        Returns
        -------
        samples : np.ndarray of shape (n_accepted, dim)
            Accepted samples.
        info : dict
            Rejection metadata.
        """
        n_candidates, dim, max_density, bounding_box = self._validate_rejection_args(
            n_candidates, dim, max_density, bounding_box
        )

        if n_candidates == 0:
            samples = np.empty((0, dim))
            return samples, self._get_info(
                n_candidates, samples, bounding_box, max_density
            )

        proposal_grid = FibonacciGridSampler().get_uniform_samples(
            n_candidates, dim + 1
        )
        candidate_samples = self._map_to_bounding_box(
            proposal_grid[:, :dim], bounding_box
        )
        density_values = self._evaluate_pdf(pdf, candidate_samples, n_candidates)

        if np.any(density_values < 0):
            raise ValueError("pdf must be nonnegative")
        if np.any(density_values > max_density * (1.0 + 1e-12)):
            raise ValueError("max_density must upper-bound pdf on the bounding box")

        rejection_threshold = proposal_grid[:, dim] * max_density
        accepted = rejection_threshold <= density_values
        samples = candidate_samples[accepted]
        info = self._get_info(n_candidates, samples, bounding_box, max_density)
        return samples, info

    @staticmethod
    def _validate_rejection_args(n_candidates, dim, max_density, bounding_box):
        n_candidates = _validate_integral_argument(n_candidates, "n_candidates")
        dim = _validate_integral_argument(dim, "dim")
        max_density = float(max_density)

        if n_candidates < 0:
            raise ValueError("n_candidates must be nonnegative")
        if dim < 1:
            raise ValueError("dim must be positive")
        if not np.isfinite(max_density) or max_density <= 0:
            raise ValueError("max_density must be positive and finite")

        if bounding_box is None:
            bounding_box = np.column_stack((np.zeros(dim), np.ones(dim)))
        else:
            bounding_box = np.asarray(bounding_box, dtype=float)

        if bounding_box.shape != (dim, 2):
            raise ValueError("bounding_box must have shape (dim, 2)")
        if not np.all(np.isfinite(bounding_box)):
            raise ValueError("bounding_box must be finite")
        if np.any(bounding_box[:, 1] <= bounding_box[:, 0]):
            raise ValueError(
                "bounding_box upper bounds must be greater than lower bounds"
            )

        return n_candidates, dim, max_density, bounding_box

    @staticmethod
    def _map_to_bounding_box(unit_samples, bounding_box):
        lower = bounding_box[:, 0]
        width = bounding_box[:, 1] - bounding_box[:, 0]
        return unit_samples * width + lower

    @staticmethod
    def _evaluate_pdf(pdf, samples, n_candidates):
        density_values = np.asarray(pdf(samples), dtype=float)
        if density_values.ndim == 0:
            density_values = np.full(n_candidates, density_values)
        else:
            density_values = density_values.reshape(-1)

        if density_values.shape[0] != n_candidates:
            raise ValueError("pdf must return one density value per candidate sample")
        if not np.all(np.isfinite(density_values)):
            raise ValueError("pdf must return finite density values")
        return density_values

    @staticmethod
    def _get_info(n_candidates, samples, bounding_box, max_density):
        n_accepted = samples.shape[0]
        return {
            "n_candidates": n_candidates,
            "n_accepted": n_accepted,
            "n_rejected": n_candidates - n_accepted,
            "acceptance_rate": n_accepted / n_candidates if n_candidates else 0.0,
            "bounding_box": bounding_box,
            "max_density": max_density,
        }


class _QMCProposalGridSampler(AbstractEuclideanSampler):
    """Base class for deterministic low-discrepancy proposal grids."""

    def sample_stochastic(self, n_samples: int, dim: int):
        """Return deterministic proposal points on the unit hypercube."""
        return self.get_uniform_samples(n_samples, dim)

    def get_uniform_samples(self, n_samples: int, dim: int):
        """Return deterministic proposal points in ``[0, 1)^dim``.

        Parameters
        ----------
        n_samples : int
            Number of proposal points.
        dim : int
            Dimension of the unit hypercube.

        Returns
        -------
        np.ndarray of shape (n_samples, dim)
        """
        n_samples, dim = self._validate_grid_args(n_samples, dim)
        if n_samples == 0:
            return np.empty((0, dim))
        return self._make_engine(dim).random(n_samples)

    @staticmethod
    def _validate_grid_args(n_samples: int, dim: int):
        n_samples = _validate_integral_argument(n_samples, "n_samples")
        dim = _validate_integral_argument(dim, "dim")
        if n_samples < 0:
            raise ValueError("n_samples must be nonnegative")
        if dim < 1:
            raise ValueError("dim must be positive")
        return n_samples, dim

    def _make_engine(self, dim: int):
        raise NotImplementedError


class SobolGridSampler(_QMCProposalGridSampler):
    """Deterministic Sobol proposal grid on the Euclidean unit hypercube."""

    def _make_engine(self, dim: int):
        return qmc.Sobol(d=dim, scramble=False)


class HaltonGridSampler(_QMCProposalGridSampler):
    """Deterministic Halton proposal grid on the Euclidean unit hypercube."""

    def _make_engine(self, dim: int):
        return qmc.Halton(d=dim, scramble=False)


def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def _validate_integral_argument(value, name: str) -> int:
    """Return a scalar integer argument without silently truncating floats."""
    try:
        array_value = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if array_value.ndim != 0:
        raise ValueError(f"{name} must be a scalar integer")
    if np.issubdtype(array_value.dtype, np.bool_):
        raise ValueError(f"{name} must be an integer")
    if np.issubdtype(array_value.dtype, np.integer):
        return int(array_value)
    if np.issubdtype(array_value.dtype, np.floating):
        float_value = float(array_value)
        if np.isfinite(float_value) and float_value.is_integer():
            return int(float_value)

    raise ValueError(f"{name} must be an integer")


def _validate_gaussian_transform_args(d, covariance, mean):
    if covariance is None:
        covariance = np.eye(d)
    else:
        covariance = np.asarray(covariance, dtype=float)

    if covariance.shape != (d, d):
        raise ValueError("covariance must have shape (dim, dim)")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("covariance must be symmetric")

    covariance_scale = max(1.0, float(np.max(np.abs(covariance))))
    tolerance = 100.0 * np.finfo(float).eps * covariance_scale
    if np.min(np.linalg.eigvalsh(covariance)) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")

    if mean is None:
        mean = np.zeros(d)
    mean = np.asarray(mean, dtype=float).ravel()
    if mean.shape != (d,):
        raise ValueError("mean must have shape (dim,)")
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite")

    return covariance, mean


def _fibonacci_eigen(d):  # pylint: disable=too-many-locals
    """Compute the eigenvector basis V and eigenvalues R of the Fibonacci matrix.

    Based on Purser, "Generalized Fibonacci Grids".

    Parameters
    ----------
    d : int
        Positive integer dimension.

    Returns
    -------
    V : np.ndarray of shape (d, d)
        Eigenvector matrix (columns are eigenvectors).
    R : np.ndarray of shape (d,)
        Eigenvalue-related scaling vector.
    """
    if d == 4:
        # Purser, Generalized Fibonacci Grids..., 7. Generalization at Higher Dimensions
        # 2*4+1==9, no prime, therefore special treatment
        p = (1 + np.sqrt(5)) / 2
        ap = 3 + np.sqrt(5)
        am = 3 - np.sqrt(5)
        bp = np.sqrt(6 * (5 + np.sqrt(5)))
        bm = np.sqrt(6 * (5 - np.sqrt(5)))
        v1 = (am - bm) / 4
        v2 = (ap - bp) / 4
        v3 = -1 / v1
        v4 = -1 / v2
        g = 1 / np.sqrt((1 + v3**2) * (1 + p**2))
        h = 1 / np.sqrt((1 + v4**2) * (1 + p**2))
        V = np.array(
            [
                [p * g, h, p * v3 * g, v4 * h],
                [g, -p * h, v3 * g, -p * v4 * h],
                [-p * v3 * g, -v4 * h, p * g, h],
                [-v3 * g, p * v4 * h, g, -p * h],
            ]
        )
        R = np.array([v1, v2, v3, v4])
    else:
        # EV of Fibonacci Matrix
        # Purser, Generalized Fibonacci Grids..., Appendix, (A.4)
        i1 = np.arange(1, d + 1).reshape(-1, 1)
        j1 = np.arange(1, d + 1).reshape(1, -1)
        V = np.cos((2 * i1 - 1) * (2 * j1 - 1) * np.pi / (4 * d + 2))
        # All columns have the same norm (Paweletz), normalize each column
        a = np.linalg.norm(V, axis=0)
        V = V / a
        j_flat = np.arange(1, d + 1)
        R = (-1) ** (j_flat - 1) / (2 * np.sin((2 * j_flat - 1) * np.pi / (4 * d + 2)))
        if not _is_prime(2 * d + 1):
            warnings.warn("2*D+1 should be prime", UserWarning, stacklevel=2)
    return V, R


class FibonacciGridSampler(AbstractEuclideanSampler):
    """Deterministic Gaussian sampler using multi-dimensional Fibonacci grids.

    Implements the Fibonacci grid sampling from:
      Frisch and Hanebeck, "Deterministic Gaussian Sampling With Generalized
      Fibonacci Grids", FUSION 2021.

    ``sample_stochastic`` returns moment-matched standard normal Fibonacci grid
    samples on R^D.  Despite the method name the samples are deterministic.
    """

    def sample_stochastic(self, n_samples: int, dim: int):
        """Return moment-matched standard normal Fibonacci grid samples.

        Despite the name, these are deterministic samples.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        dim : int
            Dimension of the Euclidean space.

        Returns
        -------
        np.ndarray of shape (n_samples, dim)
        """
        _, xy_stdMM, _ = self._fibonacci_grid(dim, n_samples)
        return xy_stdMM.T  # (n_samples, dim)

    def get_gaussian_samples(self, n_samples, dim, covariance=None, mean=None):
        """Return Fibonacci grid samples transformed to a Gaussian distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        dim : int
            Dimension of the Euclidean space.
        covariance : np.ndarray of shape (dim, dim), optional
            Covariance matrix.  Defaults to identity.
        mean : np.ndarray of shape (dim,), optional
            Mean vector.  Defaults to zeros.

        Returns
        -------
        np.ndarray of shape (n_samples, dim)
        """
        _, _, xy_gauss = self._fibonacci_grid(
            dim, n_samples, covariance=covariance, mean=mean
        )
        return xy_gauss.T  # (n_samples, dim)

    def get_uniform_samples(self, n_samples, dim):
        """Return Fibonacci grid samples uniform on [0, 1]^dim.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        dim : int
            Dimension of the Euclidean space.

        Returns
        -------
        np.ndarray of shape (n_samples, dim)
        """
        xy_equal, _, _ = self._fibonacci_grid(dim, n_samples)
        return xy_equal.T  # (n_samples, dim)

    @staticmethod
    def _fibonacci_grid(
        d, n_points, covariance=None, mean=None, rescale=True
    ):  # pylint: disable=too-many-locals,too-many-statements
        """Generate a multi-dimensional Fibonacci grid.

        Parameters
        ----------
        d : int
            Dimension.
        n_points : int
            Number of grid points.
        covariance : np.ndarray of shape (d, d), optional
            Covariance matrix for the Gaussian output.  Defaults to identity.
        mean : np.ndarray of shape (d,), optional
            Mean vector for the Gaussian output.  Defaults to zeros.
        rescale : bool, optional
            Whether to rescale the grid to fill [0, 1]^d exactly.

        Returns
        -------
        xy_equal : np.ndarray of shape (d, n_points)
            Uniform grid on [0, 1]^d.
        xy_stdMM : np.ndarray of shape (d, n_points)
            Moment-matched standard normal grid on R^d.
        xy_gauss : np.ndarray of shape (d, n_points)
            Gaussian grid on R^d with the given covariance and mean.
        """
        d = _validate_integral_argument(d, "d")
        n_points = _validate_integral_argument(n_points, "n_points")
        if d < 1:
            raise ValueError("d must be positive")
        if n_points < 0:
            raise ValueError("n_points must be nonnegative")

        covariance, mean = _validate_gaussian_transform_args(d, covariance, mean)

        if n_points == 0:
            empty_arr = np.empty((d, 0))
            return empty_arr.copy(), empty_arr.copy(), empty_arr.copy()
        if n_points == 1:
            xy_equal = np.full((d, 1), 0.5)
            xy_stdMM = np.zeros((d, 1))
            xy_gauss = mean.reshape(-1, 1).copy()
            return xy_equal, xy_stdMM, xy_gauss

        V, _ = _fibonacci_eigen(d)

        # Maximum L1 norm of columns of V (= size of outer cube)
        outer = np.max(np.sum(np.abs(V), axis=0))

        # Number of points per side of the auxiliary hypercube
        L0 = int(np.ceil(n_points ** (1.0 / d)))
        spc = 1.0 / L0
        extra = 2
        L1 = int(np.ceil(outer / spc)) + extra
        if n_points % 2 != L1 % 2:
            L1 += 1

        # Centered sampling vector with spacing spc
        vec = np.arange(L1) * spc
        vec = vec - vec.mean()

        # Build D-dimensional regular grid: each column of xy is one grid point
        grids = np.meshgrid(*([vec] * d), indexing="ij")
        xy = np.vstack([g.ravel() for g in grids])  # (d, L1^d)

        # Rotate grid by the Fibonacci eigenvectors
        xy = V @ xy

        # Identify points fully inside [-1/2, 1/2]^d
        ind = np.all((xy <= 0.5) & (xy >= -0.5), axis=0)
        assert (
            ind.sum() % 2 == n_points % 2
        ), "Parity of in-box points does not match n_points"

        # Keep only points whose non-first coordinates are in [-1/2, 1/2]
        ind0 = np.all((xy[1:, :] <= 0.5) & (xy[1:, :] >= -0.5), axis=0)  # noqa: E203
        xy = xy[:, ind0]
        ind = ind[ind0]

        # Fine-tune the number of samples by adjusting the x_1 boundary
        n_current = int(ind.sum())
        diff = n_points - n_current
        assert (
            diff % 2 == 0
        ), f"Sample count parity mismatch after slicing: expected difference to be even but got {diff}"
        n_add = diff // 2

        sort_idx = np.argsort(xy[0, :], kind="stable")  # noqa: E203
        srt = xy[0, sort_idx]

        where_le = np.where(srt <= 0.5)[0]
        where_ge = np.where(srt >= -0.5)[0]
        ibp = int(where_le[-1]) if len(where_le) > 0 else -1
        ibm = int(where_ge[0]) if len(where_ge) > 0 else len(srt)

        border_x = 0.5
        if n_add > 0:
            # Add n_add samples just outside the right boundary
            ind[sort_idx[ibp + 1 : ibp + 1 + n_add]] = True  # noqa: E203
            # Add n_add samples just outside the left boundary
            ind[sort_idx[ibm - n_add : ibm]] = True  # noqa: E203
            border_x = float(srt[ibp + n_add])
        elif n_add < 0:
            # Remove |n_add| samples just inside the right boundary
            ind[sort_idx[ibp + n_add + 1 : ibp + 1]] = False  # noqa: E203
            # Remove |n_add| samples just inside the left boundary
            ind[sort_idx[ibm : ibm - n_add]] = False  # noqa: E203
            border_x = float(srt[ibp + n_add])

        # Sanity check: border_x must lie within the grid extent
        border_vec = np.ones(d) * 0.5
        border_vec[0] = border_x
        border_vec_rot = V.T @ border_vec
        assert np.all(
            np.abs(border_vec_rot) <= np.max(vec)
        ), "Increase 'extra' variable"
        assert int(ind.sum()) == n_points

        # Extract the selected points and center them
        xy = xy[:, ind]
        xy = xy - xy.mean(axis=1, keepdims=True)

        # Rescale so that the outermost point hits ±(1/2 - 1/(2·n_points))
        if n_points > 1 and rescale:
            border_wanted = 0.5 - 1.0 / (2 * n_points)
            fac = np.max(xy, axis=1, keepdims=True) / border_wanted
            xy = xy / fac

        # Translate from [-1/2, 1/2]^d to [0, 1]^d
        xy_equal = xy + 0.5

        # Uniform → standard normal via the probit transform
        xy_std = np.sqrt(2) * scipy_erfinv(2 * xy_equal - 1)

        # Moment-match: scale so that each marginal has unit variance
        fac_mm = xy_std.std(axis=1, ddof=0, keepdims=True)
        xy_stdMM = xy_std / fac_mm

        # Transform to a Gaussian with the requested covariance and mean
        C_vals, C_vecs = np.linalg.eigh(covariance)
        C_vals = np.maximum(C_vals, 0.0)
        xy_gauss = C_vecs @ np.diag(np.sqrt(C_vals)) @ xy_stdMM + mean.reshape(-1, 1)

        return xy_equal, xy_stdMM, xy_gauss
