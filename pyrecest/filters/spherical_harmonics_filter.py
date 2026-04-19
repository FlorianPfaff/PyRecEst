import copy
import warnings

from pyrecest.backend import (  # pylint: disable=redefined-builtin
    abs,
    arccos,
    arctan2,
    array,
    clip,
    linalg,
    maximum,
    ones,
    pi,
    sqrt,
    stack,
    zeros,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)

from .abstract_filter import AbstractFilter
from .manifold_mixins import HypersphericalFilterMixin


class SphericalHarmonicsFilter(AbstractFilter, HypersphericalFilterMixin):
    """Filter on the unit sphere using spherical harmonic representations.

    Supports both the ``'identity'`` transformation (coefficients represent
    the density directly) and the ``'sqrt'`` transformation (coefficients
    represent the square-root of the density).

    References
    ----------
    Florian Pfaff, Gerhard Kurz, and Uwe D. Hanebeck,
    "Filtering on the Unit Sphere Using Spherical Harmonics",
    Proceedings of the 2017 IEEE International Conference on Multisensor
    Fusion and Integration for Intelligent Systems (MFI 2017),
    Daegu, Korea, November 2017.
    """

    def __init__(self, degree, transformation="identity"):
        HypersphericalFilterMixin.__init__(self)
        coeff_mat = zeros((degree + 1, 2 * degree + 1), dtype=complex)
        if transformation == "identity":
            coeff_mat[0, 0] = 1.0 / sqrt(4.0 * pi)
        elif transformation == "sqrt":
            coeff_mat[0, 0] = 1.0
        else:
            raise ValueError(f"Unknown transformation: '{transformation}'")
        initial_state = SphericalHarmonicsDistributionComplex(coeff_mat, transformation)
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state / state property
    # ------------------------------------------------------------------

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert isinstance(
            new_state, SphericalHarmonicsDistributionComplex
        ), "filter_state must be a SphericalHarmonicsDistributionComplex"
        self._filter_state = copy.deepcopy(new_state)

    @property
    def state(self):
        """Alias for :attr:`filter_state`."""
        return self._filter_state

    # ------------------------------------------------------------------
    # Public interface methods
    # ------------------------------------------------------------------

    def set_state(self, state):
        """Set the filter state with optional warnings about mismatches."""
        assert isinstance(
            state, SphericalHarmonicsDistributionComplex
        ), "state must be a SphericalHarmonicsDistributionComplex"
        if self._filter_state.transformation != state.transformation:
            warnings.warn(
                "setState:transDiffer: New density is transformed differently.",
                stacklevel=2,
            )
        if self._filter_state.coeff_mat.shape != state.coeff_mat.shape:
            warnings.warn(
                "setState:noOfCoeffsDiffer: New density has different number of "
                "coefficients.",
                stacklevel=2,
            )
        self._filter_state = copy.deepcopy(state)

    def get_estimate(self):
        """Return the current filter state."""
        return self._filter_state

    def get_estimate_mean(self):
        """Return the mean direction of the current filter state."""
        return self._filter_state.mean_direction()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, sys_noise):
        """Predict via spherical convolution with a *zonal* system noise SHD.

        Parameters
        ----------
        sys_noise : SphericalHarmonicsDistributionComplex
            Must be a zonal distribution (rotationally symmetric around the
            z-axis) in the same transformation as the filter state.
        """
        assert isinstance(
            sys_noise, SphericalHarmonicsDistributionComplex
        ), "sys_noise must be a SphericalHarmonicsDistributionComplex"
        if (
            self._filter_state.transformation == "sqrt"
            and sys_noise.transformation == "identity"
        ):
            state_degree = self._filter_state.coeff_mat.shape[0] - 1
            noise_degree = sys_noise.coeff_mat.shape[0] - 1
            assert 2 * state_degree == noise_degree, (
                "If the sqrt variant is used and sys_noise is given in "
                "identity form, sys_noise should have degree 2 * state_degree."
            )
        self._filter_state = self._filter_state.convolve(sys_noise)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, meas_noise, z):
        """Update by multiplying the state with a (possibly rotated) noise SHD.

        *meas_noise* should be a zonal SHD with its axis of symmetry along
        [0, 0, 1].  If the measurement *z* differs from [0, 0, 1] the noise is
        rotated to align with *z* before the multiplication.

        Parameters
        ----------
        meas_noise : SphericalHarmonicsDistributionComplex
            Zonal measurement noise (axis along [0, 0, 1]).
        z : array-like, shape (3,)
            Measurement direction on the unit sphere.
        """
        assert isinstance(
            meas_noise, SphericalHarmonicsDistributionComplex
        ), "meas_noise must be a SphericalHarmonicsDistributionComplex"
        z = array(z, dtype=float).ravel()
        z_norm = linalg.norm(z)
        not_near_north_pole = (
            abs(z[0]) > 1e-6 or abs(z[1]) > 1e-6 or abs(z[2] - 1.0) > 1e-6
        )
        if z_norm > 1e-6 and not_near_north_pole:
            warnings.warn(
                "SphericalHarmonicsFilter:rotationRequired: "
                "Performance may be low for z != [0, 0, 1]. "
                "Using update_nonlinear may yield faster results.",
                stacklevel=2,
            )
            phi = arctan2(z[1], z[0])  # azimuth
            theta = arccos(clip(z[2] / z_norm, -1.0, 1.0))  # colatitude
            meas_noise = meas_noise.rotate(0.0, theta, phi)
        self._filter_state = self._filter_state.multiply(meas_noise)

    def update_nonlinear(self, likelihood, z):
        """Nonlinear Bayesian update via a likelihood function.

        Parameters
        ----------
        likelihood : callable
            ``likelihood(z, pts)`` where *pts* is a ``(3, N)`` matrix of
            Cartesian coordinates on the unit sphere and the return value is a
            length-N array of likelihood values.
        z : array-like
            Measurement (passed through to *likelihood* unchanged).
        """
        self._update_nonlinear_impl([likelihood], [z])

    def update_nonlinear_multiple(self, likelihoods, measurements):
        """Nonlinear update using a list of likelihood functions simultaneously.

        Parameters
        ----------
        likelihoods : list of callables
            Each element is a likelihood function as described in
            :meth:`update_nonlinear`.
        measurements : list of array-like
            Corresponding measurements.
        """
        assert len(likelihoods) == len(
            measurements
        ), "likelihoods and measurements must have the same length"
        self._update_nonlinear_impl(likelihoods, measurements)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_nonlinear_impl(
        self, likelihoods, measurements
    ):  # pylint: disable=too-many-locals
        """Shared implementation for single and multiple nonlinear updates."""

        degree = self._filter_state.coeff_mat.shape[0] - 1

        # DH grid coordinates
        x_c, y_c, z_c, grid_shape = (
            SphericalHarmonicsDistributionComplex._get_dh_grid_cartesian(  # pylint: disable=protected-access
                degree
            )
        )
        # (3, N) matrix for likelihood calls
        grid_pts = stack([x_c, y_c, z_c], axis=0)

        # Evaluate current state on the DH grid
        fval_curr = (
            self._filter_state._eval_on_grid()  # pylint: disable=protected-access
        )

        # Accumulate likelihood values over all (likelihood, measurement) pairs
        likelihood_vals = ones(grid_shape, dtype=float)
        for lk, zk in zip(likelihoods, measurements):
            lv = array(lk(zk, grid_pts), dtype=float).reshape(grid_shape)
            likelihood_vals *= lv

        # Scale factor: multiplying by 2^n keeps values away from zero so that
        # the SHT fit and subsequent normalisation remain numerically stable
        # when the product likelihood is very small (e.g. many weak likelihoods).
        # This factor is divided out implicitly by the normalisation step.
        scale = float(2 ** len(likelihoods))

        if self._filter_state.transformation == "identity":
            fval_new = scale * fval_curr * likelihood_vals
        elif self._filter_state.transformation == "sqrt":
            fval_new = scale * fval_curr * sqrt(maximum(likelihood_vals, 0.0))
        else:
            raise ValueError(
                f"Unsupported transformation: '{self._filter_state.transformation}'"
            )

        self._filter_state = SphericalHarmonicsDistributionComplex._fit_from_grid(  # pylint: disable=protected-access
            fval_new, degree, self._filter_state.transformation
        )
