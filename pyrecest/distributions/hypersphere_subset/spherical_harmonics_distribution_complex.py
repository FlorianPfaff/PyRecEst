import warnings

import pyrecest
import scipy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    array,
    atleast_2d,
    column_stack,
    complex128,
    conj,
    empty,
    full,
    imag,
    isnan,
    zeros_like,
    linalg,
    maximum,
    pi,
    real,
    reshape,
    shape,
    sin,
    sqrt,
    zeros,
    cos,
    meshgrid,
    deg2rad,
)

# pylint: disable=E0611
from scipy.special import sph_harm_y

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution
from .abstract_spherical_harmonics_distribution import (
    AbstractSphericalHarmonicsDistribution,
)


class SphericalHarmonicsDistributionComplex(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation="identity", assert_real=True):
        assert (
            pyrecest.backend.__backend_name__ != "jax"  # pylint: disable=no-member
        ), "SphericalHarmonicsDistributionComplex is not supported on the JAX backend"
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)
        self.assert_real = assert_real

    def value(self, xs):
        xs = atleast_2d(xs)
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            xs[:, 0], xs[:, 1], xs[:, 2]
        )
        return self.value_sph(phi, theta)

    def value_sph(self, phi, theta):
        vals = zeros((theta.shape[0],), dtype=complex128)
        for n_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-n_curr, n_curr + 1):
                # Evaluate it for all query points at once
                y_lm = sph_harm_y(n_curr, m_curr, theta, phi)
                vals += self.coeff_mat[n_curr, n_curr + m_curr] * y_lm

        if self.assert_real:
            assert all(
                abs(imag(vals)) < 5e-8
            ), "Coefficients apparently do not represent a real function."
            return real(vals)

        return vals

    def to_spherical_harmonics_distribution_real(self):
        from .spherical_harmonics_distribution_real import (
            SphericalHarmonicsDistributionReal,
        )

        if self.transformation != "identity":
            raise ValueError("Transformation currently not supported")

        coeff_mat_real = empty(self.coeff_mat.shape, dtype=float)

        coeff_mat_real[0, 0] = real(self.coeff_mat[0, 0])

        for n in range(
            1, self.coeff_mat.shape[0]
        ):  # Use n instead of l to comply with PEP 8
            for m in range(-n, n + 1):
                if m < 0:
                    coeff_mat_real[n, n + m] = (
                        (-1) ** m
                        * sqrt(2.0)
                        * (-1 if (-m) % 2 else 1)
                        * imag(self.coeff_mat[n, n + m])
                    )
                elif m > 0:
                    coeff_mat_real[n, n + m] = (
                        sqrt(2.0)
                        * (-1 if m % 2 else 1)
                        * real(self.coeff_mat[n, n + m])
                    )
                else:  # m == 0
                    coeff_mat_real[n, n] = real(self.coeff_mat[n, n])

        shd = SphericalHarmonicsDistributionReal(
            real(coeff_mat_real), self.transformation
        )

        return shd

    def mean_direction(self):
        if self.coeff_mat.shape[0] <= 1:
            raise ValueError("Too few coefficients available to calculate the mean")

        y = imag(self.coeff_mat[1, 0] + self.coeff_mat[1, 2]) / sqrt(2.0)
        x = real(self.coeff_mat[1, 0] - self.coeff_mat[1, 2]) / sqrt(2.0)
        z = real(self.coeff_mat[1, 1])

        if linalg.norm(array([x, y, z])) < 1e-9:
            raise ValueError(
                "Coefficients of degree 1 are almost zero. Therefore, no meaningful mean is available"
            )

        mu = array([x, y, z]) / linalg.norm(array([x, y, z]))

        return mu

    @staticmethod
    def from_distribution_via_integral(dist, degree, transformation="identity"):
        assert (
            isinstance(dist, AbstractHypersphericalDistribution) and dist.dim == 2
        ), "dist must be a distribution on the sphere."
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(
            dist.pdf, degree, transformation
        )
        return shd

    @staticmethod
    def _fun_cart_to_fun_sph(fun_cart):
        """Convert a function using Cartesian coordinates to one using spherical coordinates."""

        def fun_sph(phi, theta):
            x, y, z = AbstractSphericalDistribution.sph_to_cart(
                phi.ravel(), theta.ravel()
            )
            vals = fun_cart(column_stack((x, y, z)))
            return reshape(vals, shape(theta))

        return fun_sph

    @staticmethod
    def from_function_via_integral_cart(fun_cart, degree, transformation="identity"):
        fun_sph = SphericalHarmonicsDistributionComplex._fun_cart_to_fun_sph(fun_cart)
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_sph(
            fun_sph, degree, transformation
        )
        return shd

    @staticmethod
    def from_function_via_integral_sph(fun, degree, transformation="identity"):
        if transformation == "sqrt":
            raise NotImplementedError("Transformations are not supported yet")
        if transformation == "identity":
            fun_with_trans = fun
        else:
            raise ValueError("Transformation not supported")

        coeff_mat = full((degree + 1, 2 * degree + 1), float("NaN"), dtype=complex128)

        def real_part(phi, theta, n, m):
            return real(
                fun_with_trans(array(phi), array(theta))
                * conj(array(sph_harm_y(n, m, theta, phi)))
                * sin(theta)
            )

        def imag_part(phi, theta, n, m):
            return imag(
                fun_with_trans(array(phi), array(theta))
                * conj(array(sph_harm_y(n, m, theta, phi)))
                * sin(theta)
            )

        for n in range(degree + 1):  # Use n instead of l to comply with PEP 8
            for m in range(-n, n + 1):
                real_integral, _ = scipy.integrate.nquad(
                    real_part, [[0.0, 2.0 * pi], [0.0, pi]], args=(n, m)
                )
                imag_integral, _ = scipy.integrate.nquad(
                    imag_part, [[0.0, 2.0 * pi], [0.0, pi]], args=(n, m)
                )
                real_integral = array(real_integral)
                imag_integral = array(imag_integral)
                if isnan(real_integral) or isnan(imag_integral):
                    print(f"Integration failed for l={n}, m={m}")

                coeff_mat[n, m + n] = real_integral + 1j * imag_integral

        return SphericalHarmonicsDistributionComplex(coeff_mat, transformation)

    # ------------------------------------------------------------------
    # pyshtools-based helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _coeff_mat_to_pysh(coeff_mat, degree):
        """Convert our coeff_mat to a pyshtools SHComplexCoeffs object."""
        import pyshtools as pysh  # pylint: disable=import-error

        clm = pysh.SHCoeffs.from_zeros(
            degree, kind="complex", normalization="ortho", csphase=-1
        )
        for n in range(degree + 1):
            for m in range(0, n + 1):
                clm.coeffs[0, n, m] = coeff_mat[n, n + m]
            for m in range(1, n + 1):
                clm.coeffs[1, n, m] = coeff_mat[n, n - m]
        return clm

    @staticmethod
    def _pysh_to_coeff_mat(clm, degree):
        """Convert a pyshtools SHComplexCoeffs object to our coeff_mat."""
        coeff_mat = zeros((degree + 1, 2 * degree + 1), dtype=complex128)
        max_n = min(clm.lmax, degree)
        for n in range(max_n + 1):
            for m in range(0, n + 1):
                coeff_mat[n, n + m] = clm.coeffs[0, n, m]
            for m in range(1, n + 1):
                coeff_mat[n, n - m] = clm.coeffs[1, n, m]
        return coeff_mat

    @staticmethod
    def _get_dh_grid_cartesian(degree):
        """Return (x, y, z) flat arrays and grid_shape for the DH grid at *degree*."""
        import pyshtools as pysh  # pylint: disable=import-error

        dummy = pysh.SHCoeffs.from_zeros(
            degree, kind="complex", normalization="ortho", csphase=-1
        )
        grid = dummy.expand(grid="DH", extend=False)
        lats, lons = grid.lats(), grid.lons()
        lon_mesh, lat_mesh = meshgrid(array(lons), array(lats))
        theta = deg2rad(90.0 - lat_mesh)  # colatitude in radians
        phi = deg2rad(lon_mesh)  # azimuth in radians
        x_c = sin(theta) * cos(phi)
        y_c = sin(theta) * sin(phi)
        z_c = cos(theta)
        return x_c.ravel(), y_c.ravel(), z_c.ravel(), theta.shape

    def _eval_on_grid(self, target_degree=None):
        """Evaluate this SHD on the DH grid at *target_degree* (defaults to own degree).

        The DH grid is expanded at *target_degree* so that higher-frequency grids
        can be used for intermediate computations (e.g. when squaring a degree-L
        function that has degree 2L).
        Returns a real 2-D numpy array of shape (nlat, nlon).
        """
        import pyshtools as pysh  # pylint: disable=import-error

        degree = self.coeff_mat.shape[0] - 1
        if target_degree is None:
            target_degree = degree

        # Pad our coefficients into a pysh object at target_degree
        clm_full = pysh.SHCoeffs.from_zeros(
            target_degree, kind="complex", normalization="ortho", csphase=-1
        )
        min_deg = min(degree, target_degree)
        clm_own = self._coeff_mat_to_pysh(self.coeff_mat, min_deg)
        clm_full.coeffs[0, : min_deg + 1, : min_deg + 1] = clm_own.coeffs[
            0, : min_deg + 1, : min_deg + 1
        ]
        clm_full.coeffs[1, : min_deg + 1, : min_deg + 1] = clm_own.coeffs[
            1, : min_deg + 1, : min_deg + 1
        ]
        grid = clm_full.expand(grid="DH", extend=False)
        return array(grid.data.real)

    @staticmethod
    def _fit_from_grid(grid_vals_real, degree, transformation):
        """Fit SH coefficients to real-valued grid values on a DH grid.

        *grid_vals_real* is a 2D array (numpy or backend tensor) with shape
        matching the DH grid for *degree*.  Returns a new
        :class:`SphericalHarmonicsDistributionComplex`.
        """
        import numpy as _np  # noqa: PLC0415
        import pyshtools as pysh  # pylint: disable=import-error

        grid_vals_np = _np.asarray(grid_vals_real)
        grid_obj = pysh.SHGrid.from_array(
            grid_vals_np.astype(complex), grid="DH"
        )
        clm = grid_obj.expand(lmax_calc=degree, normalization="ortho", csphase=-1)
        coeff_mat = SphericalHarmonicsDistributionComplex._pysh_to_coeff_mat(clm, degree)
        shd = SphericalHarmonicsDistributionComplex(coeff_mat, transformation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shd.normalize_in_place()
        return shd

    # ------------------------------------------------------------------
    # Public numerical methods
    # ------------------------------------------------------------------

    @staticmethod
    def from_distribution_numerical_fast(dist, degree, transformation="identity"):
        """Approximate *dist* by a degree-*degree* SHD using a DH grid.

        This is faster than :meth:`from_distribution_via_integral` because it
        uses the discrete spherical-harmonic transform instead of numerical
        quadrature.
        """
        if transformation not in ("identity", "sqrt"):
            raise ValueError(f"Unsupported transformation: '{transformation}'")

        x_c, y_c, z_c, grid_shape = (
            SphericalHarmonicsDistributionComplex._get_dh_grid_cartesian(degree)
        )
        xs = column_stack([x_c, y_c, z_c])
        fvals = array(dist.pdf(xs), dtype=float).reshape(grid_shape)

        if transformation == "sqrt":
            fvals = sqrt(maximum(fvals, 0.0))

        return SphericalHarmonicsDistributionComplex._fit_from_grid(
            fvals, degree, transformation
        )

    def convolve(self, other):  # pylint: disable=too-many-locals
        """Spherical convolution with a *zonal* distribution *other*.

        For the ``'identity'`` transformation the standard frequency-domain
        formula is used (exact for bandlimited functions).  For the ``'sqrt'``
        transformation a grid-based approach with a 2× finer intermediate grid
        is used so that squaring the sqrt functions introduces no aliasing.
        """
        assert isinstance(
            other, SphericalHarmonicsDistributionComplex
        ), "other must be a SphericalHarmonicsDistributionComplex"

        degree = self.coeff_mat.shape[0] - 1

        if self.transformation == "identity" and other.transformation == "identity":
            # Direct frequency-domain formula: h_{n,m} = sqrt(4π/(2n+1)) * f_{n,m} * g_{n,0}
            h_lm = zeros_like(self.coeff_mat)
            for n in range(degree + 1):
                factor = (
                    sqrt(4.0 * pi / (2 * n + 1))
                    * other.coeff_mat[n, n]
                )
                for m in range(-n, n + 1):
                    h_lm[n, n + m] = factor * self.coeff_mat[n, n + m]
            result = SphericalHarmonicsDistributionComplex(h_lm, "identity")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result.normalize_in_place()
            return result

        if self.transformation == "sqrt" and other.transformation == "sqrt":
            # Use a grid twice as fine to avoid aliasing when squaring.
            degree_fine = 2 * degree

            # Recover p = f^2 and q = g^2 on the fine grid, then SHT to degree.
            f_grid = self._eval_on_grid(target_degree=degree_fine)
            p_grid = f_grid**2

            g_grid = other._eval_on_grid(target_degree=degree_fine)  # pylint: disable=protected-access
            q_grid = g_grid**2

            import numpy as _np  # noqa: PLC0415
            import pyshtools as pysh  # pylint: disable=import-error

            def _grid_to_coeff(grid_vals):
                grid_vals_np = _np.asarray(grid_vals)
                grid_obj = pysh.SHGrid.from_array(
                    grid_vals_np.astype(complex), grid="DH"
                )
                clm = grid_obj.expand(
                    lmax_calc=degree, normalization="ortho", csphase=-1
                )
                return self._pysh_to_coeff_mat(clm, degree)

            p_lm = _grid_to_coeff(p_grid)
            q_lm = _grid_to_coeff(q_grid)

            # Convolution formula on the identity coefficients
            r_lm = zeros_like(p_lm)
            for n in range(degree + 1):
                factor = sqrt(4.0 * pi / (2 * n + 1)) * q_lm[n, n]
                for m in range(-n, n + 1):
                    r_lm[n, n + m] = factor * p_lm[n, n + m]

            # Evaluate r on the standard DH grid, take sqrt, refit
            r_shd_id = SphericalHarmonicsDistributionComplex(r_lm, "identity")
            r_grid = r_shd_id._eval_on_grid()  # pylint: disable=protected-access
            sqrt_r_grid = sqrt(maximum(r_grid, 0.0))

            return SphericalHarmonicsDistributionComplex._fit_from_grid(
                sqrt_r_grid, degree, "sqrt"
            )

        raise ValueError(
            "convolve: mixed transformations are not supported. "
            "Both self and other must use the same transformation."
        )

    def multiply(self, other):
        """Pointwise multiplication in physical space (Bayesian update step).

        Works for both ``'identity'`` and ``'sqrt'`` transformations.
        For ``'sqrt'``: ``sqrt(p) * sqrt(q) = sqrt(p*q)`` which is the correct
        sqrt-transformed product density.
        """
        assert isinstance(
            other, SphericalHarmonicsDistributionComplex
        ), "other must be a SphericalHarmonicsDistributionComplex"
        assert self.transformation == other.transformation, (
            "multiply: both distributions must use the same transformation"
        )

        degree = self.coeff_mat.shape[0] - 1

        f_grid = self._eval_on_grid()
        g_grid = other._eval_on_grid()  # pylint: disable=protected-access

        h_grid = f_grid * g_grid

        return SphericalHarmonicsDistributionComplex._fit_from_grid(
            h_grid, degree, self.transformation
        )

    def rotate(self, alpha, beta, gamma):
        """Rotate the distribution by ZYZ Euler angles (in radians).

        Parameters
        ----------
        alpha : float
            First rotation angle around Z (radians).
        beta : float
            Second rotation angle around Y (radians).
        gamma : float
            Third rotation angle around Z (radians).
        """
        degree = self.coeff_mat.shape[0] - 1
        clm = self._coeff_mat_to_pysh(self.coeff_mat, degree)
        clm_rot = clm.rotate(
            alpha * 180.0 / pi,
            beta * 180.0 / pi,
            gamma * 180.0 / pi,
            degrees=True,
            body=True,
        )
        coeff_mat_rot = self._pysh_to_coeff_mat(clm_rot, degree)
        return SphericalHarmonicsDistributionComplex(
            coeff_mat_rot, self.transformation
        )
