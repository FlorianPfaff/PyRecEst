import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    arange,
    array,
    exp,
    log,
    ndim,
    pi,
    real,
    sqrt,
    zeros,
)

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from .hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


class ToroidalFourierDistribution(
    HypertoroidalFourierDistribution, AbstractToroidalDistribution
):
    """
    Distribution on the torus [0, 2π)^2 represented by a 2-D Fourier series.

    Inherits all Fourier-based methods from HypertoroidalFourierDistribution and
    all toroidal-specific methods (e.g. mean_4D, covariance_4D_numerical,
    circular_correlation_jammalamadaka_numerical) from AbstractToroidalDistribution.

    This port follows:
        Florian Pfaff, Gerhard Kurz, Uwe D. Hanebeck,
        "Multivariate Angular Filtering Using Fourier Series",
        Journal of Advances in Information Fusion, 11(2):206-226, Dec 2016.
    """

    def __init__(self, coeff_mat, transformation: str = "sqrt"):
        cm = array(coeff_mat) if not hasattr(coeff_mat, "shape") else coeff_mat
        if ndim(cm) < 2 or (ndim(cm) >= 2 and any(s == 1 for s in cm.shape)):
            raise ValueError(
                "ToroidalFourierDistribution:VectorGiven: "
                "ToroidalFourierDistributions require a 2-D coefficient matrix "
                "with at least 3 entries in each dimension."
            )

        HypertoroidalFourierDistribution.__init__(self, coeff_mat, transformation)
        AbstractToroidalDistribution.__init__(self)
        assert self.dim == 2, (
            "ToroidalFourierDistribution requires a 2-D coefficient matrix, "
            f"but got dim={self.dim}."
        )

    # ------------------------------------------------------------------
    # Analytical integral with optional bounds
    # ------------------------------------------------------------------
    def integrate(self, integration_boundaries=None):  # pylint: disable=too-many-locals
        """
        Compute the integral of the pdf over the given rectangular region.

        Parameters
        ----------
        integration_boundaries : array-like, shape (2, 2), optional
            [[l1, r1], [l2, r2]]. Defaults to [[0, 2π], [0, 2π]].

        Returns
        -------
        float
            Value of the integral.
        """
        if integration_boundaries is None:
            lo = zeros(2)
            r = 2 * pi * array([1.0, 1.0])
        else:
            bounds = array(integration_boundaries)
            lo = bounds[:, 0]
            r = bounds[:, 1]

        # Work in identity representation
        if self.transformation == "sqrt":
            # Square the sqrt-representation to get the identity one
            target = tuple(2 * s - 1 for s in self.coeff_mat.shape)
            tfd = self.transform_via_coefficients("square", target)
        elif self.transformation == "identity":
            tfd = self
        else:
            raise ValueError(
                "integrate: transformation not supported for analytical integration."
            )

        C = tfd.coeff_mat
        maxk = [(s - 1) // 2 for s in C.shape]

        # Build per-dimension factor vectors
        #   For k ≠ 0: factor = -i/k * (exp(i*k*r) - exp(i*k*l))
        #   For k  = 0: factor = r - l
        factor_vecs = []
        for d in range(2):
            mk = int(maxk[d])
            k_range = arange(-mk, mk + 1, dtype=float)
            fv = zeros(2 * mk + 1, dtype=complex)
            for idx, kval in enumerate(k_range):
                ki = float(kval)
                if ki == 0.0:
                    fv[idx] = float(r[d]) - float(lo[d])
                else:
                    fv[idx] = (
                        -1j
                        / ki
                        * (exp(1j * ki * float(r[d])) - exp(1j * ki * float(lo[d])))
                    )
            factor_vecs.append(fv)

        # result = real(sum(C * outer(fv0, fv1)))
        fv0 = factor_vecs[0].reshape(-1, 1)
        fv1 = factor_vecs[1].reshape(1, -1)
        result = real((C * (fv0 * fv1)).sum())
        return float(result)

    # ------------------------------------------------------------------
    # 4-D covariance
    # ------------------------------------------------------------------
    def covariance_4d(self):
        """
        Compute the 4×4 covariance of [cos x1, sin x1, cos x2, sin x2].

        Delegates to the numerical implementation from AbstractToroidalDistribution.
        """
        return self.covariance_4D_numerical()

    # ------------------------------------------------------------------
    # Circular correlation (analytical override)
    # ------------------------------------------------------------------
    def circular_correlation_jammalamadaka(self) -> float:
        """
        Jammalamadaka–SenGupta circular correlation, computed analytically
        from the Fourier coefficients.
        """
        # Obtain identity-representation coefficients
        if self.transformation == "sqrt":
            tfd = self.transform_via_coefficients(
                "square", tuple(2 * s - 1 for s in self.coeff_mat.shape)
            )
        elif self.transformation == "identity":
            tfd = self
        else:
            raise ValueError(
                "circular_correlation_jammalamadaka: unsupported transformation."
            )

        m = tfd.mean_direction()  # [θ1, θ2] circular mean

        # Truncate/expand to 5×5 to have indices ±1 and ±2 available
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tfd5 = tfd.truncate((5, 5))

        C = tfd5.coeff_mat
        # Centre indices (0-based) for 5×5 matrix: j=k=2
        j, k = 2, 2

        # E[sin(x1 - μ1) * sin(x2 - μ2)]
        e_sin_a_sin_b = (
            -(C[j + 1, k + 1]) * pi**2 * exp(1j * (m[0] + m[1]))
            - (C[j - 1, k - 1]) * pi**2 * exp(-1j * (m[0] + m[1]))
            + (C[j + 1, k - 1]) * pi**2 * exp(1j * (m[0] - m[1]))
            + (C[j - 1, k + 1]) * pi**2 * exp(-1j * (m[0] - m[1]))
        )

        # E[sin²(x1 - μ1)]
        e_sin_a_sq = (
            0.5
            - C[j + 2, k] * exp(2j * m[0]) * pi**2
            - C[j - 2, k] * exp(-2j * m[0]) * pi**2
        )

        # E[sin²(x2 - μ2)]
        e_sin_b_sq = (
            0.5
            - C[j, k + 2] * exp(2j * m[1]) * pi**2
            - C[j, k - 2] * exp(-2j * m[1]) * pi**2
        )

        rhoc = real(e_sin_a_sin_b) / sqrt(real(e_sin_a_sq) * real(e_sin_b_sq))
        return float(rhoc)

    # ------------------------------------------------------------------
    # Convert to ToroidalWrappedNormalDistribution
    # ------------------------------------------------------------------
    def to_twn(self) -> "HypertoroidalWrappedNormalDistribution":
        """
        Approximate this distribution with a Toroidal Wrapped Normal.

        Uses the moment-matching formula via Fourier coefficients.

        Returns
        -------
        ToroidalWrappedNormalDistribution
            The best-fitting TWN.
        """
        # Work in identity representation
        if self.transformation == "identity":
            tfd = self
        elif self.transformation == "sqrt":
            tfd = self.transform_via_coefficients("square")
        else:
            raise ValueError("to_twn: unsupported transformation.")

        mu = tfd.mean_direction()  # circular mean [θ1, θ2]

        # Shift to zero mean, then truncate to 3×3
        tfd_zero = tfd.shift(-mu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tfd3 = tfd_zero.truncate((3, 3))

        C = tfd3.coeff_mat
        # In 3×3 (0-indexed), centre is at (1,1) → C_{0,0}
        # C[2,1] = C_{1, 0},  C[1,2] = C_{0, 1}
        # C[0,2] = C_{-1, 1}, C[2,2] = C_{1, 1}
        dim = tfd3.dim  # == 2
        cov_11 = real(-2 * dim * log(2 * pi) - 2 * log(C[2, 1]))
        cov_22 = real(-2 * dim * log(2 * pi) - 2 * log(C[1, 2]))
        cov_12 = real(0.5 * log(C[0, 2]) - 0.5 * log(C[2, 2]))

        cov = array([[float(cov_11), float(cov_12)], [float(cov_12), float(cov_22)]])

        from .toroidal_wrapped_normal_distribution import (
            ToroidalWrappedNormalDistribution,
        )

        return ToroidalWrappedNormalDistribution(mu, cov)

    # ------------------------------------------------------------------
    # Class-method factory overrides (ensure ToroidalFourierDistribution is returned)
    # ------------------------------------------------------------------
    @classmethod
    def from_function(cls, fun, n_coefficients, desired_transformation="sqrt"):
        """
        Construct a ToroidalFourierDistribution from a function on the torus.

        Parameters
        ----------
        fun : callable
            Function of two arrays (x1_grid, x2_grid) returning pdf values.
        n_coefficients : int or tuple[int, int]
            Number of Fourier coefficients per dimension (must be odd ≥ 3).
        desired_transformation : str
            'sqrt', 'identity', or 'log'.
        """
        if isinstance(n_coefficients, int):
            n_coefficients = (n_coefficients, n_coefficients)
        elif len(n_coefficients) == 1:
            n_coefficients = (int(n_coefficients[0]), int(n_coefficients[0]))
        n_coefficients = tuple(int(n) for n in n_coefficients)
        hfd = HypertoroidalFourierDistribution.from_function(
            fun, n_coefficients, desired_transformation
        )
        return cls(hfd.coeff_mat, hfd.transformation)

    @classmethod
    def from_function_values(
        cls,
        fvals,
        n_coefficients=None,
        desired_transformation: str = "sqrt",
        already_transformed: bool = False,
    ) -> "ToroidalFourierDistribution":
        """
        Construct a ToroidalFourierDistribution from function values on a grid.

        Parameters
        ----------
        fvals : 2-D array
            Function values on a regular grid over [0, 2π)^2.
        n_coefficients : tuple[int, int] or None
            If None, inferred from fvals shape.
        desired_transformation : str
        already_transformed : bool
        """
        if n_coefficients is None:
            n_coefficients = fvals.shape
        n_coefficients = tuple(int(n) for n in n_coefficients)
        return super().from_function_values(
            fvals,
            n_coefficients=n_coefficients,
            desired_transformation=desired_transformation,
            already_transformed=already_transformed,
        )

    @classmethod
    def from_distribution(
        cls,
        distribution,
        n_coefficients,
        desired_transformation: str = "sqrt",
    ) -> "ToroidalFourierDistribution":
        """
        Approximate a given toroidal distribution with a ToroidalFourierDistribution.

        Parameters
        ----------
        distribution : AbstractHypertoroidalDistribution
            Distribution to approximate (must have dim == 2).
        n_coefficients : int or tuple[int, int]
            Number of Fourier coefficients per dimension.
        desired_transformation : str
        """
        if isinstance(n_coefficients, int):
            n_coefficients = (n_coefficients, n_coefficients)
        elif len(n_coefficients) == 1:
            n_coefficients = (int(n_coefficients[0]), int(n_coefficients[0]))
        n_coefficients = tuple(int(n) for n in n_coefficients)
        hfd = super().from_distribution(
            distribution, n_coefficients, desired_transformation
        )
        return cls(hfd.coeff_mat, hfd.transformation)
