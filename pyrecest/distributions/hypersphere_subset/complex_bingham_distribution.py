# pylint: disable=no-name-in-module,no-member,redefined-builtin
"""Complex Bingham Distribution.

Ported from the MATLAB libDirectional library:
  ComplexBinghamDistribution.m  (libDirectional/lib/distributions/complexHypersphere/)

Reference:
  Kent, J. T. "The Complex Bingham Distribution and Shape Analysis."
  Journal of the Royal Statistical Society. Series B (Methodological), 1994, 285-299.
"""

from copy import copy

import pyrecest.backend
from pyrecest.backend import (
    abs,
    all,
    allclose,
    arange,
    array,
    asarray,
    complex128,
    concatenate,
    conj,
    cumsum,
    diag,
    diff,
    einsum,
    empty,
    exp,
    flip,
    linalg,
    linspace,
    log,
    max,
    minimum,
    pi,
    prod,
    random,
    real,
    sign,
    sort,
    sqrt,
    sum,
    zeros,
)
from scipy.optimize import least_squares

from .abstract_complex_hyperspherical_distribution import (
    AbstractComplexHypersphericalDistribution,
)


class ComplexBinghamDistribution(AbstractComplexHypersphericalDistribution):
    """Complex Bingham distribution on the complex unit hypersphere.

    The distribution is defined on the complex unit sphere
    S^{2d-1} = {z in C^d : ||z|| = 1} with pdf

        p(z) proportional to exp(z^H B z),

    where B is a d x d Hermitian parameter matrix.

    Attributes
    ----------
    B : array, shape (d, d), dtype complex128
        Hermitian parameter matrix.
    log_norm_const : float
        Negative log normalization constant (i.e. -log C(B) where C(B) is
        the normalising constant), stored so that
        pdf(z) = exp(log_norm_const + Re(z^H B z)).
    """

    def __init__(self, B):
        """Construct a ComplexBinghamDistribution.

        Parameters
        ----------
        B : array_like, shape (d, d)
            Hermitian parameter matrix (B == B^H must hold).
        """
        assert (
            pyrecest.backend.__backend_name__ != "jax"  # pylint: disable=no-member
        ), "ComplexBinghamDistribution is not supported on the JAX backend"
        B = asarray(B, dtype=complex128)
        assert allclose(B, conj(B).T, atol=1e-10), "B must be Hermitian"
        self.B = B
        super().__init__(B.shape[0])
        self.log_norm_const = ComplexBinghamDistribution.log_norm(B)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def pdf(self, xs):
        """Evaluate the pdf at one or more points on the complex unit sphere.

        Parameters
        ----------
        xs : array_like, shape (d,) or (d, n)
            Each column (or the single vector) is a point on the complex unit
            sphere.

        Returns
        -------
        array, shape (n,) or float
            Pdf value(s).
        """
        xs = asarray(xs, dtype=complex128)
        single = xs.ndim == 1
        if single:
            xs = xs[:, None]
        # Re(xs^H B xs) for each column
        Bxs = self.B @ xs  # (d, n)
        vals = real(einsum("ij,ij->j", conj(xs), Bxs))  # shape (n,)
        p = exp(self.log_norm_const + vals)
        return float(p[0]) if single else p

    def sample(self, n):
        """Draw samples from the complex Bingham distribution.

        Uses the rejection-sampling algorithm of
        Kent, Constable & Er (2004) "Simulation for the complex Bingham
        distribution", Statistics and Computing, 14, 53-57.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        array, shape (d, n), dtype complex128
            Sampled unit vectors (each column has unit norm).
        """
        d = self.complex_dim
        if d < 2:
            raise ValueError("Sampling requires d >= 2.")

        # Eigendecomposition of -B (eigenvectors V, eigenvalues Lambda of -B)
        eigenvalues_neg, V = linalg.eigh(-self.B)  # sorted ascending
        # Sort descending (eigh gives ascending order, flip to get descending)
        eigenvalues_neg = flip(eigenvalues_neg, axis=0)
        V = flip(V, axis=1)

        # Shift so the last (smallest) eigenvalue of -B becomes 0
        Lambda = eigenvalues_neg - eigenvalues_neg[-1]  # Lambda >= 0, Lambda[-1] = 0

        # Precompute for truncated-exponential CDF inversion
        Lam = Lambda[:-1]  # shape (d-1,)

        samples = zeros((d, n), dtype=complex128)
        for k in range(n):
            # Rejection loop
            while True:
                S = zeros(d)
                U = random.uniform(size=(int(d - 1),))
                for i in range(d - 1):
                    if Lambda[i] < 0.03:
                        # Nearly-uniform truncated exponential
                        S[i] = U[i]
                    else:
                        S[i] = -(1.0 / Lam[i]) * log(1.0 - U[i] * (1.0 - exp(-Lam[i])))
                if sum(S[:-1]) < 1.0:
                    break
            S[-1] = 1.0 - sum(S[:-1])

            # Random phases
            theta = 2.0 * pi * random.uniform(size=(int(d),))
            weighted_phases = sqrt(S) * exp(1j * theta)
            samples[:, k] = V @ weighted_phases

        return samples

    # ------------------------------------------------------------------
    # Static / class methods
    # ------------------------------------------------------------------

    @staticmethod
    def log_norm(B):
        """Compute the *negative* log normalization constant -log C(B).

        The formula used is (Kent 1994, eq. 2.3) after eigenvalue shift:

            C(lambda) = 2*pi^d * sum_j exp(lambda_j) / prod_{k!=j}(lambda_j - lambda_k)

        For nearly-equal eigenvalues a small perturbation is applied so that
        the Vandermonde denominators remain well-conditioned, matching the
        approach in the original MATLAB implementation.

        Parameters
        ----------
        B : array_like, shape (d, d)
            Hermitian parameter matrix.

        Returns
        -------
        float
            Negative log normalization constant.
        """
        assert (
            pyrecest.backend.__backend_name__ != "jax"  # pylint: disable=no-member
        ), "ComplexBinghamDistribution.log_norm is not supported on the JAX backend"
        B = asarray(B, dtype=complex128)
        d = B.shape[0]

        # Real eigenvalues of a Hermitian matrix
        eigenvalues = linalg.eigvalsh(B)  # sorted ascending

        # Shift so the maximum eigenvalue is 0
        eigenvalue_shift = float(eigenvalues[-1])
        eigenvalues = eigenvalues - eigenvalue_shift

        # Perturb near-equal eigenvalues for numerical stability
        eigenvalues = ComplexBinghamDistribution._perturb_eigenvalues(eigenvalues)

        if all(abs(eigenvalues) < 1e-3):
            # All eigenvalues near zero: use limiting uniform value C = 2*pi^d / (d-1)!
            log_C_shifted = log(2.0) + d * log(pi) - float(sum(log(arange(1, d))))
        else:
            # Analytical formula: C = 2*pi^d * sum_j exp(lambda_j) / prod_{k!=j}(lambda_j - lambda_k)
            log_C_shifted = ComplexBinghamDistribution._log_norm_from_eigenvalues(
                eigenvalues
            )

        # Apply eigenvalue-shift correction: C(lambda) = exp(shift) * C(lambda - shift)
        log_C = log_C_shifted + eigenvalue_shift

        return -log_C  # inverted convention: log_norm_const = -log C

    @staticmethod
    def _log_norm_from_eigenvalues(eigenvalues):
        """Log normalization for shifted eigenvalues (max = 0) via partial fractions.

        Computes log(2*pi^d * sum_j exp(lambda_j) / prod_{k!=j}(lambda_j - lambda_k)).
        """
        d = len(eigenvalues)
        log_prefix = log(2.0) + d * log(pi)

        # For each j compute sign_j * exp(log_term_j) where
        #   log_term_j = lambda_j - sum_{k!=j} log|lambda_j - lambda_k|
        # diff_matrix[j, k] = eigenvalues[j] - eigenvalues[k] for j != k
        diff_matrix = eigenvalues[:, None] - eigenvalues[None, :]
        log_terms = empty(d)
        signs = empty(d)
        for j in range(d):
            mask = arange(d) != j
            diffs = diff_matrix[j, mask]
            signs[j] = prod(sign(diffs))
            log_terms[j] = eigenvalues[j] - sum(log(abs(diffs)))

        # log(|sum_j sign_j * exp(log_term_j)|) via log-sum-exp
        max_log = max(log_terms)
        scaled = sum(signs * exp(log_terms - max_log))
        log_sum = max_log + log(abs(scaled))

        return log_prefix + log_sum

    @staticmethod
    def _perturb_eigenvalues(eigenvalues):
        """Sort eigenvalues descending and enforce minimum spacing of 0.01.

        Mirrors MATLAB's makeSureEigenvaluesAreNotTooClose.
        """
        lam = flip(sort(eigenvalues), axis=0)
        diffs = minimum(diff(lam), -0.01)  # enforce gap >= 0.01
        return concatenate([lam[:1], lam[0:1] + cumsum(diffs)])

    @classmethod
    def fit(cls, Z):
        """Maximum-likelihood fit of a complex Bingham distribution to data.

        Parameters
        ----------
        Z : array_like, shape (d, n)
            Complex unit vectors (columns); it is assumed that ||z_i|| = 1.

        Returns
        -------
        ComplexBinghamDistribution
        """
        Z = asarray(Z, dtype=complex128)
        n = Z.shape[1]
        S = Z @ conj(Z).T / n  # sample scatter matrix
        B = cls._estimate_parameter_matrix(S)
        return cls(B)

    @staticmethod
    def _estimate_parameter_matrix(S):
        """Compute the ML estimate of B from the scatter matrix S.

        The eigenvectors of B equal those of S.  The eigenvalues of B are
        found by solving the moment-matching equations

            d log C(lambda) / d lambda_k  =  s_k,  k = 1, ..., d

        where s_k are the eigenvalues of S, using a least-squares solver
        with finite-difference gradients.

        Parameters
        ----------
        S : array, shape (d, d), complex Hermitian
            Sample scatter matrix (E[z z^H] estimate).

        Returns
        -------
        array, shape (d, d), complex Hermitian
            Estimated parameter matrix B.
        """
        d = S.shape[0]
        eigenvalues_S, V = linalg.eigh(S)  # ascending

        def grad_log_c(lam):
            """Numerical gradient of log C via forward finite differences."""
            eps = 1e-6
            B_diag = diag(array(lam, dtype=complex128))
            log_c0 = ComplexBinghamDistribution.log_norm(B_diag)
            grad = empty(d)
            for i in range(d):
                lam_p = copy(lam)
                lam_p[i] += eps
                log_cp = ComplexBinghamDistribution.log_norm(
                    diag(array(lam_p, dtype=complex128))
                )
                # log_norm_const = -log C, so d(log C)/dlambda_i = -d(log_norm_const)/dlambda_i
                grad[i] = (-log_cp - (-log_c0)) / eps
            return grad

        # Initial guess: spread eigenvalues below zero
        initial_eigenvalues = linspace(-(d - 1) * 10, -10, int(d - 1))

        def residuals(x):
            lam = concatenate([x, array([0.0])])
            return grad_log_c(lam) - eigenvalues_S

        result = least_squares(
            residuals,
            initial_eigenvalues,
            method="lm",
            ftol=1e-15,
            xtol=1e-10,
            max_nfev=int(1e4),
        )
        lam_B = concatenate([result.x, array([0.0])])
        B = V @ diag(array(lam_B, dtype=complex128)) @ conj(V).T
        B = 0.5 * (B + conj(B).T)  # enforce exact Hermitian symmetry
        return B

    @staticmethod
    def cauchy_schwarz_divergence(cB1, cB2):
        """Cauchy-Schwarz divergence between two complex Bingham distributions.

        D_CS(p, q) = half * [log C(2*B1) + log C(2*B2)] - log C(B1+B2) >= 0

        Using the stored negative log normalization (log_norm = -log C):

            D_CS = log_norm(B1+B2) - half * [log_norm(2*B1) + log_norm(2*B2)]

        This matches the MATLAB libDirectional CauchySchwarzDivergence convention.

        Parameters
        ----------
        cB1, cB2 : ComplexBinghamDistribution or array_like
            Distributions or Hermitian parameter matrices.

        Returns
        -------
        float
            Non-negative divergence value.
        """
        assert (
            pyrecest.backend.__backend_name__ != "jax"  # pylint: disable=no-member
        ), "ComplexBinghamDistribution.cauchy_schwarz_divergence is not supported on the JAX backend"
        if isinstance(cB1, ComplexBinghamDistribution):
            B1 = cB1.B
        else:
            B1 = asarray(cB1, dtype=complex128)
        if isinstance(cB2, ComplexBinghamDistribution):
            B2 = cB2.B
        else:
            B2 = asarray(cB2, dtype=complex128)

        assert allclose(B1, conj(B1).T, atol=1e-10), "B1 must be Hermitian"
        assert allclose(B2, conj(B2).T, atol=1e-10), "B2 must be Hermitian"

        log_c1 = ComplexBinghamDistribution.log_norm(2.0 * B1)
        log_c2 = ComplexBinghamDistribution.log_norm(2.0 * B2)
        log_c3 = ComplexBinghamDistribution.log_norm(B1 + B2)

        return log_c3 - 0.5 * (log_c1 + log_c2)
