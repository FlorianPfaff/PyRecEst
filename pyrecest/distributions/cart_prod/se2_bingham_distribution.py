# pylint: disable=redefined-builtin,no-name-in-module,no-member
import numpy as np
from pyrecest.backend import (
    all,
    allclose,
    argmax,
    argsort,
    array,
    column_stack,
    concatenate,
    diag,
    exp,
    linalg,
    ones,
    pi,
    random,
    sort,
    sqrt,
    sum,
)
from scipy.integrate import quad
from scipy.special import iv

from ..abstract_se2_distribution import AbstractSE2Distribution
from ..hypersphere_subset.bingham_distribution import BinghamDistribution
from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution


class SE2BinghamDistribution(AbstractSE2Distribution):
    """
    Distribution on SE(2) = S^1 x R^2.

    The density is f(x) = (1/NC) * exp(x^T * C * x) where x is the dual
    quaternion representation (first two components on S^1, last two
    components in R^2).

    C is a 4x4 symmetric matrix partitioned as::

        C = [ C1   C2^T ]
            [ C2   C3   ]

    where:
      - C1 (2x2): symmetric, controls the Bingham (rotational) part
      - C2 (2x2): coupling between rotation and translation
      - C3 (2x2): symmetric, negative-definite, controls the Gaussian (translational) part

    Reference:
    Igor Gilitschenski, Gerhard Kurz, Simon J. Julier, Uwe D. Hanebeck,
    "A New Probability Distribution for Simultaneous Representation of
    Uncertain Position and Orientation",
    Proceedings of the 17th International Conference on Information Fusion
    (Fusion 2014), Salamanca, Spain, July 2014.
    """

    def __init__(self, C, C2=None, C3=None):
        """
        Create an SE2BinghamDistribution.

        Parameters
        ----------
        C : array_like, shape (4, 4) or (2, 2)
            If C2 and C3 are not provided, this is the full 4x4 parameter
            matrix.  Otherwise it is the 2x2 Bingham (rotational) part C1.
        C2 : array_like, shape (2, 2), optional
            Coupling matrix between rotation and translation.
        C3 : array_like, shape (2, 2), optional
            Symmetric negative-definite matrix for the translational part.
        """
        AbstractSE2Distribution.__init__(self)

        assert (C2 is None) == (
            C3 is None
        ), "Either both C2 and C3 must be provided, or neither."

        if C2 is None:
            assert C.shape == (4, 4), "C must be 4x4 when C2 and C3 are not provided."
            assert allclose(C, C.T, atol=1e-6), "Full C matrix must be symmetric."
            self.C = C
            self.C1 = C[:2, :2]
            self.C2 = C[2:, :2]
            self.C3 = C[2:, 2:]
        else:
            assert C.shape == (2, 2), "C1 must be 2x2."
            assert C2.shape == (2, 2), "C2 must be 2x2."
            assert C3.shape == (2, 2), "C3 must be 2x2."
            assert allclose(C, C.T, atol=1e-6), "C1 must be symmetric."
            assert allclose(C3, C3.T, atol=1e-6), "C3 must be symmetric."
            self.C1 = C
            self.C2 = C2
            self.C3 = C3
            self.C = column_stack(
                [
                    column_stack([self.C1, self.C2.T]).T,
                    column_stack([self.C2, self.C3]).T,
                ]
            ).T

        assert all(linalg.eigvalsh(self.C3) <= 0), "C3 must be negative semi-definite."

        self._nc = None  # lazily computed

    @property
    def nc(self):
        """Normalization constant (lazily computed)."""
        if self._nc is None:
            self._nc = self._compute_nc()
        return self._nc

    def _compute_nc(self):
        """
        Compute the normalization constant.

        NC = 2*pi * sqrt(det(-0.5 * C3^{-1})) * F_bingham(Z_bm)

        where Z_bm are the eigenvalues of the Schur complement
        BM = C1 - C2^T * C3^{-1} * C2,
        and F_bingham is the 2D Bingham normalization constant
        F = 2*pi * exp((z1+z2)/2) * I_0((z2-z1)/2).
        """
        C1 = array(self.C1, dtype=float)
        C2 = array(self.C2, dtype=float)
        C3 = array(self.C3, dtype=float)
        C3_inv = linalg.inv(C3)
        bm = C1 - C2.T @ C3_inv @ C2
        z = sort(linalg.eigvalsh(bm))  # ascending
        # 2D Bingham normalization on S^1
        b_nc = 2.0 * pi * exp((z[0] + z[1]) / 2.0) * iv(0, float((z[1] - z[0]) / 2.0))
        nc = 2.0 * pi * sqrt(linalg.det(-0.5 * C3_inv)) * b_nc
        return float(nc)

    def pdf(self, xs):
        """
        Evaluate the probability density at the given points.

        Parameters
        ----------
        xs : array_like, shape (N, 4) or (N, 3)
            Evaluation points in dual quaternion (N x 4) or angle-pos
            (N x 3) representation.

        Returns
        -------
        p : array, shape (N,)
            Density values.
        """
        xs = array(xs)
        if xs.ndim == 1:
            xs = xs.reshape(1, -1)
        if xs.shape[1] == 3:
            xs = AbstractSE2Distribution.angle_pos_to_dual_quaternion(xs)
        assert xs.shape[1] == 4, "Input must have 4 columns (dual quaternion)."
        return (1.0 / self.nc) * exp(sum(xs * (xs @ self.C.T), axis=1))

    def mode(self):
        """
        Compute one mode of the distribution.

        Because of antipodal symmetry, -mode is equally valid.

        Returns
        -------
        m : array, shape (4,)
            Mode in dual quaternion representation.
        """
        C1 = array(self.C1, dtype=float)
        C2 = array(self.C2, dtype=float)
        C3 = array(self.C3, dtype=float)
        C3_inv = linalg.inv(C3)
        bingham_c = C1 - C2.T @ C3_inv @ C2
        eigenvalues, eigenvectors = linalg.eigh(bingham_c)
        idx = int(argmax(eigenvalues))
        m_rot = eigenvectors[:, idx]
        m_lin = -C3_inv @ C2 @ m_rot
        return array(concatenate([m_rot, m_lin]))

    def sample(self, n):
        """
        Draw n samples from the distribution.

        Sampling uses a two-step procedure:
        1. Sample the rotational part from the Bingham marginal.
        2. Sample the translational part from the Gaussian conditional.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        s : array, shape (n, 4)
            Samples in dual quaternion representation.
        """
        assert n > 0, "n must be positive."
        C3_inv = linalg.inv(array(self.C3, dtype=float))

        # Step 1: sample Bingham marginal via Schur complement eigendecomp
        bingham_c = array(self.C1, dtype=float) - array(
            self.C2, dtype=float
        ).T @ C3_inv @ array(self.C2, dtype=float)
        eigenvalues, eigenvectors = linalg.eigh(bingham_c)
        order = argsort(eigenvalues)  # ascending
        eigenvalues = eigenvalues[order]
        b = BinghamDistribution(
            array(eigenvalues - eigenvalues[-1]), array(eigenvectors[:, order])
        )
        bingham_samples = b.sample(n)  # (n, 2)

        # Step 2: sample Gaussian conditional
        # mean_i = -C3^{-1} * C2 * x_rot_i
        cov = -0.5 * C3_inv
        means = (-C3_inv @ array(self.C2, dtype=float) @ array(bingham_samples).T).T
        # Sample n zero-mean perturbations from N(0, cov) and add respective means
        noise = random.multivariate_normal(array([0.0, 0.0]), cov, (n,))
        lin_samples = means + noise

        return column_stack([bingham_samples, lin_samples])

    def marginalize_linear(self):
        """
        Return the marginal distribution over the periodic (rotational) part.

        The marginal is the Bingham distribution corresponding to the Schur
        complement BM = C1 - C2^T * C3^{-1} * C2.

        Returns
        -------
        b : BinghamDistribution
            Marginal Bingham distribution on S^1.
        """
        C1 = array(self.C1, dtype=float)
        C2 = array(self.C2, dtype=float)
        C3 = array(self.C3, dtype=float)
        C3_inv = linalg.inv(C3)
        bm = C1 - C2.T @ C3_inv @ C2
        eigenvalues, eigenvectors = linalg.eigh(bm)
        order = argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        z = eigenvalues - eigenvalues[-1]
        m = eigenvectors
        return BinghamDistribution(z, m)

    def marginalize_periodic(self):
        """
        Return the marginal distribution over the linear (translational) part.

        The marginal is computed by numerically integrating out the rotational
        component.

        Returns
        -------
        dist : CustomLinearDistribution
            Marginal distribution over R^2.
        """
        from pyrecest.backend import to_numpy  # pylint: disable=import-outside-toplevel

        C_mat = to_numpy(array(self.C, dtype=float))
        nc = self.nc

        def _marginal_pdf(xs):
            xs = np.atleast_2d(xs)
            out = np.empty(xs.shape[0])
            for i, x_lin in enumerate(xs):
                # Integrate exp(x^T C x) over S^1 using the angle parametrisation
                def integrand(theta, xl=x_lin):
                    x_rot = np.array([np.cos(theta), np.sin(theta)])
                    x = np.concatenate([x_rot, xl])
                    return np.exp(float(x @ C_mat @ x))

                val, _ = quad(integrand, 0.0, 2.0 * np.pi)
                out[i] = val / nc
            return array(out)

        return CustomLinearDistribution(_marginal_pdf, self.lin_dim)

    @staticmethod
    def fit(samples, weights=None):
        """
        Estimate SE2BinghamDistribution parameters from samples.

        Parameters
        ----------
        samples : array_like, shape (N, 4) or (N, 3)
            Samples in dual quaternion (N x 4) or angle-pos (N x 3) form.
        weights : array_like, shape (N,), optional
            Non-negative weights (need not sum to 1).  Defaults to uniform.

        Returns
        -------
        dist : SE2BinghamDistribution
            Fitted distribution.
        """
        samples = array(samples, dtype=float)
        if samples.shape[1] == 3:
            samples = AbstractSE2Distribution.angle_pos_to_dual_quaternion(samples)
        assert samples.shape[1] == 4

        n = samples.shape[0]
        if weights is None:
            weights = ones(n) / n
        else:
            weights = array(weights, dtype=float)
            weights = weights / sum(weights)

        w = weights.reshape(-1, 1)
        s_rot, s_lin = samples[:, :2], samples[:, 2:]

        # Bingham block: estimate Schur complement from weighted scatter
        schur_c = SE2BinghamDistribution._schur_from_scatter(s_rot, w)

        # Gaussian block: estimate C2 and C3 via weighted regression
        c2_est, c3_est = SE2BinghamDistribution._fit_gaussian_block(s_rot, s_lin, w)

        # Recover C1 from Schur complement definition
        c1_est = schur_c + c2_est.T @ linalg.inv(c3_est) @ c2_est
        c1_est = 0.5 * (c1_est + c1_est.T)

        return SE2BinghamDistribution(c1_est, c2_est, c3_est)

    @staticmethod
    def _schur_from_scatter(s_rot, w):
        """Return the estimated Schur complement C1 - C2' C3^{-1} C2 from samples."""
        scatter = (s_rot * w).T @ s_rot
        eigenvalues, eigenvectors = linalg.eigh(scatter)
        order = argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        z = eigenvalues - eigenvalues[-1]
        return eigenvectors @ diag(z) @ eigenvectors.T

    @staticmethod
    def _fit_gaussian_block(s_rot, s_lin, w):
        """Return estimated (C2, C3) via weighted linear regression."""
        reg_a = (s_rot * w).T @ s_rot
        # Use pinv for numerical stability when reg_a is nearly singular
        reg_beta = (s_lin * w).T @ s_rot @ linalg.pinv(reg_a)
        residuals = s_lin - s_rot @ reg_beta.T
        reg_cov = (residuals * w).T @ residuals
        # Use pinv: reg_cov may be ill-conditioned when samples cluster on a subspace
        c3_est = linalg.pinv(-2.0 * reg_cov)
        c3_est = 0.5 * (c3_est + c3_est.T)
        return -c3_est @ reg_beta, c3_est
