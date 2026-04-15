# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    argsort,
    array,
    concatenate,
    diag,
    exp,
    eye,
    linalg,
    max,
    maximum,
    ones,
    pi,
    sort,
    sum,
    zeros,
)
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import iv

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class BinghamDistribution(AbstractHypersphericalDistribution):
    def __init__(self, Z, M):
        AbstractHypersphericalDistribution.__init__(self, M.shape[0] - 1)

        assert M.shape[1] == self.input_dim, "M is not square"
        assert Z.shape[0] == self.input_dim, "Z has wrong length"
        assert Z.ndim == 1, "Z needs to be a 1-D vector"
        assert Z[-1] == 0, "Last entry of Z needs to be zero"
        assert all(Z[:-1] <= Z[1:]), "Values in Z have to be ascending"

        # Verify that M is orthogonal
        epsilon = array(0.001)
        assert max(abs(M @ M.T - eye(self.dim + 1))) < epsilon, "M is not orthogonal"

        self.Z = Z
        self.M = M
        self._F = None
        self._dF = None

    @property
    def F(self):
        if self._F is None:
            # Currently, only supporting numerical integration
            # Temporarily set _F to 1 to use .integrate_numerically to calculate the normalization constant
            self._F = 1
            self._F = self.integrate_numerically()
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @staticmethod
    def calculate_F(Z):
        """Uses analytical method. Supports 2-D and 4-D distributions."""
        if Z.shape[0] == 2:
            # F = exp((Z[0]+Z[1])/2) * 2*pi * I_0(|Z[0]-Z[1]|/2)
            return float(
                exp((Z[0] + Z[1]) / 2) * 2 * pi * iv(0, abs(float(Z[0] - Z[1])) / 2)
            )
        assert Z.shape[0] == 4

        def J(Z, u):
            return iv(0, 0.5 * abs(Z[0] - Z[1]) * u) * iv(
                0, 0.5 * abs(Z[2] - Z[3]) * (1 - u)
            )

        def ifun(u):
            return J(Z, u) * exp(
                0.5 * (Z[0] + Z[1]) * u + 0.5 * (Z[2] + Z[3]) * (1 - u)
            )

        return 2 * pi**2 * quad(ifun, 0, 1)[0]

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1

        C = self.M @ diag(self.Z) @ self.M.T
        p = 1 / self.F * exp(sum(xs * (xs @ C), axis=-1))
        return p

    def mean_direction(self):
        raise NotImplementedError(
            "Due to its symmetry, the mean direction is undefined for Bingham distributions."
        )

    def mean_axis(self):
        """
        Returns the principal axis of the Bingham distribution as a unit vector
        in R^{dim+1}. Because of antipodal symmetry, v and -v represent the
        same axis; this method returns one of them.
        """
        # Second-moment / scatter matrix
        S = self.moment()

        # Eigen-decomposition of S (symmetric by construction)
        D, V = linalg.eigh(S)

        # Index of largest eigenvalue
        order = argsort(D)
        axis = V[:, order[-1]]

        # Optionally enforce unit norm (usually already true)
        # axis = axis / linalg.norm(axis)

        return axis

    def multiply(self, B2):
        assert isinstance(B2, BinghamDistribution)
        if self.dim != B2.dim:
            raise ValueError("Dimensions do not match")

        C = (
            self.M @ diag(self.Z.ravel()) @ self.M.T
            + B2.M @ diag(B2.Z.ravel()) @ B2.M.T
        )  # New exponent

        C = 0.5 * (C + C.T)  # Symmetrize
        D, V = linalg.eigh(C)
        order = argsort(D)  # Sort eigenvalues
        V = V[:, order]
        Z_ = D[order]
        Z_ = Z_ - Z_[-1]  # Ensure last entry is zero
        M_ = V
        return BinghamDistribution(Z_, M_)

    def sample(self, n):
        return self.sample_metropolis_hastings(n)

    @property
    def dF(self):
        if self._dF is None:
            self._dF = self.calculate_dF()
        return self._dF

    def calculate_dF(self):
        dim = self.Z.shape[0]  # Assuming Z is a property of the object
        dF = zeros(dim)
        epsilon = 0.001
        for i in range(dim):
            # Using finite differences
            dZ = zeros(dim)
            dZ[i] = epsilon
            F1 = self.calculate_F(self.Z + dZ)
            F2 = self.calculate_F(self.Z - dZ)
            dF[i] = (F1 - F2) / (2 * epsilon)
        return dF

    def sample_kent(self, n):
        raise NotImplementedError("Not yet implemented.")

    def moment(self):
        """
        Returns:
            S (numpy.ndarray): scatter/covariance matrix in R^d
        """
        D = diag(self.dF / self.F)
        # It should already be normalized, but numerical inaccuracies can lead to values unequal to 1
        D = D / sum(diag(D))
        S = self.M @ D @ self.M.T
        S = (S + S.T) / 2  # Enforce symmetry
        return S

    def mode(self):
        """Returns the mode of the Bingham distribution.

        The mode is the eigenvector corresponding to Z=0 (the maximum), i.e.,
        the last column of M.

        Returns:
            mode (numpy.ndarray): mode as a unit vector in R^{dim+1}
        """
        return self.M[:, -1]

    def sample_deterministic(self, _spread=0.5):
        """Returns deterministic sigma-point samples and weights.

        Generates 2*(dim+1) sigma points as ±columns of M with weights
        derived from the normalized moments, so that the weighted scatter
        matrix equals the distribution's moment matrix.

        Parameters:
            _spread (float): spread parameter reserved for future use (e.g., tuning
                the sigma-point placement); currently the samples are always ±M columns

        Returns:
            samples (numpy.ndarray): shape (dim+1, 2*(dim+1)), columns are samples
            weights (numpy.ndarray): shape (2*(dim+1),), non-negative weights summing to 1
        """
        d = self.dF / self.F
        d = d / sum(d)  # normalize
        # ±columns of M with equal weight d_i/2 for both signs
        samples = concatenate([self.M, -self.M], axis=1)
        weights = concatenate([d / 2, d / 2])
        return samples, weights

    @staticmethod
    def _right_mult_matrix(q):
        """Right multiplication matrix for complex (2D) or quaternion (4D).

        For 2D complex q = [a, b]: z * q corresponds to [[a, -b], [b, a]] * z
        For 4D quaternion q = [w, x, y, z]: p * q = R(q) * p where R is returned.
        """
        if q.shape[0] == 2:
            return array([[q[0], -q[1]], [q[1], q[0]]])
        if q.shape[0] == 4:
            w, x, y, z = q[0], q[1], q[2], q[3]
            return array(
                [
                    [w, -x, -y, -z],
                    [x, w, z, -y],
                    [y, -z, w, x],
                    [z, y, -x, w],
                ]
            )
        raise ValueError("Only 2D and 4D are supported")

    def compose(self, B2):
        """Compose two Bingham distributions via complex or quaternion multiplication.

        Computes the Bingham distribution approximating the scatter matrix of
        the product x*y, where x ~ self and y ~ B2 are independent.

        Parameters:
            B2 (BinghamDistribution): second distribution

        Returns:
            BinghamDistribution: composed distribution
        """
        assert isinstance(B2, BinghamDistribution)
        assert self.dim == B2.dim, "Dimensions must match"
        assert self.dim in (1, 3), "Compose only supported for 2D and 4D distributions"

        d2 = B2.dF / B2.F
        d2 = d2 / sum(d2)
        S1 = self.moment()

        n = self.input_dim
        S = zeros((n, n))
        for j in range(n):
            R_j = BinghamDistribution._right_mult_matrix(B2.M[:, j])
            S = S + d2[j] * R_j @ S1 @ R_j.T

        S = (S + S.T) / 2
        return BinghamDistribution.fit_to_moment(S)

    @staticmethod
    def fit_to_moment(S):
        """Fit a Bingham distribution to a given scatter/moment matrix.

        Finds Z and M such that the moment of B(Z, M) matches S.

        Parameters:
            S (numpy.ndarray): symmetric positive semi-definite matrix with trace 1
                (or will be normalized)

        Returns:
            BinghamDistribution: fitted distribution
        """
        n = S.shape[0]
        S_np = array(S, dtype=float)
        S_np = (S_np + S_np.T) / 2

        # Eigendecompose S: eigenvectors sorted by ascending eigenvalue
        eigenvalues, M_np = linalg.eigh(S_np)
        eigenvalues = eigenvalues.real
        M_np = M_np.real

        # Normalize eigenvalues to get target moments (they should sum to 1)
        eigenvalues = maximum(eigenvalues, 0)
        ev_sum = eigenvalues.sum()
        if ev_sum == 0:
            target_d = ones(n) / n
        else:
            target_d = eigenvalues / ev_sum

        def moment_residual(z_free):
            Z_cand = concatenate((z_free, array([0.0])))
            Z_sorted = sort(Z_cand)
            M_sorted = M_np[:, argsort(Z_cand)]
            try:
                B_temp = BinghamDistribution(array(Z_sorted), array(M_sorted))
                d = array(B_temp.dF / B_temp.F, dtype=float)
                d = d / d.sum()
                return d[:-1] - target_d[:-1]
            except (
                AssertionError,
                ValueError,
                RuntimeError,
            ):  # pylint: disable=broad-except
                return ones(n - 1) * 1e6

        # Initial guess: scale based on target moments relative to last
        z0 = -(target_d[-1] - target_d[:-1]) * 10.0
        z_sol = fsolve(moment_residual, z0, full_output=False)

        Z_out = concatenate((z_sol, array([0.0])))
        idx = argsort(Z_out)
        Z_final = Z_out[idx]
        M_final = M_np[:, idx]

        return BinghamDistribution(array(Z_final), array(M_final))
