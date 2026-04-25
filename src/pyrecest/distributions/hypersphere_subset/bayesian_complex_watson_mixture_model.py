# pylint: disable=redefined-builtin,no-name-in-module,no-member
"""
Bayesian Complex Watson Mixture Model.

Extends the complex Watson mixture model by adding:
- A complex Bingham prior for the mode vectors.
- A Dirichlet prior for the mixture weights.

Fitting is performed via a variational EM algorithm.

Reference:
    Derived from libDirectional (MATLAB):
    https://github.com/libDirectional/libDirectional
"""

import pyrecest.backend
from pyrecest.backend import (  # pylint: disable=no-name-in-module,no-member
    allclose,
    any,
    arange,
    argsort,
    asarray,
    copy,
    diag,
    exp,
    eye,
    flip,
    full,
    isnan,
    linalg,
    linspace,
    log,
    maximum,
    ones,
    real,
    zeros,
)
from scipy.special import digamma  # pylint: disable=no-name-in-module

from .complex_watson_distribution import ComplexWatsonDistribution


class BayesianComplexWatsonMixtureModel:
    """
    Bayesian complex Watson mixture model.

    Stores the posterior parameters (complex Bingham matrices B, concentration
    parameters, and Dirichlet parameter alpha) resulting from fitting the model
    to observations.

    Attributes
    ----------
    B : ndarray of shape (D, D, K), complex
        Complex Bingham parameter matrices for each component.
    concentrations : ndarray of shape (K,)
        Concentration parameters (kappa) for each Watson component.
    alpha : ndarray of shape (K,)
        Dirichlet parameter vector (proportional to mixture weights).
    K : int
        Number of mixture components.
    dim : int
        Dimension D of the complex space C^D.
    """

    def __init__(self, B, concentrations, alpha):
        """
        Construct a BayesianComplexWatsonMixtureModel from posterior parameters.

        Args:
            B: (D, D, K) complex array of Hermitian Bingham parameter matrices.
            concentrations: (K,) array of concentration parameters.
            alpha: (K,) Dirichlet parameter vector.
        """
        B = asarray(B, dtype=complex)
        concentrations = asarray(concentrations, dtype=float).ravel()
        alpha = asarray(alpha, dtype=float).ravel()

        K = alpha.shape[0]
        assert B.shape[2] == K, "B.shape[2] must equal len(alpha)"
        assert concentrations.shape[0] == K, "len(concentrations) must equal len(alpha)"

        for k in range(K):
            assert allclose(
                B[:, :, k], B[:, :, k].conj().T, atol=1e-6
            ), f"B[:,:,{k}] must be Hermitian"

        self.B = B
        self.concentrations = concentrations
        self.alpha = alpha
        self.K = K
        self.dim = B.shape[0]

    @staticmethod
    def fit(Z, parameters):
        """
        Fit the model to observations Z using the given hyperparameters.

        Args:
            Z: (D, N) complex matrix of unit-vector observations.
            parameters (dict): Hyperparameter dict. Must contain:
                - ``initial``: dict with keys ``B`` (D,D,K complex), ``alpha`` (K,),
                  and ``kappa`` (scalar or (K,)).
                - ``prior``: dict with keys ``B`` (D,D,K complex) and ``alpha`` (K,).
                - ``I`` (int): number of EM iterations.
                Optionally:
                - ``uniformComponent`` (bool): if True, last component is uniform.
                - ``prior['saliencies']`` (float or (N,)): observation saliencies.

        Returns:
            tuple: (BayesianComplexWatsonMixtureModel, posterior dict)
        """
        posterior = BayesianComplexWatsonMixtureModel.estimate_posterior(Z, parameters)
        model = BayesianComplexWatsonMixtureModel(
            posterior["B"], posterior["kappa"], posterior["alpha"]
        )
        return model, posterior

    @staticmethod
    def fit_default(Z, K):
        """
        Fit the model with default hyperparameters.

        Args:
            Z: (D, N) complex matrix of unit-vector observations.
                The feature dimension D must be less than 100.
            K (int): Number of mixture components.

        Returns:
            tuple: (BayesianComplexWatsonMixtureModel, posterior dict)
        """
        assert (
            Z.shape[0] < 100
        ), "fit_default assumes D < 100 (feature dimension, not sample count)"
        D = Z.shape[0]
        parameters = BayesianComplexWatsonMixtureModel.parameters_default(D, K)
        return BayesianComplexWatsonMixtureModel.fit(Z, parameters)

    @staticmethod
    def parameters_default(D, K):
        """
        Build the default hyperparameter dict for dimension D and K components.

        Args:
            D (int): Dimension of the complex space.
            K (int): Number of mixture components.

        Returns:
            dict: Default parameter dict compatible with ``fit`` and
                ``estimate_posterior``.
        """
        parameters = {}
        parameters["initial"] = {
            "B": zeros((D, D, K), dtype=complex),
            "kappa": 20.0,
            "alpha": 1.0 / K + linspace(-0.14 / K, 0.14 / K, K),
        }
        parameters["prior"] = {
            "B": zeros((D, D, K), dtype=complex),
            "alpha": ones(K) / K,
            "saliencies": 1.0,
        }
        parameters["I"] = 40
        parameters["uniformComponent"] = False
        return parameters

    @staticmethod
    def estimate_posterior(
        Z, parameters
    ):  # pylint: disable=too-many-locals,too-many-statements
        """
        Run the variational EM algorithm to estimate posterior parameters.

        E-step: update soft assignments gamma[n, k] using the current parameters.
        M-step: update B, alpha, and kappa from the weighted sufficient statistics.

        Args:
            Z: (D, N) complex matrix of unit-vector observations.
            parameters (dict): Hyperparameter dict as returned by
                ``parameters_default``.

        Returns:
            dict: Posterior with keys ``B``, ``kappa``, ``alpha``, ``gamma``.
        """
        uniform_component = parameters.get("uniformComponent", False)

        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError(
                "estimate_posterior is not supported on the JAX backend."
            )

        assert "initial" in parameters
        assert "B" in parameters["initial"]
        assert "alpha" in parameters["initial"]
        assert "kappa" in parameters["initial"]
        assert "prior" in parameters
        assert "B" in parameters["prior"]
        assert "alpha" in parameters["prior"]
        assert "I" in parameters

        Z = asarray(Z, dtype=complex)
        D, N = Z.shape
        K = len(asarray(parameters["initial"]["alpha"]).ravel())

        B_init = asarray(parameters["initial"]["B"], dtype=complex)
        for k in range(K):
            assert allclose(
                B_init[:, :, k], B_init[:, :, k].conj().T, atol=1e-6
            ), "initial B must be Hermitian"

        kappa_init = parameters["initial"]["kappa"]
        if asarray(kappa_init).ndim == 0:
            kappa_init_arr = full((K,), float(kappa_init))
        else:
            kappa_init_arr = copy(asarray(kappa_init, dtype=float).ravel())

        posterior = {
            "B": copy(B_init),
            "alpha": copy(asarray(parameters["initial"]["alpha"], dtype=float).ravel()),
            "kappa": kappa_init_arr,
            "gamma": zeros((N, K)),
        }

        # Precompute outer products Z[:, n] * conj(Z)[:, n] reshaped to (D*D, N)
        ZZ = (Z[:, None, :] * Z.conj()[None, :, :]).reshape(D * D, N)

        # Log saliencies shape (N, K)
        saliencies = parameters["prior"].get("saliencies", 1.0)
        saliencies_arr = asarray(saliencies)
        if saliencies_arr.ndim == 0:
            ln_saliencies = full((N, K), log(max(float(saliencies_arr), 1e-7)))
        else:
            saliencies_vec = saliencies_arr.ravel()
            assert len(saliencies_vec) == N
            ln_saliencies = log(maximum(saliencies_vec, 1e-7))[:, None] * ones((N, K))

        prior_B = asarray(parameters["prior"]["B"], dtype=complex)
        prior_alpha = asarray(parameters["prior"]["alpha"], dtype=float).ravel()
        concentration_max = 500.0

        for _ in range(parameters["I"]):
            # E-step
            log_gamma = copy(ln_saliencies)

            quad = BayesianComplexWatsonMixtureModel.quadratic_expectation(
                ZZ.reshape(D, D, N), posterior["B"]
            )
            log_gamma += posterior["kappa"][None, :] * quad
            log_gamma += ComplexWatsonDistribution.log_norm(D, posterior["kappa"])[
                None, :
            ]
            log_gamma += digamma(posterior["alpha"])[None, :]

            log_gamma_row_max = log_gamma.max(axis=1, keepdims=True)
            if (
                pyrecest.backend.__backend_name__ == "pytorch"
            ):  # pylint: disable=no-member
                log_gamma_row_max = log_gamma_row_max.values
            log_gamma = log_gamma - log_gamma_row_max
            gamma = exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)

            assert not any(isnan(gamma)), "NaN in gamma during E-step"
            posterior["gamma"] = gamma

            # M-step
            N_k = gamma.sum(axis=0)

            posterior["alpha"] = prior_alpha + N_k

            cov_matrix = (ZZ @ asarray(gamma, dtype=complex)) / maximum(
                N_k[None, :], 1e-300
            )
            cov_matrix = cov_matrix.reshape(D, D, K)

            for k in range(K):
                posterior["B"][:, :, k] = (
                    posterior["kappa"][k] * N_k[k] * cov_matrix[:, :, k]
                    + prior_B[:, :, k]
                )
                posterior["B"][:, :, k] = 0.5 * (
                    posterior["B"][:, :, k] + posterior["B"][:, :, k].conj().T
                )

            for k in range(K):
                if uniform_component and k == K - 1:
                    posterior["kappa"][k] = 0.0
                    continue

                cov_k = cov_matrix[:, :, k].reshape(D, D, 1)
                Bk = posterior["B"][:, :, k].reshape(D, D, 1)
                quad_k = float(
                    real(
                        BayesianComplexWatsonMixtureModel.quadratic_expectation(
                            cov_k, Bk
                        )[0, 0]
                    )
                )
                posterior["kappa"][k] = (
                    ComplexWatsonDistribution._hypergeometric_ratio_inverse(  # pylint: disable=protected-access
                        quad_k, D, kappa_max=concentration_max
                    )
                )

        return posterior

    @staticmethod
    def quadratic_expectation(dyadic_products, B):  # pylint: disable=too-many-locals
        """
        Compute E_{z ~ cBingham(B_k)}[z^H * A * z] for each A in dyadic_products.

        Approximation:
        - Large eigenvalue regime (any eigenvalue > 1): use first-order moments of
          the complex Bingham, computed by numerical differentiation.
        - Otherwise: assume uniform (E[zz^H] = I/D).

        Args:
            dyadic_products: (D, D, N) complex array of D x D matrices A_n.
            B: (D, D, K) complex Hermitian Bingham parameters, or (D, D) for K=1.

        Returns:
            ndarray of shape (N, K): real values E[z^H A_n z].
        """
        dyadic_products = asarray(dyadic_products, dtype=complex)
        B = asarray(B, dtype=complex)

        if B.ndim == 2:
            B = B[:, :, None]

        D = B.shape[0]
        N = dyadic_products.shape[2] if dyadic_products.ndim == 3 else 1
        K = B.shape[2]

        if dyadic_products.ndim == 2:
            dyadic_products = dyadic_products[:, :, None]

        dp_reshaped = dyadic_products.reshape(D * D, N)
        E = zeros((N, K))

        for k in range(K):
            Bk = 0.5 * (B[:, :, k] + B[:, :, k].conj().T)
            eigenvalues, U = linalg.eigh(Bk)

            idx = flip(argsort(eigenvalues), axis=0)
            Lambda = real(eigenvalues[idx])
            U = U[:, idx]

            if any(Lambda > 1.0):
                Lambda_perturbed = Lambda + arange(1, D + 1) * 1e-2
                Lambda_shifted = Lambda_perturbed - Lambda_perturbed.max()
                c_diag = asarray(
                    _complex_bingham_first_order_moments(Lambda_shifted, D),
                    dtype=complex,
                )
                cov_k = U @ diag(c_diag) @ U.conj().T
            else:
                cov_k = eye(D, dtype=complex) / D

            cov_vec = cov_k.ravel()
            E[:, k] = real(dp_reshaped.T @ cov_vec.conj())

        return E


# ---------------------------------------------------------------------------
# Helpers: first-order moments and simplex integral
# ---------------------------------------------------------------------------


def _complex_bingham_first_order_moments(Lambda_shifted, D):
    """
    Compute E[|z_i|^2] for a diagonal complex Bingham with shifted eigenvalues.

    Uses numerical differentiation of log(int_simplex exp(Lambda.s) ds).

    Args:
        Lambda_shifted: D-dim real array, shifted so max = 0.
        D (int): Dimension.

    Returns:
        ndarray: D-dim non-negative real array normalised to sum 1.
    """
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "_complex_bingham_first_order_moments is not supported on the JAX backend."
        )
    # Convert to Python float list to avoid tensor shallow-copy issues (PyTorch)
    lambda_list = [float(x) for x in asarray(Lambda_shifted, dtype=float)]
    eps = 1e-5
    log_F0 = float(pyrecest.backend.log(max(_simplex_integral(lambda_list), 1e-300)))
    moments = zeros(D)
    for i in range(D):
        l_plus = list(lambda_list)  # list() always deep-copies Python floats
        l_plus[i] += eps
        log_F_plus = float(pyrecest.backend.log(max(_simplex_integral(l_plus), 1e-300)))
        moments[i] = (log_F_plus - log_F0) / eps

    total = float(moments.sum())
    if total > 1e-10:
        moments = moments / total
    else:
        moments = ones(D) / D
    return moments


def _simplex_integral(Lambda):
    """
    Compute int_{standard simplex} exp(Lambda . s) ds.

    Uses the partial-fractions / divided-differences formula:
        I = sum_i exp(Lambda_i) / prod_{j!=i} (Lambda_i - Lambda_j)

    Args:
        Lambda: D-dim real array of eigenvalues.

    Returns:
        float: Integral value (always positive).
    """
    # Convert to Python float list to avoid tensor-backend issues (PyTorch/JAX)
    lambda_list = [float(x) for x in asarray(Lambda, dtype=float)]
    D = len(lambda_list)

    if D == 1:
        return float(exp(lambda_list[0]))

    # Tiny perturbation to resolve exact degeneracy
    lambda_list = [lambda_list[i] + i * 1e-10 for i in range(D)]

    result = 0.0
    for i in range(D):
        denom = 1.0
        for j in range(D):
            if j != i:
                denom *= lambda_list[i] - lambda_list[j]
        if abs(denom) > 1e-300:
            result += float(exp(lambda_list[i])) / denom
    return result
