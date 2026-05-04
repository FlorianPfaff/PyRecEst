import copy
import warnings

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import any as backend_any
from pyrecest.backend import (
    array,
    asarray,
    concatenate,
    einsum,
    exp,
    less_equal,
    linalg,
    log,
    stack,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    transpose,
)

from ..distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from ..distributions.nonperiodic.gaussian_distribution import GaussianDistribution
from ..distributions.nonperiodic.gaussian_mixture import GaussianMixture
from .abstract_filter import AbstractFilter
from .manifold_mixins import HypercylindricalFilterMixin


class StateSpaceSubdivisionFilter(AbstractFilter, HypercylindricalFilterMixin):
    """
    Filter for state spaces that are a Cartesian product of a periodic/bounded
    manifold (represented by a grid distribution) and a linear space (represented
    by per-grid-point Gaussians).

    The filter state is a :class:`StateSpaceSubdivisionGaussianDistribution`.

    This is the Python port of ``StateSpaceSubdivisionFilter`` from libDirectional.
    """

    def __init__(self, initial_state: StateSpaceSubdivisionGaussianDistribution):
        assert isinstance(initial_state, StateSpaceSubdivisionGaussianDistribution)
        HypercylindricalFilterMixin.__init__(self)
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property (override setter for type checking + warning)
    # ------------------------------------------------------------------

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state: StateSpaceSubdivisionGaussianDistribution):
        assert isinstance(
            new_state, StateSpaceSubdivisionGaussianDistribution
        ), "filter_state must be a StateSpaceSubdivisionGaussianDistribution."
        if self._filter_state is not None and len(
            self._filter_state.linear_distributions
        ) != len(new_state.linear_distributions):
            warnings.warn(
                "Number of components differ.",
                UserWarning,
                stacklevel=2,
            )
        self._filter_state = copy.deepcopy(new_state)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_linear(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        transition_density=None,
        covariance_matrices=None,
        system_matrices=None,
        linear_input_vectors=None,
    ):
        """
        Perform the prediction step.

        Parameters
        ----------
        transition_density : AbstractConditionalDistribution or None
            Conditional grid distribution f(next | current) for the
            periodic/bounded part, where ``grid_values[i, j] = f(next=i | current=j)``.
            If ``None``, the periodic transition is assumed to be a Dirac
            (identity), and only the linear part is updated.
        covariance_matrices : array or None
            Process-noise covariance(s) for the linear part.
            Shape ``(lin_dim, lin_dim)`` for a single matrix applied to all
            areas, or ``(lin_dim, lin_dim, n_areas)`` for per-area matrices
            (indexed by the *prior* area in Case 3, or the current area in
            Case 2).  ``None`` means no additive noise.
        system_matrices : array or None
            System matrix/matrices for the linear part.
            Shape ``(lin_dim, lin_dim)`` for a single matrix, or
            ``(lin_dim, lin_dim, n_areas)`` for per-area matrices (indexed
            by the *prior* area).  ``None`` means the identity matrix.
        linear_input_vectors : array or None
            Deterministic input vector(s) for the linear part.
            Shape ``(lin_dim,)`` for a single vector or
            ``(lin_dim, n_areas)`` for per-area vectors (indexed by the
            *prior* area).  ``None`` means zero input.
        """
        state = self._filter_state
        n_areas = len(state.linear_distributions)

        if (
            transition_density is None
            and covariance_matrices is None
            and system_matrices is None
            and linear_input_vectors is None
        ):
            warnings.warn(
                "Nothing to do for this prediction step.",
                UserWarning,
                stacklevel=2,
            )
            return

        if transition_density is None:
            # ----------------------------------------------------------
            # Case 2: No uncertainty in the periodic domain.
            # Only the linear distributions are updated.
            # ----------------------------------------------------------
            for i in range(n_areas):
                mu_i = state.linear_distributions[i].mu
                C_i = state.linear_distributions[i].C

                if system_matrices is not None:
                    F = (
                        system_matrices
                        if system_matrices.ndim == 2
                        else system_matrices[:, :, i]
                    )
                    mu_i = F @ mu_i
                    C_i = F @ C_i @ F.T

                if linear_input_vectors is not None:
                    u = (
                        linear_input_vectors
                        if linear_input_vectors.ndim == 1
                        else linear_input_vectors[:, i]
                    )
                    mu_i = mu_i + u

                if covariance_matrices is not None:
                    Q = (
                        covariance_matrices
                        if covariance_matrices.ndim == 2
                        else covariance_matrices[:, :, i]
                    )
                    C_i = C_i + Q

                state.linear_distributions[i].mu = mu_i
                state.linear_distributions[i].C = C_i

        else:
            # ----------------------------------------------------------
            # Case 3: A transition density for the periodic part is given.
            # ----------------------------------------------------------
            # weightsJoint[i, j] = f(next=i | current=j) * P(current=j)
            # transition_density.grid_values has shape (n, n) with
            # grid_values[i, j] = f(next=i | current=j).
            # state.gd.grid_values has shape (n,) = P(current=j).
            # Broadcasting (n,n) * (n,) multiplies each row i by the
            # current weights, giving weights_joint[i, j] as desired.
            weights_joint = transition_density.grid_values * state.gd.grid_values

            # New marginal for the periodic part (Chapman-Kolmogorov):
            # new_gd_values[i] = (manifold_size / n) * sum_j weights_joint[i, j]
            manifold_size = state.gd.get_manifold_size()
            new_gd_values = manifold_size / n_areas * backend_sum(weights_joint, axis=1)

            # Apply linear system to each *current* (prior) linear distribution.
            x_preds = []
            c_preds = []
            for j in range(n_areas):
                mu_j = state.linear_distributions[j].mu
                C_j = state.linear_distributions[j].C

                if system_matrices is not None:
                    F = (
                        system_matrices
                        if system_matrices.ndim == 2
                        else system_matrices[:, :, j]
                    )
                    mu_j = F @ mu_j
                    C_j = F @ C_j @ F.T

                if linear_input_vectors is not None:
                    u = (
                        linear_input_vectors
                        if linear_input_vectors.ndim == 1
                        else linear_input_vectors[:, j]
                    )
                    mu_j = mu_j + u

                if covariance_matrices is not None:
                    Q = (
                        covariance_matrices
                        if covariance_matrices.ndim == 2
                        else covariance_matrices[:, :, j]
                    )
                    C_j = C_j + Q

                x_preds.append(mu_j)
                c_preds.append(C_j)

            # For each next-area i, reduce the Gaussian mixture of all
            # weighted prior-area contributions to a single Gaussian.
            means = array(x_preds)  # (n_areas, lin_dim)
            covs_stacked = stack(c_preds, axis=2)  # (lin_dim, lin_dim, n_areas)

            new_linear_distributions = list(state.linear_distributions)
            for i in range(n_areas):
                row_weights = weights_joint[i, :]
                total = float(backend_sum(row_weights))
                if total < 1e-300:
                    # Keep the old distribution if no probability mass flows here
                    continue
                norm_weights = row_weights / total

                mu_new, C_new = (
                    GaussianMixture.mixture_parameters_to_gaussian_parameters(
                        means, covs_stacked, norm_weights
                    )
                )
                new_dist = copy.deepcopy(state.linear_distributions[i])
                new_dist.mu = mu_new
                new_dist.C = C_new
                new_linear_distributions[i] = new_dist

            state.linear_distributions = new_linear_distributions
            state.gd.grid_values = new_gd_values
            state.gd.normalize_in_place(warn_unnorm=False)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self, likelihood_periodic_grid=None, likelihoods_linear=None
    ):  # pylint: disable=too-many-locals
        """
        Perform the measurement update step.

        Parameters
        ----------
        likelihood_periodic_grid : array or AbstractDistribution or None
            Likelihood values on the periodic grid.  If an
            ``AbstractDistribution`` is given, it is evaluated at the grid
            points.  Must have the same shape as ``filter_state.gd.grid_values``
            or be ``None`` (uniform likelihood over the bounded domain).
        likelihoods_linear : list of GaussianDistribution or None
            Gaussian likelihood(s) for the linear part.  Either a list with
            one element (applied to all areas) or a list with as many elements
            as there are grid points.  ``None`` means uniform likelihood.
        """
        state = self._filter_state
        n_areas = len(state.linear_distributions)

        # Normalise the likelihoods_linear argument
        if likelihoods_linear is not None and len(likelihoods_linear) == 0:
            likelihoods_linear = None

        if likelihood_periodic_grid is None and likelihoods_linear is None:
            warnings.warn(
                "Nothing to do for this update step.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Convert distribution to grid-value array if necessary
        if likelihood_periodic_grid is not None and hasattr(
            likelihood_periodic_grid, "pdf"
        ):
            grid = state.gd.get_grid()
            likelihood_periodic_grid = likelihood_periodic_grid.pdf(grid)

        # Update grid weights from periodic likelihood.
        # Flatten to 1-D to avoid shape-broadcasting surprises (e.g. when
        # pdf() returns shape (n, 1) instead of (n,)).
        if likelihood_periodic_grid is not None:
            state.gd.grid_values = state.gd.grid_values * asarray(
                likelihood_periodic_grid
            ).reshape(-1)

        if likelihoods_linear is not None:
            n_likelihoods = len(likelihoods_linear)
            assert n_likelihoods in (
                1,
                n_areas,
            ), "likelihoods_linear must have 1 or n_areas elements."

            if n_likelihoods == 1:
                self._update_single_linear_likelihood(likelihoods_linear[0])
            else:
                # Compute the grid-weight factor for each area i:
                #   factor_i = N(mu_pred_i; mu_like_j, C_pred_i + C_like_j)
                # where j = i if n_likelihoods > 1 else 0.
                # Collect into an array and multiply in one shot (backend-safe).
                pdf_values = []
                for i in range(n_areas):
                    j = i if n_likelihoods > 1 else 0
                    combined_cov = (
                        state.linear_distributions[i].C + likelihoods_linear[j].C
                    )
                    temp_g = GaussianDistribution(
                        likelihoods_linear[j].mu, combined_cov, check_validity=False
                    )
                    pdf_values.append(
                        asarray(temp_g.pdf(state.linear_distributions[i].mu)).reshape(
                            (1,)
                        )
                    )
                factors = concatenate(pdf_values)  # shape (n_areas,)
                state.gd.grid_values = state.gd.grid_values * factors

                # Update the linear distributions (Kalman update in information form):
                #   C_new^{-1} = C_prior^{-1} + C_like^{-1}
                #   mu_new = C_new (C_prior^{-1} mu_prior + C_like^{-1} mu_like)
                for i in range(n_areas):
                    j = i if n_likelihoods > 1 else 0
                    C_prior_inv = linalg.inv(state.linear_distributions[i].C)
                    C_like_inv = linalg.inv(likelihoods_linear[j].C)
                    C_new_inv = C_prior_inv + C_like_inv
                    C_new = linalg.inv(C_new_inv)
                    mu_new = C_new @ (
                        C_prior_inv @ state.linear_distributions[i].mu
                        + C_like_inv @ likelihoods_linear[j].mu
                    )
                    state.linear_distributions[i].mu = mu_new
                    state.linear_distributions[i].C = C_new

        state.gd.normalize_in_place(warn_unnorm=False)

    def _update_single_linear_likelihood(
        self, likelihood
    ):  # pylint: disable=too-many-locals
        """Vectorized update for one linear Gaussian likelihood."""

        state = self._filter_state
        likelihood_mu = asarray(likelihood.mu, dtype=float).reshape(-1)
        likelihood_cov = asarray(likelihood.C, dtype=float)
        means = stack([dist.mu for dist in state.linear_distributions])
        covariances = stack([dist.C for dist in state.linear_distributions])

        combined_covariances = covariances + likelihood_cov
        deltas = means - likelihood_mu
        combined_determinants = linalg.det(combined_covariances)
        if bool(backend_any(less_equal(combined_determinants, 0.0))):
            raise ValueError("Combined covariance matrices must be positive definite.")

        combined_precisions = linalg.inv(combined_covariances)
        solved_deltas = einsum("nij,nj->ni", combined_precisions, deltas)
        exponents = einsum("ni,ni->n", deltas, solved_deltas)
        dimension = likelihood_mu.shape[0]
        factors = exp(
            -0.5
            * (
                dimension * log(2.0 * 3.141592653589793)
                + log(combined_determinants)
                + exponents
            )
        )
        state.gd.grid_values = state.gd.grid_values * factors

        prior_precisions = linalg.inv(covariances)
        likelihood_precision = linalg.inv(likelihood_cov)
        posterior_precisions = prior_precisions + likelihood_precision
        posterior_covariances = linalg.inv(posterior_precisions)
        posterior_covariances = 0.5 * (
            posterior_covariances + transpose(posterior_covariances, (0, 2, 1))
        )

        prior_information_means = einsum("nij,nj->ni", prior_precisions, means)
        likelihood_information_mean = likelihood_precision @ likelihood_mu
        posterior_means = einsum(
            "nij,nj->ni",
            posterior_covariances,
            prior_information_means + likelihood_information_mean,
        )

        for distribution, posterior_mean, posterior_covariance in zip(
            state.linear_distributions,
            posterior_means,
            posterior_covariances,
            strict=True,
        ):
            distribution.mu = posterior_mean
            distribution.C = posterior_covariance

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    def get_estimate(self) -> StateSpaceSubdivisionGaussianDistribution:
        """Return the current filter state."""
        return self._filter_state

    def get_point_estimate(self):
        """Return the hybrid mean (periodic mean + linear mean)."""
        return self._filter_state.hybrid_mean()
