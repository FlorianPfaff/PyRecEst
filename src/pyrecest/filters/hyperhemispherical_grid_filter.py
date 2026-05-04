import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    array,
    exp,
    linalg,
    sum,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_watson_distribution import (
    HyperhemisphericalWatsonDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_mixture import (
    HypersphericalMixture,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
)

from .abstract_grid_filter import AbstractGridFilter
from .manifold_mixins import HyperhemisphericalFilterMixin


class HyperhemisphericalGridFilter(AbstractGridFilter, HyperhemisphericalFilterMixin):
    """
    Grid-based recursive Bayesian filter on the hyperhemisphere.

    The state is represented as a :class:`HyperhemisphericalGridDistribution`.

    Ported from libDirectional's ``HyperhemisphericalGridFilter.m``.
    """

    def __init__(self, no_of_grid_points, dim, grid_type="leopardi_symm"):
        """
        Parameters
        ----------
        no_of_grid_points : int
            Number of grid points on the hemisphere.
        dim : int
            Manifold dimension of the hyperhemisphere (e.g. 2 for S2-half).
        grid_type : str
            Grid type, defaults to ``'leopardi_symm'``.
        """
        initial_state = HyperhemisphericalGridDistribution.from_distribution(
            HyperhemisphericalUniformDistribution(dim),
            no_of_grid_points,
            grid_type,
        )
        HyperhemisphericalFilterMixin.__init__(self)
        AbstractGridFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property / setter
    # ------------------------------------------------------------------

    @property
    def filter_state(self):
        return super().filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, HyperhemisphericalGridDistribution):
            if not allclose(
                self._filter_state.get_grid(), new_state.get_grid(), atol=1e-10
            ):
                warnings.warn(
                    "setState:gridDiffers: New density is defined on a different grid.",
                    RuntimeWarning,
                )
            self._filter_state = new_state
        elif isinstance(new_state, HypersphericalGridDistribution):
            warnings.warn(
                "setState:fullSphere: Called set_state with a GridDistribution on the "
                "entire hypersphere. Please ensure it is at least symmetric.",
                RuntimeWarning,
            )
            n_half = self._filter_state.grid.shape[0]
            hemi_grid = new_state.get_grid()[:n_half]
            hemi_values = new_state.grid_values[:n_half]
            new_hemi = HyperhemisphericalGridDistribution(hemi_grid, hemi_values)
            if not allclose(
                self._filter_state.get_grid(), new_hemi.get_grid(), atol=1e-10
            ):
                warnings.warn(
                    "setState:gridDiffers: New density is defined on a different grid.",
                    RuntimeWarning,
                )
            self._filter_state = new_hemi
        else:
            warnings.warn(
                "setState:nonGrid: new_state is not a "
                "HyperhemisphericalGridDistribution. Transforming with a number of "
                "coefficients equal to that of the filter.",
                RuntimeWarning,
            )
            new_state = HyperhemisphericalGridDistribution.from_distribution(
                new_state,
                self._filter_state.grid_values.shape[0],
                self._filter_state.grid_type,
            )
            self._filter_state = new_state

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, d_sys):
        """
        Predict assuming an identity system model with noise ``d_sys``.

        Supported: :class:`HyperhemisphericalWatsonDistribution`,
        :class:`WatsonDistribution`, symmetric two-component
        :class:`HypersphericalMixture` of VMF distributions.

        Parameters
        ----------
        d_sys : AbstractDistribution
            System noise distribution with the same ``dim`` as the filter.
        """
        assert (
            d_sys.dim == self.dim - 1
        ), f"d_sys.dim ({d_sys.dim}) must equal filter manifold dim ({self.dim - 1})"
        warnings.warn(
            "PredictIdentity:Inefficient: Using inefficient prediction. Consider "
            "precalculating the SdHalfCondSdHalfGridDistribution and using "
            "predict_nonlinear_via_transition_density.",
            UserWarning,
        )
        f_trans = HyperhemisphericalGridFilter.sys_noise_to_transition_density(
            d_sys, self._filter_state.grid_values.shape[0]
        )
        self.predict_nonlinear_via_transition_density(f_trans)

    def predict_nonlinear_via_transition_density(self, f_trans):
        """
        Perform prediction using a precomputed transition density.

        Parameters
        ----------
        f_trans : SdHalfCondSdHalfGridDistribution
            Must use the same grid as the current filter state.
        """
        from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import (  # pylint: disable=import-outside-toplevel
            SdHalfCondSdHalfGridDistribution,
        )

        assert isinstance(
            f_trans, SdHalfCondSdHalfGridDistribution
        ), "f_trans must be a SdHalfCondSdHalfGridDistribution"
        assert allclose(
            self._filter_state.get_grid(), f_trans.get_grid(), atol=1e-10
        ), (
            "predictNonlinearViaTransitionDensity:gridDiffers: "
            "f_trans is using an incompatible grid."
        )

        self._filter_state = self._filter_state.normalize()
        n_grid = self._filter_state.grid_values.shape[0]
        manifold_size = self._filter_state.get_manifold_size()
        grid_values_new = (
            manifold_size
            / n_grid
            * f_trans.grid_values
            @ self._filter_state.grid_values
        )
        self._filter_state = HyperhemisphericalGridDistribution(
            self._filter_state.get_grid(), grid_values_new
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, meas_noise, z):
        """
        Perform a measurement update assuming an identity measurement model.

        Supported noise: :class:`HyperhemisphericalWatsonDistribution`,
        :class:`WatsonDistribution`, :class:`VonMisesFisherDistribution`
        (with ``mu[-1] == 0``), symmetric :class:`HypersphericalMixture`.

        Parameters
        ----------
        meas_noise : AbstractDistribution
            Measurement noise centred at ``[0, …, 0, 1]``.
        z : array, shape (dim,)
            Measurement on the hemisphere.
        """
        assert z.shape[0] == self.dim

        if isinstance(meas_noise, HyperhemisphericalWatsonDistribution):
            meas_noise = meas_noise.set_mode(z)
        elif isinstance(meas_noise, WatsonDistribution):
            from pyrecest.backend import (  # pylint: disable=import-outside-toplevel
                linalg,
            )

            standard_pole = array([*([0.0] * (self.dim - 1)), 1.0])
            if linalg.norm(meas_noise.mu - standard_pole) > 1e-6:
                raise ValueError(
                    "UpdateIdentity:UnexpectedMeas: mu needs to be [0;...; 0; 1]."
                )
            meas_noise = WatsonDistribution(z, meas_noise.kappa)
        elif isinstance(meas_noise, VonMisesFisherDistribution) and z[-1] == 0:
            from pyrecest.backend import (  # pylint: disable=import-outside-toplevel
                linalg,
            )

            standard_pole = array([*([0.0] * (self.dim - 1)), 1.0])
            if linalg.norm(meas_noise.mu - standard_pole) > 1e-6:
                raise ValueError(
                    "UpdateIdentity:UnexpectedMeas: mu needs to be [0;...; 0; 1]."
                )
            meas_noise = VonMisesFisherDistribution(z, meas_noise.kappa)
        elif (
            isinstance(meas_noise, HypersphericalMixture)
            and len(meas_noise.dists) == 2
            and all(abs(w - 0.5) < 1e-12 for w in meas_noise.w)
        ):
            meas_noise.dists[0].mu = z
            meas_noise.dists[1].mu = -z
        else:
            raise ValueError(
                "UpdateIdentity:UnsupportedNoise: unsupported measurement noise type."
            )

        curr_grid = self._filter_state.get_grid()
        meas_gd = HyperhemisphericalGridDistribution(
            curr_grid, 2.0 * meas_noise.pdf(curr_grid)
        )
        self._filter_state = self._filter_state.multiply(meas_gd)

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """
        Compute a point estimate from the dominant scatter-matrix eigenvector.

        Returns
        -------
        p : array, shape (dim,)
            Point estimate on the upper hemisphere.
        """
        gd_full = self._filter_state.to_full_sphere()
        weights = gd_full.grid_values / sum(gd_full.grid_values)
        S = gd_full.grid.T @ (gd_full.grid * weights[:, None])
        S = 0.5 * (S + S.T)
        _, eigenvectors = linalg.eigh(S)
        p = eigenvectors[:, -1]
        if p[-1] < 0:
            p = -p
        return p

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sys_noise_to_transition_density(d_sys, no_grid_points):
        """
        Build a :class:`SdHalfCondSdHalfGridDistribution` from a noise distribution.

        Parameters
        ----------
        d_sys : AbstractDistribution
            Supported: :class:`HyperhemisphericalWatsonDistribution`,
            :class:`WatsonDistribution`, symmetric :class:`HypersphericalMixture`.
        no_grid_points : int
            Number of grid points on the hemisphere.

        Returns
        -------
        SdHalfCondSdHalfGridDistribution
        """
        from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import (  # pylint: disable=import-outside-toplevel
            SdHalfCondSdHalfGridDistribution,
        )

        if isinstance(
            d_sys, (HyperhemisphericalWatsonDistribution, WatsonDistribution)
        ):
            kappa = d_sys.kappa

            def trans_cp(grid, _grid):
                dots_sq = (grid @ grid.T) ** 2
                w0 = WatsonDistribution(grid[0], kappa)
                return 2.0 * w0.norm_const * exp(kappa * dots_sq)

        elif (
            isinstance(d_sys, HypersphericalMixture)
            and len(d_sys.dists) == 2
            and all(abs(w - 0.5) < 1e-12 for w in d_sys.w)
            and allclose(d_sys.dists[0].mu, -d_sys.dists[1].mu, atol=1e-12)
            and d_sys.dists[0].kappa == d_sys.dists[1].kappa
        ):
            kappa = d_sys.dists[0].kappa

            def trans_cp(grid, _grid):
                dots = grid @ grid.T
                vmf0 = VonMisesFisherDistribution(grid[0], kappa)
                return vmf0.norm_const * (exp(kappa * dots) + exp(-kappa * dots))

        else:
            raise ValueError(
                "sys_noise_to_transition_density: unsupported distribution. "
                "Must be Watson or symmetric two-component HypersphericalMixture."
            )

        embedding_dim = d_sys.dim + 1
        product_dim = 2 * embedding_dim
        return SdHalfCondSdHalfGridDistribution.from_function(
            trans_cp,
            no_grid_points,
            fun_does_cartesian_product=True,
            grid_type="leopardi_symm",
            dim=product_dim,
        )
