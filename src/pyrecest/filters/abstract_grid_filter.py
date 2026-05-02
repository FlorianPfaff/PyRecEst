import warnings

from beartype import beartype
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

from .abstract_filter import AbstractFilter


class AbstractGridFilter(AbstractFilter):
    @beartype
    def __init__(self, state_init: AbstractGridDistribution):
        AbstractFilter.__init__(self, state_init)

    @property
    def filter_state(self):
        """Expose the parent property so we can attach a setter to it."""
        return super().filter_state

    @filter_state.setter
    @beartype
    def filter_state(self, new_state: AbstractManifoldSpecificDistribution):
        if not isinstance(new_state, AbstractGridDistribution):
            warnings.warn(
                "new_state is not a GridDistribution. Transforming the distribution with a number of coefficients equal to that of the filter.",
                RuntimeWarning,
            )
            new_state = self.filter_state.from_distribution(
                new_state,
                self.filter_state.grid_values.shape[0],
                self.filter_state.enforce_pdf_nonnegative,
            )
        elif self.filter_state.grid_values.shape != new_state.grid_values.shape:
            warnings.warn(
                "New grid has a different number of grid points.", RuntimeWarning
            )

        self._filter_state = new_state

    def update_nonlinear(self, likelihood, z):
        grid_vals_new = self.filter_state.grid_values * likelihood(
            z, self.filter_state.get_grid()
        )
        assert grid_vals_new.shape == self.filter_state.grid_values.shape

        self.filter_state.grid_values = grid_vals_new
        self.filter_state.normalize_in_place(warn_unnorm=False)

    def update_model(self, measurement_model, z):
        """Update the grid state using a reusable likelihood model.

        Parameters
        ----------
        measurement_model : object
            Model object exposing either ``likelihood_values(z, grid)`` or a
            callable ``likelihood(z, grid)`` attribute. The method must return
            one likelihood value per point of ``self.filter_state.get_grid()``.
        z : array-like
            Measurement passed to the model.

        Notes
        -----
        This adapter preserves the existing :meth:`update_nonlinear` API and
        simply delegates to it after extracting the likelihood capability from
        the model object.
        """
        if hasattr(measurement_model, "likelihood_values"):
            likelihood = measurement_model.likelihood_values
        elif hasattr(measurement_model, "likelihood") and callable(
            measurement_model.likelihood
        ):
            likelihood = measurement_model.likelihood
        else:
            raise TypeError(
                "measurement_model must expose likelihood_values(z, grid) "
                "or a callable likelihood(z, grid) attribute."
            )
        self.update_nonlinear(likelihood, z)

    def predict_model(self, transition_model):
        """Predict using a reusable grid transition-density model.

        Parameters
        ----------
        transition_model : object
            Model object exposing either ``transition_density_for_filter(self)``
            or a ``transition_density`` attribute compatible with this filter's
            density-based prediction method.

        Notes
        -----
        This generic adapter is intentionally conservative: concrete grid
        filters still define the actual density-based prediction method. If a
        filter does not expose ``predict_nonlinear_via_transition_density``, a
        clear ``NotImplementedError`` is raised.
        """
        if not hasattr(self, "predict_nonlinear_via_transition_density"):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement "
                "predict_nonlinear_via_transition_density."
            )

        if hasattr(transition_model, "transition_density_for_filter"):
            transition_density = transition_model.transition_density_for_filter(self)
        elif hasattr(transition_model, "transition_density"):
            transition_density = transition_model.transition_density
        else:
            raise TypeError(
                "transition_model must expose transition_density_for_filter(filter) "
                "or a transition_density attribute."
            )
        self.predict_nonlinear_via_transition_density(transition_density)

    def plot_filter_state(self):
        self.filter_state.plot_state()
