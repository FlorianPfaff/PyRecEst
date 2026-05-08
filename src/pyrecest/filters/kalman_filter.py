# pylint: disable=no-name-in-module,no-member

from pyrecest.backend import atleast_1d, atleast_2d, eye
from pyrecest.distributions import GaussianDistribution

from ._linear_gaussian import (
    linear_gaussian_predict,
    linear_gaussian_update,
    linear_gaussian_update_robust,
)
from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin

_MODEL_ATTRIBUTE_SENTINEL = object()


def _get_required_model_attribute(model, *names):
    """Return the first available attribute from a structural model object."""
    for name in names:
        value = getattr(model, name, _MODEL_ATTRIBUTE_SENTINEL)
        if value is not _MODEL_ATTRIBUTE_SENTINEL:
            return value

    options = ", ".join(f"`{name}`" for name in names)
    raise AttributeError(
        f"{type(model).__name__} must expose one of the following "
        f"attributes: {options}."
    )


def _get_optional_model_attribute(model, *names):
    """Return the first available optional structural-model attribute."""
    for name in names:
        value = getattr(model, name, _MODEL_ATTRIBUTE_SENTINEL)
        if value is not _MODEL_ATTRIBUTE_SENTINEL:
            return value

    return None


class KalmanFilter(AbstractFilter, EuclideanFilterMixin):
    """Kalman filter for linear Gaussian Euclidean state-space models.

    The filter state is stored as a :class:`GaussianDistribution` with mean
    vector shape ``(n,)`` and covariance matrix shape ``(n, n)``. Prediction
    uses ``x_k = F x_{k-1} + u + w`` and updates use
    ``z_k = H x_k + v``.
    """

    def __init__(self, initial_state):
        """Create a Kalman filter from a Gaussian state.

        Parameters
        ----------
        initial_state : GaussianDistribution or tuple
            Initial state distribution. Tuples must contain ``(mean,
            covariance)`` with mean shape ``(n,)`` or scalar shape for a
            one-dimensional state, and covariance shape ``(n, n)``.
        """
        EuclideanFilterMixin.__init__(self)
        AbstractFilter.__init__(self, self._coerce_state(initial_state))

    @staticmethod
    def _coerce_state(state):
        if isinstance(state, GaussianDistribution):
            return GaussianDistribution(
                atleast_1d(state.mu),
                atleast_2d(state.C),
                check_validity=False,
            )
        if isinstance(state, tuple) and len(state) == 2:
            mean, covariance = state
            return GaussianDistribution(
                atleast_1d(mean),
                atleast_2d(covariance),
                check_validity=False,
            )
        raise ValueError(
            "state must be a GaussianDistribution or a tuple of (mean, covariance)"
        )

    @property
    def dim(self):
        """Return the dimension ``n`` of the Euclidean state vector."""
        return self._filter_state.dim

    @property
    def filter_state(self) -> GaussianDistribution:
        """Return a Gaussian copy of the current filter state."""
        return GaussianDistribution(
            self._filter_state.mu,
            self._filter_state.C,
            check_validity=False,
        )

    @filter_state.setter
    def filter_state(self, new_state):
        """
        Set the filter state.

        Parameters
        ----------
        new_state : GaussianDistribution or tuple
            Replacement state distribution. Tuples must contain ``(mean,
            covariance)``.
        """
        self._filter_state = self._coerce_state(new_state)

    def predict_identity(self, sys_noise_cov, sys_input=None):
        """Predict one step with an identity transition matrix.

        Parameters
        ----------
        sys_noise_cov : array-like, shape (n, n)
            Additive process-noise covariance.
        sys_input : array-like, shape (n,), optional
            Additive deterministic input applied after the identity transition.
        """
        self.predict_linear(eye(self.dim), sys_noise_cov, sys_input)

    def predict_linear(
        self,
        system_matrix,
        sys_noise_cov,
        sys_input=None,
    ):
        """Predict one step with a linear Gaussian system model.

        Parameters
        ----------
        system_matrix : array-like, shape (n, n)
            State-transition matrix ``F``.
        sys_noise_cov : array-like, shape (n, n)
            Additive process-noise covariance ``Q``.
        sys_input : array-like, shape (n,), optional
            Additive deterministic input ``u``.
        """
        new_mean, new_covariance = linear_gaussian_predict(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            system_matrix=system_matrix,
            sys_noise_cov=sys_noise_cov,
            sys_input=sys_input,
        )
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )

    def predict_model(self, transition_model):
        """Predict one step with a linear Gaussian transition model object.

        The model object is consumed structurally. It must expose a
        ``system_matrix`` attribute and either ``system_noise_cov`` or
        ``sys_noise_cov``. If present, ``sys_input`` or ``system_input`` is
        forwarded as the deterministic transition input.

        Parameters
        ----------
        transition_model : object
            Linear Gaussian transition model object compatible with
            :meth:`predict_linear`.
        """
        system_matrix = _get_required_model_attribute(
            transition_model,
            "system_matrix",
        )
        sys_noise_cov = _get_required_model_attribute(
            transition_model,
            "system_noise_cov",
            "sys_noise_cov",
        )
        sys_input = _get_optional_model_attribute(
            transition_model,
            "sys_input",
            "system_input",
        )
        self.predict_linear(system_matrix, sys_noise_cov, sys_input)

    def update_identity(
        self,
        meas_noise,
        measurement,
        *,
        return_diagnostics=False,
        scale=1.0,
        action="updated",
    ):
        """Update with a measurement matrix equal to the identity.

        Parameters
        ----------
        meas_noise : array-like, shape (n, n)
            Measurement-noise covariance.
        measurement : array-like, shape (n,)
            Measurement vector.
        return_diagnostics : bool, optional
            If true, return update diagnostics after updating the filter state.
        scale : float, optional
            Multiplicative measurement-noise scale used for the update.
        action : str, optional
            Caller-defined diagnostic label for the update action.
        """
        return self.update_linear(
            measurement=measurement,
            measurement_matrix=eye(self.dim),
            meas_noise=meas_noise,
            return_diagnostics=return_diagnostics,
            scale=scale,
            action=action,
        )

    def update_linear(
        self,
        measurement,
        measurement_matrix,
        meas_noise,
        *,
        return_diagnostics=False,
        scale=1.0,
        action="updated",
    ):
        """Update the state with a linear Gaussian measurement model.

        Parameters
        ----------
        measurement : array-like, shape (m,)
            Measurement vector ``z``.
        measurement_matrix : array-like, shape (m, n)
            Measurement matrix ``H`` mapping state vectors to measurement
            vectors.
        meas_noise : array-like, shape (m, m)
            Measurement-noise covariance ``R``.
        return_diagnostics : bool, optional
            If true, return a diagnostics dictionary after updating the filter
            state. The dictionary contains ``nis``, ``residual``, ``scale``,
            and ``action``.
        scale : float, optional
            Multiplicative measurement-noise scale used for the update.
        action : str, optional
            Caller-defined diagnostic label for the update action.
        """
        result = linear_gaussian_update(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            measurement=measurement,
            measurement_matrix=measurement_matrix,
            meas_noise=meas_noise,
            return_diagnostics=return_diagnostics,
            scale=scale,
            action=action,
        )
        if return_diagnostics:
            new_mean, new_covariance, diagnostics = result
        else:
            new_mean, new_covariance = result
            diagnostics = None
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )
        return diagnostics

    def update_linear_robust(
        self,
        measurement,
        measurement_matrix,
        meas_noise,
        robust_method="student-t",
        *,
        student_t_dof=4.0,
        huber_threshold=2.0,
        gate_threshold=None,
        inflation_alpha=1.0,
        return_diagnostics=False,
    ):
        """Update with a robust linear Gaussian measurement model.

        This method preserves the standard Kalman update behavior when
        ``robust_method`` is ``None`` or ``"none"``. For ``"student-t"`` and
        ``"huber"``, the measurement covariance is adaptively inflated based on
        the innovation's normalized squared Mahalanobis distance before the
        Joseph-form update is applied.
        """
        result = linear_gaussian_update_robust(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            measurement=measurement,
            measurement_matrix=measurement_matrix,
            meas_noise=meas_noise,
            robust_method=robust_method,
            student_t_dof=student_t_dof,
            huber_threshold=huber_threshold,
            gate_threshold=gate_threshold,
            inflation_alpha=inflation_alpha,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            new_mean, new_covariance, diagnostics = result
        else:
            new_mean, new_covariance = result
            diagnostics = None
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )
        return diagnostics

    def update_model_robust(
        self,
        measurement_model,
        measurement,
        robust_method="student-t",
        **kwargs,
    ):
        """Robustly update using a structural linear Gaussian model object."""
        measurement_matrix = _get_required_model_attribute(
            measurement_model,
            "measurement_matrix",
        )
        meas_noise = _get_required_model_attribute(
            measurement_model,
            "meas_noise",
            "measurement_noise_cov",
        )
        return self.update_linear_robust(
            measurement,
            measurement_matrix,
            meas_noise,
            robust_method=robust_method,
            **kwargs,
        )

    def update_model(
        self,
        measurement_model,
        measurement,
        *,
        return_diagnostics=False,
        scale=1.0,
        action="updated",
    ):
        """Update the state with a linear Gaussian measurement model object.

        The model object is consumed structurally. It must expose a
        ``measurement_matrix`` attribute and either ``meas_noise`` or
        ``measurement_noise_cov``.

        Parameters
        ----------
        measurement_model : object
            Linear Gaussian measurement model object compatible with
            :meth:`update_linear`.
        measurement : array-like, shape (m,)
            Measurement vector ``z``.
        return_diagnostics : bool, optional
            If true, return update diagnostics after updating the filter state.
        scale : float, optional
            Multiplicative measurement-noise scale used for the update.
        action : str, optional
            Caller-defined diagnostic label for the update action.
        """
        measurement_matrix = _get_required_model_attribute(
            measurement_model,
            "measurement_matrix",
        )
        meas_noise = _get_required_model_attribute(
            measurement_model,
            "meas_noise",
            "measurement_noise_cov",
        )
        return self.update_linear(
            measurement,
            measurement_matrix,
            meas_noise,
            return_diagnostics=return_diagnostics,
            scale=scale,
            action=action,
        )

    def get_point_estimate(self):
        """Return the posterior mean vector with shape ``(n,)``."""
        return self._filter_state.mu
