"""
SE(2) Unscented Kalman Filter using dual-quaternion state representation.

Conventions:
    ``_dual_quaternion_multiply(a, b)`` denotes the ordered SE(2)
    composition ``a [⊕] b``.  Prediction uses right-multiplicative process
    increments, i.e. ``x_{t+1} = x_t [⊕] v``.  The identity measurement model
    uses the matching right-perturbation convention ``z = x [⊕] v``.

Reference:
    A Stochastic Filter for Planar Rigid-Body Motions,
    Igor Gilitschenski, Gerhard Kurz, and Uwe D. Hanebeck,
    Proceedings of the 2015 IEEE International Conference on Multisensor
    Fusion and Integration for Intelligent Systems (MFI),
    San Diego, USA, 2015.
"""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    allclose,
    array,
    asarray,
    column_stack,
    concatenate,
    isclose,
    isfinite,
    linalg,
    mean,
    transpose,
    vstack,
    zeros,
)
from pyrecest.distributions import GaussianDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import SE2FilterMixin


def _to_python_bool(value):
    """Convert scalar backend booleans to Python bools for validation."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _validate_se2_gaussian(distribution, role):
    if not isinstance(distribution, GaussianDistribution):
        raise ValueError(f"{role} must be a GaussianDistribution.")

    mu = asarray(distribution.mu)
    covariance = asarray(distribution.C)
    if mu.shape != (4,):
        raise ValueError(f"{role} mean must be a 4-D vector.")
    if covariance.shape != (4, 4):
        raise ValueError(f"{role} covariance must be 4x4.")
    if not _to_python_bool(backend_all(isfinite(mu))):
        raise ValueError(f"{role} mean must be finite.")
    if not _to_python_bool(backend_all(isfinite(covariance))):
        raise ValueError(f"{role} covariance must be finite.")

    rotation_norm = linalg.norm(mu[0:2])
    if not _to_python_bool(isfinite(rotation_norm)):
        raise ValueError(f"{role} rotation part must have a finite norm.")
    if not _to_python_bool(isclose(rotation_norm, 1.0)):
        raise ValueError(f"{role} rotation part must be normalised.")
    if not _to_python_bool(allclose(covariance, transpose(covariance))):
        raise ValueError(f"{role} covariance must be symmetric.")
    if not _to_python_bool(backend_all(linalg.eigvalsh(covariance) > 0.0)):
        raise ValueError(f"{role} covariance must be positive definite.")
    return mu, covariance


def _validate_se2_measurement(z):
    measurement = asarray(z).ravel()
    if measurement.shape != (4,):
        raise ValueError("measurement z must be a 4-D vector.")
    if not _to_python_bool(backend_all(isfinite(measurement))):
        raise ValueError("measurement z must be finite.")

    rotation_norm = linalg.norm(measurement[0:2])
    if not _to_python_bool(isfinite(rotation_norm)):
        raise ValueError("measurement z rotation part must have a finite norm.")
    if not _to_python_bool(isclose(rotation_norm, 1.0)):
        raise ValueError("measurement z rotation part must be normalised.")
    return measurement


def _dual_quaternion_multiply(dq1, dq2):
    """Multiply two SE(2) dual quaternions in ordered composition order.

    Each dual quaternion is represented as a length-4 array
    ``[q1, q2, d1, d2]`` where ``[q1, q2]`` is the unit-norm
    rotation part and ``[d1, d2]`` is the dual (translation) part.

    The formula is derived from the 4×4 matrix representation used in
    the libDirectional SE2 class::

        M(dq) = [[ q1,  q2,  0,  0 ],
                 [-q2,  q1,  0,  0 ],
                 [-d1,  d2, q1, -q2],
                 [-d2, -d1, q2,  q1]]

    and ``dq_product = M(dq1) @ M(dq2)``.  In this module, that ordered
    product is denoted ``dq1 [⊕] dq2``.

    Parameters
    ----------
    dq1, dq2 : array_like, shape (4,)
        SE(2) dual quaternions.

    Returns
    -------
    array, shape (4,)
        Ordered product ``dq1 [⊕] dq2``.
    """
    a, b, c, d = dq1[0], dq1[1], dq1[2], dq1[3]
    e, f, g, h = dq2[0], dq2[1], dq2[2], dq2[3]
    return array(
        [
            a * e - b * f,
            b * e + a * f,
            c * e + d * f + a * g - b * h,
            d * e - c * f + b * g + a * h,
        ]
    )


class SE2UKF(AbstractFilter, SE2FilterMixin):
    """Unscented Kalman Filter for planar rigid-body motion on SE(2).

    The state is represented as a :class:`~pyrecest.distributions.GaussianDistribution`
    over the 4-D dual-quaternion embedding of SE(2).  The first two
    entries of the mean encode the rotation (and must satisfy
    ``||mu[0:2]|| == 1``); the last two entries encode the translation.

    Ordered composition follows :func:`_dual_quaternion_multiply`: ``a [⊕] b``
    means the module-level product ``_dual_quaternion_multiply(a, b)``.
    Prediction therefore composes state samples with process increments as
    ``x_t [⊕] v``.

    Reference:
        A Stochastic Filter for Planar Rigid-Body Motions,
        Igor Gilitschenski, Gerhard Kurz, and Uwe D. Hanebeck,
        IEEE MFI 2015.
    """

    def __init__(self):
        # Initialise with a trivial identity-like state.
        initial_state = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            array(
                [
                    [0.25, 0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property
    # ------------------------------------------------------------------

    @property
    def filter_state(self) -> GaussianDistribution:
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state: GaussianDistribution):
        _validate_se2_gaussian(new_state, "filter_state")
        self._filter_state = new_state

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, gauss_sys: GaussianDistribution):
        """Predict with a right-multiplicative increment model on SE(2).

        The motion model is::

            x_{t+1} = x_t [⊕] v

        where ``v`` is the process increment/noise.  The mean of ``gauss_sys``
        encodes the deterministic dual-quaternion increment; its covariance
        encodes uncertainty around that increment.

        Parameters
        ----------
        gauss_sys : GaussianDistribution
            System noise/increment distribution.  Must have a 4-D mean (first
            two entries normalised) and a 4×4 covariance.
        """
        # pylint: disable=too-many-locals
        mu_sys, C_sys = _validate_se2_gaussian(gauss_sys, "system noise")

        mu = asarray(self._filter_state.mu)
        C = asarray(self._filter_state.C)

        # --- State sigma points: [mu, mu ± 2*L[:,k]] for k=0..3 (9 pts) ---
        L = linalg.cholesky(C)  # 4×4 lower-triangular
        cols = [mu]
        for k in range(4):
            cols.append(mu + 2.0 * L[:, k])
        for k in range(4):
            cols.append(mu - 2.0 * L[:, k])
        state_samples = column_stack(cols)  # 4×9

        # Normalise rotation part of state sigma points.
        norms = linalg.norm(state_samples[0:2, :], axis=0)
        state_samples = concatenate(
            [state_samples[0:2, :] / norms[None, :], state_samples[2:, :]], axis=0
        )

        # --- Noise sigma points: [mu_sys, mu_sys ± L_n[:,k]] for k=0..3 (9 pts) ---
        L_n = linalg.cholesky(4.0 * C_sys)  # 4×4
        n_cols = [mu_sys]
        for k in range(4):
            n_cols.append(mu_sys + L_n[:, k])
        for k in range(4):
            n_cols.append(mu_sys - L_n[:, k])
        noise_samples = column_stack(n_cols)  # 4×9

        # Normalise rotation part of noise sigma points.
        norms = linalg.norm(noise_samples[0:2, :], axis=0)
        noise_samples = concatenate(
            [noise_samples[0:2, :] / norms[None, :], noise_samples[2:, :]], axis=0
        )

        # --- Predicted samples: all 81 = 9×9 state [⊕] noise combinations ---
        pred_cols = []
        for i in range(9):
            for j in range(9):
                pred_cols.append(
                    _dual_quaternion_multiply(state_samples[:, i], noise_samples[:, j])
                )
        pred_samples = column_stack(pred_cols)  # 4×81

        # Mean and covariance from the propagated prediction samples.
        # Store C as an actual covariance, not as the uncentered second moment
        # E[x x^T].  GaussianDistribution.C is used elsewhere as a covariance,
        # so it must be centered around the stored mean.
        new_mu = mean(pred_samples, axis=1)
        new_mu = concatenate([new_mu[0:2] / linalg.norm(new_mu[0:2]), new_mu[2:]])

        deviations = pred_samples - new_mu[:, None]
        CP = deviations @ deviations.T / pred_samples.shape[1]
        new_C = (CP + CP.T) / 2.0  # symmetrise

        self._filter_state = GaussianDistribution(array(new_mu), array(new_C))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, gauss_meas: GaussianDistribution, z):
        """Incorporate a dual-quaternion measurement.

        The measurement model is::

            z = x [⊕] v

        where ``v`` is the measurement noise, interpreted as a right-hand
        perturbation under the same ordered composition convention used for
        prediction.

        Parameters
        ----------
        gauss_meas : GaussianDistribution
            Measurement noise distribution.  Must have a 4-D mean (first
            two entries normalised) and a 4×4 covariance.
        z : array_like, shape (4,)
            Measurement in dual-quaternion representation.
        """
        # pylint: disable=too-many-locals
        mu_meas, C_meas = _validate_se2_gaussian(gauss_meas, "measurement noise")
        z = _validate_se2_measurement(z)

        mu = asarray(self._filter_state.mu)
        C = asarray(self._filter_state.C)

        # Take the closer antipodal representative.
        if linalg.norm(z - mu) > linalg.norm(-z - mu):
            z = -z

        # --- Augmented state and covariance ---
        # Concatenate state mean (4-D) and noise mean (4-D) into an 8-D augmented state.
        x_aug = concatenate([mu, mu_meas])  # 8-D augmented state
        C_aug = concatenate(
            [
                concatenate([C, zeros((4, 4))], axis=1),
                concatenate([zeros((4, 4)), C_meas], axis=1),
            ],
            axis=0,
        )  # 8×8

        # --- Augmented sigma points: [x_aug, x_aug ± L_aug[:,k]] for k=0..7 (17 pts) ---
        L_aug = linalg.cholesky(8.0 * C_aug)  # 8×8
        aug_cols = [x_aug]
        for k in range(8):
            aug_cols.append(x_aug + L_aug[:, k])
        for k in range(8):
            aug_cols.append(x_aug - L_aug[:, k])
        aug_samples = column_stack(aug_cols)  # 8×17

        # Normalise rotation part of state sigma points (rows 0–1).
        norms = linalg.norm(aug_samples[0:2, :], axis=0)
        aug_samples = concatenate(
            [aug_samples[0:2, :] / norms[None, :], aug_samples[2:, :]], axis=0
        )

        # Extract and normalise the noise-rotation part (rows 4–5).
        noise_rot = aug_samples[4:6, :]
        norms_n = linalg.norm(noise_rot, axis=0)
        # Build the full normalised noise vectors (4-D each column).
        norm_noise = vstack([noise_rot / norms_n[None, :], aug_samples[6:8, :]])  # 4×17

        # --- Apply measurement function: z_i = state_i [⊕] noise_i ---
        meas_cols = []
        for i in range(17):
            meas_cols.append(
                _dual_quaternion_multiply(aug_samples[0:4, i], norm_noise[:, i])
            )
        meas_samples = column_stack(meas_cols)  # 4×17

        # --- Covariance matrices ---
        meas_mean = mean(meas_samples, axis=1)
        meas_dev = meas_samples - meas_mean[:, None]  # 4×17

        # Cross-covariance: (aug_samples - x_aug) * meas_dev' / 17
        # Because the column sum of meas_dev is zero (it is mean-centred),
        # using aug_samples directly is equivalent to using centred aug_samples.
        cross = aug_samples @ meas_dev.T / 17.0  # 8×4 (P_XY)
        P_Y = meas_dev @ meas_dev.T / 17.0  # 4×4 (innovation covariance)

        # --- Kalman update ---
        K = cross @ linalg.inv(P_Y)  # 8×4
        x_aug_upd = x_aug + K @ (z - meas_mean)  # 8-D
        C_aug_upd = C_aug - K @ P_Y @ K.T  # 8×8

        new_mu = x_aug_upd[0:4]
        new_C = C_aug_upd[0:4, 0:4]
        new_C = (new_C + new_C.T) / 2.0  # symmetrise

        # Renormalise rotation part of the mean.
        new_mu = concatenate([new_mu[0:2] / linalg.norm(new_mu[0:2]), new_mu[2:]])

        self._filter_state = GaussianDistribution(array(new_mu), array(new_C))

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the mean of the current state estimate."""
        return self._filter_state.mu
