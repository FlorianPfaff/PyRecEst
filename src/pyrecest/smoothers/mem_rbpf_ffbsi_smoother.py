"""Rao-Blackwellized FFBSi smoother for MEM-RBPF trackers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker
from scipy.special import logsumexp

from .abstract_smoother import AbstractSmoother


@dataclass(frozen=True)
class MEMRBPFForwardRecord:
    """Weighted filtering approximation at one MEM-RBPF measurement time."""

    kinematic_state: Any
    covariance: Any
    theta: Any
    axis_mean: Any
    axis_covariance: Any
    weights: Any
    system_matrix: Any | None = None
    sys_noise: Any | None = None
    axis_system_matrix: Any | None = None
    axis_sys_noise: Any | None = None
    orientation_process_variance: float = 0.0

    @classmethod
    def from_tracker(
        cls, tracker: MEMRBPFTracker, *, use_latest: bool = True
    ) -> "MEMRBPFForwardRecord":
        """Create a record from a :class:`MEMRBPFTracker` instance.

        Newer trackers may expose ``get_filtering_state`` to return a stored
        post-update/pre-resampling snapshot. For older trackers this falls back
        to the current tracker arrays.
        """
        if hasattr(tracker, "get_filtering_state"):
            return cls.from_mapping(tracker.get_filtering_state(use_latest=use_latest))
        return cls(
            kinematic_state=np.asarray(tracker.kinematic_state, dtype=float),
            covariance=np.asarray(tracker.covariance, dtype=float),
            theta=np.asarray(tracker.theta, dtype=float),
            axis_mean=np.asarray(tracker.axis, dtype=float),
            axis_covariance=np.asarray(tracker.axis_covariances, dtype=float),
            weights=np.asarray(tracker.weights, dtype=float),
            system_matrix=np.asarray(tracker.system_matrix, dtype=float),
            sys_noise=np.asarray(tracker.sys_noise, dtype=float),
            axis_system_matrix=getattr(tracker, "axis_system_matrix", np.eye(2)),
            axis_sys_noise=np.asarray(tracker.axis_sys_noise, dtype=float),
            orientation_process_variance=float(tracker.orientation_process_variance),
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MEMRBPFForwardRecord":
        """Create a record from a mapping of MEM-RBPF arrays."""
        axis_mean = mapping.get("axis_mean", mapping.get("axis"))
        axis_covariance = mapping.get(
            "axis_covariance", mapping.get("axis_covariances")
        )
        if axis_mean is None or axis_covariance is None:
            raise ValueError("mapping must contain axis/axis_mean and axis_covariance")
        return cls(
            kinematic_state=mapping["kinematic_state"],
            covariance=mapping["covariance"],
            theta=mapping["theta"],
            axis_mean=axis_mean,
            axis_covariance=axis_covariance,
            weights=mapping["weights"],
            system_matrix=mapping.get("system_matrix"),
            sys_noise=mapping.get("sys_noise"),
            axis_system_matrix=mapping.get("axis_system_matrix"),
            axis_sys_noise=mapping.get("axis_sys_noise"),
            orientation_process_variance=float(
                mapping.get("orientation_process_variance", 0.0)
            ),
        )

    def copy(self) -> "MEMRBPFForwardRecord":
        """Return a detached copy of this record."""
        return MEMRBPFForwardRecord(
            np.asarray(self.kinematic_state, dtype=float).copy(),
            np.asarray(self.covariance, dtype=float).copy(),
            np.asarray(self.theta, dtype=float).copy(),
            np.asarray(self.axis_mean, dtype=float).copy(),
            np.asarray(self.axis_covariance, dtype=float).copy(),
            np.asarray(self.weights, dtype=float).copy(),
            (
                None
                if self.system_matrix is None
                else np.asarray(self.system_matrix, dtype=float).copy()
            ),
            (
                None
                if self.sys_noise is None
                else np.asarray(self.sys_noise, dtype=float).copy()
            ),
            (
                None
                if self.axis_system_matrix is None
                else np.asarray(self.axis_system_matrix, dtype=float).copy()
            ),
            (
                None
                if self.axis_sys_noise is None
                else np.asarray(self.axis_sys_noise, dtype=float).copy()
            ),
            float(self.orientation_process_variance),
        )

    @property
    def n_particles(self) -> int:
        return int(np.asarray(self.theta).reshape(-1).shape[0])

    @property
    def state_dim(self) -> int:
        return int(np.asarray(self.kinematic_state).reshape(-1).shape[0])

    @property
    def axis_dim(self) -> int:
        return int(np.asarray(self.axis_mean).reshape(self.n_particles, -1).shape[1])


@dataclass(frozen=True)
class RBFFBSiResult:
    """Output of :class:`MEMRBPFFFBSiSmoother`."""

    states: np.ndarray
    kinematic_mean: np.ndarray
    kinematic_covariance: np.ndarray
    theta_samples: np.ndarray
    axis_samples: np.ndarray
    index_samples: np.ndarray
    sample_states: np.ndarray

    @property
    def kinematic_cov(self) -> np.ndarray:
        """Backward-compatible alias for ``kinematic_covariance``."""
        return self.kinematic_covariance


class MEMRBPFFFBSiSmoother(AbstractSmoother):
    """Fixed-interval Rao-Blackwellized FFBSi smoother for MEM-RBPF records.

    The MEM-RBPF representation consists of one global kinematic Gaussian and a
    weighted orientation-particle approximation with conditional Gaussian
    semi-axis states. This smoother runs RTS smoothing for the global kinematic
    Gaussian and backward-simulates orientation/semi-axis trajectories.
    """

    def __init__(
        self,
        n_trajectories: int | None = None,
        sample_axis: bool = True,
        angle_wrap_terms: int = 2,
        axis_floor: float = 1e-9,
    ):
        if n_trajectories is not None and int(n_trajectories) <= 0:
            raise ValueError("n_trajectories must be positive")
        if int(angle_wrap_terms) < 0:
            raise ValueError("angle_wrap_terms must be non-negative")
        if axis_floor <= 0.0:
            raise ValueError("axis_floor must be positive")
        self.n_trajectories = None if n_trajectories is None else int(n_trajectories)
        self.sample_axis = bool(sample_axis)
        self.angle_wrap_terms = int(angle_wrap_terms)
        self.axis_floor = float(axis_floor)

    @classmethod
    def _as_record(cls, record) -> MEMRBPFForwardRecord:
        if isinstance(record, MEMRBPFForwardRecord):
            return record.copy()
        if isinstance(record, MEMRBPFTracker):
            return MEMRBPFForwardRecord.from_tracker(record)
        if isinstance(record, Mapping):
            return MEMRBPFForwardRecord.from_mapping(record)
        if isinstance(record, tuple) and len(record) in (5, 6):
            kwargs = {} if len(record) == 5 else dict(record[5])
            return MEMRBPFForwardRecord(
                record[0], record[1], record[2], record[3], record[4], **kwargs
            )
        raise ValueError(
            "record must be a MEMRBPFForwardRecord, MEMRBPFTracker, mapping, or tuple"
        )

    @classmethod
    def _normalize_records(cls, records: Sequence) -> list[MEMRBPFForwardRecord]:
        normalized = [cls._as_record(record) for record in records]
        if not normalized:
            raise ValueError("at least one forward record is required")
        return normalized

    @staticmethod
    def _prepare_rng(rng):
        if rng is None:
            return np.random.default_rng()
        if isinstance(rng, Integral):
            return np.random.default_rng(int(rng))
        if isinstance(rng, np.random.Generator):
            return rng
        raise TypeError(
            "rng must be None, an integer seed, or a numpy.random.Generator"
        )

    @staticmethod
    def _symmetrize(matrix):
        matrix = np.asarray(matrix, dtype=float)
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _nearest_psd(cls, matrix, floor: float = 1e-10):
        matrix = cls._symmetrize(matrix)
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.maximum(vals, floor)
        return (vecs * vals) @ vecs.T

    @staticmethod
    def _safe_probs(weights):
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.maximum(weights, 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            return np.full(weights.shape, 1.0 / weights.size)
        return weights / total

    @staticmethod
    def _normalize_log_probs(log_probs):
        log_probs = np.asarray(log_probs, dtype=float)
        finite = np.isfinite(log_probs)
        if not np.any(finite):
            return np.full(log_probs.shape, 1.0 / log_probs.size)
        safe = np.where(finite, log_probs, -np.inf)
        probs = np.exp(safe - logsumexp(safe))
        return probs / np.sum(probs)

    @staticmethod
    def _default_matrix(record, attr_name: str, dim: int, default_scale: float = 0.0):
        value = getattr(record, attr_name)
        if value is None:
            return default_scale * np.eye(dim)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(dim)
        return arr

    def _rts_smooth_kinematics(self, records: list[MEMRBPFForwardRecord]):
        xs = np.stack(
            [np.asarray(r.kinematic_state, dtype=float).reshape(-1) for r in records]
        )
        ps = np.stack([self._nearest_psd(r.covariance) for r in records])
        smoothed_x = xs.copy()
        smoothed_p = ps.copy()
        for time_idx in range(len(records) - 2, -1, -1):
            dim = xs[time_idx].shape[0]
            system_matrix = self._default_matrix(
                records[time_idx], "system_matrix", dim, 1.0
            )
            sys_noise = self._default_matrix(records[time_idx], "sys_noise", dim, 0.0)
            pred_x = system_matrix @ xs[time_idx]
            pred_p = self._nearest_psd(
                system_matrix @ ps[time_idx] @ system_matrix.T + sys_noise
            )
            gain = ps[time_idx] @ system_matrix.T @ np.linalg.pinv(pred_p)
            smoothed_x[time_idx] = xs[time_idx] + gain @ (
                smoothed_x[time_idx + 1] - pred_x
            )
            smoothed_p[time_idx] = self._nearest_psd(
                ps[time_idx] + gain @ (smoothed_p[time_idx + 1] - pred_p) @ gain.T
            )
        return smoothed_x, smoothed_p

    @staticmethod
    def _wrapped_normal_logpdf_many(delta, variance: float, period: float, terms: int):
        variance = max(float(variance), 1e-12)
        delta = np.asarray(delta, dtype=float)
        delta = (delta + 0.5 * period) % period - 0.5 * period
        shifts = np.arange(-terms, terms + 1, dtype=float) * period
        values = -0.5 * (
            (delta[:, None] + shifts[None, :]) ** 2 / variance
            + np.log(2.0 * np.pi * variance)
        )
        return logsumexp(values, axis=1)

    @classmethod
    def _log_mvn_many(cls, x, means, covariances):
        x = np.asarray(x, dtype=float).reshape(-1)
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)
        out = np.empty(means.shape[0], dtype=float)
        const = x.size * np.log(2.0 * np.pi)
        for idx in range(means.shape[0]):
            cov = cls._nearest_psd(covariances[idx])
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0.0 or not np.isfinite(logdet):
                out[idx] = -np.inf
                continue
            diff = x - means[idx]
            out[idx] = -0.5 * (const + logdet + diff @ np.linalg.pinv(cov) @ diff)
        return out

    @classmethod
    def _draw_or_mean_gaussian(cls, rng, mean, covariance, *, sample: bool):
        mean = np.asarray(mean, dtype=float).reshape(-1)
        if not sample:
            return mean.copy()
        return rng.multivariate_normal(mean, cls._nearest_psd(covariance))

    def _axis_prediction(self, record: MEMRBPFForwardRecord):
        axis = np.asarray(record.axis_mean, dtype=float).reshape(
            record.n_particles, record.axis_dim
        )
        cov = np.asarray(record.axis_covariance, dtype=float).reshape(
            record.n_particles, record.axis_dim, record.axis_dim
        )
        system_matrix = self._default_matrix(
            record, "axis_system_matrix", record.axis_dim, 1.0
        )
        sys_noise = self._default_matrix(record, "axis_sys_noise", record.axis_dim, 0.0)
        pred_axis = axis @ system_matrix.T
        pred_cov = system_matrix @ cov @ system_matrix.T + sys_noise.reshape(
            1, record.axis_dim, record.axis_dim
        )
        pred_cov = np.stack([self._nearest_psd(curr) for curr in pred_cov])
        return axis, cov, pred_axis, pred_cov, system_matrix

    def _axis_backward_kernel(self, mean_t, cov_t, axis_next, system_matrix, sys_noise):
        cov_t = self._nearest_psd(cov_t)
        pred_mean = system_matrix @ mean_t
        pred_cov = self._nearest_psd(
            system_matrix @ cov_t @ system_matrix.T + sys_noise
        )
        gain = cov_t @ system_matrix.T @ np.linalg.pinv(pred_cov)
        mean = mean_t + gain @ (axis_next - pred_mean)
        cov = cov_t - gain @ pred_cov @ gain.T
        return mean, self._nearest_psd(cov)

    @staticmethod
    def _rotation(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _sample_states_from_components(
        self, kinematic_mean, theta_samples, axis_samples, full_axis_lengths
    ):
        n_samples, n_steps = theta_samples.shape
        axis_scale = 2.0 if full_axis_lengths else 1.0
        state_dim = kinematic_mean.shape[1] + 3
        sample_states = np.zeros((n_samples, n_steps, state_dim), dtype=float)
        sample_states[:, :, : kinematic_mean.shape[1]] = kinematic_mean[
            np.newaxis, :, :
        ]
        sample_states[:, :, kinematic_mean.shape[1]] = theta_samples % (2.0 * np.pi)
        sample_states[:, :, -2:] = axis_scale * np.maximum(
            np.abs(axis_samples), self.axis_floor
        )
        return sample_states

    def _point_estimate_from_samples(
        self, kinematic_mean, theta_samples, axis_samples, full_axis_lengths
    ):
        n_steps = theta_samples.shape[1]
        state_dim = kinematic_mean.shape[1] + 3
        states = np.zeros((n_steps, state_dim), dtype=float)
        states[:, : kinematic_mean.shape[1]] = kinematic_mean
        axis_scale = 2.0 if full_axis_lengths else 1.0
        for time_idx in range(n_steps):
            semi_axes = np.maximum(
                np.abs(axis_samples[:, time_idx, :]), self.axis_floor
            )
            mean_extent = np.zeros((2, 2), dtype=float)
            for theta, axes in zip(theta_samples[:, time_idx], semi_axes):
                rot = self._rotation(theta)
                mean_extent += rot @ np.diag(axes**2) @ rot.T
            mean_extent /= semi_axes.shape[0]
            mean_extent = self._symmetrize(mean_extent)
            vals, vecs = np.linalg.eigh(mean_extent)
            order = np.argsort(vals)[::-1]
            vals = np.maximum(vals[order], 0.0)
            vecs = vecs[:, order]
            states[time_idx, kinematic_mean.shape[1]] = np.arctan2(
                vecs[1, 0], vecs[0, 0]
            ) % (2.0 * np.pi)
            states[time_idx, -2:] = axis_scale * np.maximum(
                np.sqrt(vals[:2]), self.axis_floor
            )
        return states

    def smooth(
        self,
        records: Sequence,
        rng=None,
        *,
        n_trajectories: int | None = None,
        sample_axis: bool | None = None,
        angle_wrap_terms: int | None = None,
        full_axis_lengths: bool = True,
    ) -> RBFFBSiResult:
        """Run fixed-interval RB-FFBSi over explicit MEM-RBPF records."""
        record_list = self._normalize_records(records)
        rng = self._prepare_rng(rng)
        n_steps = len(record_list)
        final_particles = record_list[-1].n_particles
        n_samples = (
            self.n_trajectories if n_trajectories is None else int(n_trajectories)
        )
        if n_samples is None:
            n_samples = final_particles
        if n_samples <= 0:
            raise ValueError("n_trajectories must be positive")
        do_sample_axis = self.sample_axis if sample_axis is None else bool(sample_axis)
        terms = (
            self.angle_wrap_terms if angle_wrap_terms is None else int(angle_wrap_terms)
        )

        kinematic_mean, kinematic_covariance = self._rts_smooth_kinematics(record_list)
        theta_samples = np.zeros((n_samples, n_steps), dtype=float)
        axis_samples = np.zeros(
            (n_samples, n_steps, record_list[-1].axis_dim), dtype=float
        )
        index_samples = np.zeros((n_samples, n_steps), dtype=int)

        final_probs = self._safe_probs(record_list[-1].weights)
        final_indices = rng.choice(
            final_particles, size=n_samples, replace=True, p=final_probs
        )
        index_samples[:, -1] = final_indices
        theta_samples[:, -1] = np.asarray(record_list[-1].theta, dtype=float)[
            final_indices
        ]
        final_axis = np.asarray(record_list[-1].axis_mean, dtype=float).reshape(
            final_particles, -1
        )
        final_cov = np.asarray(record_list[-1].axis_covariance, dtype=float).reshape(
            final_particles, record_list[-1].axis_dim, record_list[-1].axis_dim
        )
        for sample_idx, particle_idx in enumerate(final_indices):
            axis_samples[sample_idx, -1] = self._draw_or_mean_gaussian(
                rng,
                final_axis[particle_idx],
                final_cov[particle_idx],
                sample=do_sample_axis,
            )

        for time_idx in range(n_steps - 2, -1, -1):
            record = record_list[time_idx]
            weights = self._safe_probs(record.weights)
            axis, cov, pred_axis, pred_cov, axis_system = self._axis_prediction(record)
            axis_noise = self._default_matrix(
                record, "axis_sys_noise", record.axis_dim, 0.0
            )
            for sample_idx in range(n_samples):
                log_weight = np.log(np.maximum(weights, np.finfo(float).tiny))
                log_angle = self._wrapped_normal_logpdf_many(
                    theta_samples[sample_idx, time_idx + 1]
                    - np.asarray(record.theta, dtype=float),
                    record.orientation_process_variance,
                    2.0 * np.pi,
                    terms,
                )
                log_axis = self._log_mvn_many(
                    axis_samples[sample_idx, time_idx + 1], pred_axis, pred_cov
                )
                probs = self._normalize_log_probs(log_weight + log_angle + log_axis)
                particle_idx = int(rng.choice(record.n_particles, p=probs))
                index_samples[sample_idx, time_idx] = particle_idx
                theta_samples[sample_idx, time_idx] = np.asarray(
                    record.theta, dtype=float
                )[particle_idx]
                mean, cov_s = self._axis_backward_kernel(
                    axis[particle_idx],
                    cov[particle_idx],
                    axis_samples[sample_idx, time_idx + 1],
                    axis_system,
                    axis_noise,
                )
                axis_samples[sample_idx, time_idx] = self._draw_or_mean_gaussian(
                    rng, mean, cov_s, sample=do_sample_axis
                )

        sample_states = self._sample_states_from_components(
            kinematic_mean, theta_samples, axis_samples, full_axis_lengths
        )
        states = self._point_estimate_from_samples(
            kinematic_mean, theta_samples, axis_samples, full_axis_lengths
        )
        return RBFFBSiResult(
            states=states,
            kinematic_mean=kinematic_mean,
            kinematic_covariance=kinematic_covariance,
            theta_samples=theta_samples,
            axis_samples=axis_samples,
            index_samples=index_samples,
            sample_states=sample_states,
        )


MEMRBPF_FFBSiSmoother = MEMRBPFFFBSiSmoother
RBFFBSiSmoother = MEMRBPFFFBSiSmoother
