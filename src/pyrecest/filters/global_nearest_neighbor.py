# pylint: disable=redefined-builtin,no-name-in-module,no-member,duplicate-code
import warnings

from pyrecest.backend import (
    all,
    any,
    asarray,
    empty,
    full,
    linalg,
    repeat,
    squeeze,
    stack,
    where,
)
from pyrecest.utils.assignment import min_cost_max_cardinality_assignment
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from .abstract_nearest_neighbor_tracker import AbstractNearestNeighborTracker


class GlobalNearestNeighbor(AbstractNearestNeighborTracker):
    """Global nearest-neighbor tracker for linear/Gaussian multitarget tracking.

    Besides the built-in geometric association costs, this implementation can
    optionally fuse an externally computed ``pairwise_cost_matrix`` of shape
    ``(n_targets, n_meas)``. This is useful for domains such as longitudinal
    calcium-imaging cell tracking where association should depend on arbitrary
    pairwise cues like ROI overlap, footprint correlation, or appearance
    embeddings in addition to centroid distance.

    Set ``association_param["maximize_cardinality"]`` to ``True`` when the
    association step should keep as many gated track/measurement pairs as
    possible before minimizing assignment cost. This matches frame-to-frame
    object linking use cases where track birth/death is handled outside the
    GNN update.
    """

    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        default_association_param = {
            "distance_metric_pos": "Mahalanobis",
            "square_dist": True,
            "max_new_tracks": 10,
            "gating_distance_threshold": chi2.ppf(0.999, 2),
            "pairwise_cost_weight": 1.0,
            "maximize_cardinality": False,
        }
        if association_param is None:
            association_param = default_association_param
        else:
            association_param = {
                **default_association_param,
                **association_param,
            }

        super().__init__(
            initial_prior,
            association_param,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

    @staticmethod
    def _validate_pairwise_cost_matrix(pairwise_cost_matrix, n_targets, n_meas):
        if pairwise_cost_matrix is None:
            return None
        pairwise_cost_matrix = asarray(pairwise_cost_matrix, dtype=float)
        if pairwise_cost_matrix.shape != (n_targets, n_meas):
            raise ValueError(
                "pairwise_cost_matrix must have shape "
                f"({n_targets}, {n_meas}), got {pairwise_cost_matrix.shape}."
            )
        return pairwise_cost_matrix

    def _apply_pairwise_cost_matrix(self, dists, pairwise_cost_matrix):
        if pairwise_cost_matrix is None:
            return dists
        pairwise_cost_weight = self.association_param.get("pairwise_cost_weight", 1.0)
        if pairwise_cost_weight == 0.0:
            return dists
        return dists + pairwise_cost_weight * pairwise_cost_matrix

    # pylint: disable=too-many-locals,too-many-positional-arguments
    def find_association(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
        pairwise_cost_matrix=None,
    ):
        """Find the minimum-cost measurement-to-track assignment.

        Parameters
        ----------
        measurements : array-like, shape (dim_meas, n_meas)
            Measurements for the current update step.
        measurement_matrix : array-like
            Linear measurement model.
        cov_mats_meas : array-like
            Measurement covariance matrix or per-measurement covariance tensor.
        warn_on_no_meas_for_track : bool, optional
            Whether to emit a warning when a track remains unassigned.
        pairwise_cost_matrix : array-like, optional
            Additional target/measurement association costs of shape
            ``(n_targets, n_meas)``. These costs are added to the geometric cost
            matrix before running the Hungarian algorithm.
        """
        n_targets = len(self.filter_bank)
        n_meas = measurements.shape[1]

        valid_shared_covariance = cov_mats_meas.ndim == 2
        valid_per_measurement_covariances = (
            cov_mats_meas.ndim == 3 and cov_mats_meas.shape[2] == n_meas
        )
        if not (valid_shared_covariance or valid_per_measurement_covariances):
            raise ValueError(
                "cov_mats_meas must be a matrix or contain one covariance per measurement."
            )
        all_gaussians = [filter.filter_state for filter in self.filter_bank]
        all_means_prior = stack([gaussian.mu for gaussian in all_gaussians], axis=1)
        all_cov_mats_prior = stack([gaussian.C for gaussian in all_gaussians], axis=2)

        if self.association_param["distance_metric_pos"].lower() == "euclidean":
            dists = cdist(
                measurements.T, (measurement_matrix @ all_means_prior).T, "euclidean"
            ).T
        elif self.association_param["distance_metric_pos"].lower() == "mahalanobis":
            dists = empty((n_targets, n_meas))

            all_cov_mat_state_equal = all(
                all_cov_mats_prior
                == repeat(
                    all_cov_mats_prior[:, :, 0][:, :, None],
                    all_cov_mats_prior.shape[2],
                    axis=2,
                )
            )
            all_cov_mat_meas_equal = cov_mats_meas.ndim == 2 or all(
                cov_mats_meas
                == repeat(
                    cov_mats_meas[:, :, 0][:, :, None],
                    cov_mats_meas.shape[2],
                    axis=2,
                )
            )

            if all_cov_mat_meas_equal and all_cov_mat_state_equal:
                shared_cov_mats_meas = (
                    cov_mats_meas if cov_mats_meas.ndim == 2 else cov_mats_meas[:, :, 0]
                )
                curr_cov_mahalanobis = (
                    measurement_matrix
                    @ all_cov_mats_prior[:, :, 0]
                    @ measurement_matrix.T
                    + shared_cov_mats_meas
                )
                dists = cdist(
                    (measurement_matrix @ all_means_prior).T,
                    measurements.T,
                    "mahalanobis",
                    VI=linalg.inv(curr_cov_mahalanobis),
                )
            elif all_cov_mat_meas_equal:
                shared_cov_mats_meas = (
                    cov_mats_meas if cov_mats_meas.ndim == 2 else cov_mats_meas[:, :, 0]
                )
                all_mats_mahalanobis = empty(
                    (
                        measurements.shape[0],
                        measurements.shape[0],
                        all_cov_mats_prior.shape[2],
                    )
                )
                for i in range(all_cov_mats_prior.shape[2]):
                    all_mats_mahalanobis[:, :, i] = (
                        measurement_matrix
                        @ all_cov_mats_prior[:, :, i]
                        @ measurement_matrix.T
                        + shared_cov_mats_meas
                    )
                for i in range(n_targets):
                    dists[i, :] = cdist(
                        (measurement_matrix @ all_means_prior[:, i]).T[None],
                        measurements.T,
                        "mahalanobis",
                        VI=linalg.inv(all_mats_mahalanobis[:, :, i]),
                    )
            else:
                for i in range(n_targets):
                    for j in range(n_meas):
                        curr_cov_mahalanobis = (
                            measurement_matrix
                            @ all_cov_mats_prior[:, :, i]
                            @ measurement_matrix.T
                            + cov_mats_meas[:, :, j]
                        )
                        dists[i, j] = squeeze(
                            cdist(
                                (measurement_matrix @ all_means_prior[:, i]).T[None],
                                measurements[:, j].T[None],
                                "mahalanobis",
                                VI=linalg.inv(curr_cov_mahalanobis),
                            )
                        )
        else:
            raise ValueError("Association scheme not recognized")

        if self.association_param.get("square_dist", False):
            dists = dists * dists

        pairwise_cost_matrix = self._validate_pairwise_cost_matrix(
            pairwise_cost_matrix, n_targets, n_meas
        )
        dists = self._apply_pairwise_cost_matrix(dists, pairwise_cost_matrix)

        if self.association_param.get("maximize_cardinality", False):
            gated_dists = where(
                dists <= self.association_param["gating_distance_threshold"],
                dists,
                float("inf"),
            )
            assignment_result = min_cost_max_cardinality_assignment(gated_dists)
            association = assignment_result["assignment"]
            association = where(association < 0, n_meas, association)
        else:
            # Pad to square and add max_new_tracks rows and columns
            pad_to = max(n_targets, n_meas) + self.association_param["max_new_tracks"]
            association_matrix = full(
                (pad_to, pad_to), self.association_param["gating_distance_threshold"]
            )
            association_matrix[: dists.shape[0], : dists.shape[1]] = dists

            # Use the Hungarian algorithm to find the optimal assignment
            _, col_ind = linear_sum_assignment(association_matrix)

            association = col_ind[:n_targets]

        if warn_on_no_meas_for_track and any(association >= n_meas):
            warnings.warn(
                "GNN: No measurement was within gating threshold for at least one target.",
                stacklevel=2,
            )

        return association

    def update_linear(
        self,
        measurements,
        measurement_matrix,
        covMatsMeas,
        pairwise_cost_matrix=None,
    ):
        """Update the tracker with an optional additional association cost matrix."""
        self._ensure_numpy_backend()
        if len(self.filter_bank) == 0:
            warnings.warn("Currently, there are zero targets")
            return
        self._validate_measurement_matrix_shape(measurement_matrix, measurements)
        association = self.find_association(
            measurements,
            measurement_matrix,
            covMatsMeas,
            pairwise_cost_matrix=pairwise_cost_matrix,
        )
        currMeasCov = covMatsMeas
        for i in range(self.get_number_of_targets()):
            if association[i] < measurements.shape[1]:
                if covMatsMeas.ndim != 2:
                    currMeasCov = covMatsMeas[:, :, association[i]]
                self.filter_bank[i].update_linear(
                    measurements[:, association[i]], measurement_matrix, currMeasCov
                )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
