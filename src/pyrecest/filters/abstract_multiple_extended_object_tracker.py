"""Abstract interface for multiple extended-object trackers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from pyrecest.backend import asarray, concatenate, empty, stack

from .abstract_multitarget_tracker import AbstractMultitargetTracker


@dataclass
class ExtendedObjectEstimate:  # pylint: disable=too-many-instance-attributes
    """Extracted estimate of one extended object.

    The extent field is intentionally untyped because different EOT models use
    different shape parameterizations, e.g., random matrices, star-convex radial
    functions, polygons, or subobject collections.
    """

    label: Optional[Hashable]
    kinematics: Any
    extent: Any
    existence_probability: Optional[float] = None
    measurement_rate: Optional[float] = None
    weight: Optional[float] = None
    covariance: Any = None
    extent_uncertainty: Any = None
    status: Optional[str] = None
    source_component: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtendedObjectAssociationResult:
    """Association result for one extended-object measurement scan.

    In multiple extended-object tracking, one object can generate multiple
    detections in one scan. Therefore, associations are represented as object to
    measurement-cell assignments rather than one-to-one target/measurement pairs.
    """

    object_to_measurement_indices: dict[Optional[Hashable], list[int]] = field(
        default_factory=dict
    )
    clutter_indices: list[int] = field(default_factory=list)
    birth_cell_indices: list[list[int]] = field(default_factory=list)
    global_hypotheses: Any = None
    selected_partition: Optional[list[list[int]]] = None
    log_likelihood: Optional[float] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultipleExtendedObjectStepResult:  # pylint: disable=too-many-instance-attributes
    """Summary of one predict/update step of a MEOT tracker."""

    estimates: list[ExtendedObjectEstimate] = field(default_factory=list)
    association: Optional[ExtendedObjectAssociationResult] = None
    created_labels: list[Optional[Hashable]] = field(default_factory=list)
    deleted_labels: list[Optional[Hashable]] = field(default_factory=list)
    confirmed_labels: list[Optional[Hashable]] = field(default_factory=list)
    cardinality_distribution: Any = None
    expected_number_of_objects: Optional[float] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-instance-attributes
class AbstractMultipleExtendedObjectTracker(AbstractMultitargetTracker):
    """Base class for trackers of multiple extended objects.

    The canonical high-level output is a list of :class:`ExtendedObjectEstimate`
    instances. ``get_point_estimate`` remains available for compatibility with
    :class:`AbstractMultitargetTracker` and returns one vectorized estimate per
    extracted object.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        extraction_threshold: Optional[float] = None,
        max_extracted_objects: Optional[int] = None,
        extract_confirmed_only: bool = True,
        log_prior_estimates: bool = True,
        log_posterior_estimates: bool = True,
        log_prior_extents: bool = False,
        log_posterior_extents: bool = False,
        log_prior_measurement_rates: bool = False,
        log_posterior_measurement_rates: bool = False,
        log_cardinality: bool = False,
        log_associations: bool = False,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

        self.extraction_threshold = extraction_threshold
        self.max_extracted_objects = max_extracted_objects
        self.extract_confirmed_only = bool(extract_confirmed_only)

        self.log_prior_extents = bool(log_prior_extents)
        self.log_posterior_extents = bool(log_posterior_extents)
        self.log_prior_measurement_rates = bool(log_prior_measurement_rates)
        self.log_posterior_measurement_rates = bool(log_posterior_measurement_rates)
        self.log_cardinality = bool(log_cardinality)
        self.log_associations = bool(log_associations)

        self.prior_measurement_rates_over_time = None
        self.posterior_measurement_rates_over_time = None
        self.cardinality_over_time = None
        self.associations_over_time = None
        self.latest_step_result: Optional[MultipleExtendedObjectStepResult] = None

        if self.log_prior_extents:
            self.prior_extents_over_time = self.history.register(
                "prior_extents", pad_with_nan=True
            )
        if self.log_posterior_extents:
            self.posterior_extents_over_time = self.history.register(
                "posterior_extents", pad_with_nan=True
            )
        if self.log_prior_measurement_rates:
            self.prior_measurement_rates_over_time = self.history.register(
                "prior_measurement_rates", pad_with_nan=True
            )
        if self.log_posterior_measurement_rates:
            self.posterior_measurement_rates_over_time = self.history.register(
                "posterior_measurement_rates", pad_with_nan=True
            )
        if self.log_cardinality:
            self.cardinality_over_time = self.history.register("cardinality")
        if self.log_associations:
            self.associations_over_time = self.history.register("associations")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @abstractmethod
    def predict(
        self,
        dt: Optional[float] = None,
        dynamic_model: Any = None,
        process_noise: Any = None,
        survival_probability: Any = None,
        birth_model: Any = None,
        **kwargs,
    ) -> None:
        """Propagate the multi-object posterior one time step."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @abstractmethod
    def update(
        self,
        measurements: Any,
        measurement_model: Any = None,
        meas_noise_cov: Any = None,
        detection_probability: Any = None,
        clutter_model: Any = None,
        measurement_partitions: Any = None,
        sensor_state: Any = None,
        **kwargs,
    ) -> MultipleExtendedObjectStepResult:
        """Update from one complete scan of measurements.

        ``measurement_partitions`` may contain precomputed cells or alternative
        partitions for trackers that separate partitioning from association.
        """

    def step(
        self,
        measurements: Any,
        predict_kwargs: Optional[dict[str, Any]] = None,
        update_kwargs: Optional[dict[str, Any]] = None,
    ) -> MultipleExtendedObjectStepResult:
        """Run a complete predict/update step and record enabled histories."""
        predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
        update_kwargs = {} if update_kwargs is None else dict(update_kwargs)

        self.predict(**predict_kwargs)
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extents()
        if self.log_prior_measurement_rates:
            self.store_prior_measurement_rates()

        result = self.update(measurements, **update_kwargs)
        if result is None:
            result = MultipleExtendedObjectStepResult(
                estimates=self.get_object_estimates()
            )
        elif not result.estimates:
            result.estimates = self.get_object_estimates()

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()
        if self.log_posterior_measurement_rates:
            self.store_posterior_measurement_rates()
        if self.log_cardinality:
            self.cardinality_over_time = self.record_history(
                "cardinality", self._get_cardinality_history_value(), copy_value=True
            )
        if self.log_associations and result.association is not None:
            self.associations_over_time = self.record_history(
                "associations", result.association, copy_value=True
            )

        self.latest_step_result = result
        return result

    @abstractmethod
    def get_object_estimates(
        self,
        extraction_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        confirmed_only: Optional[bool] = None,
    ) -> list[ExtendedObjectEstimate]:
        """Return extracted extended-object estimates."""

    def get_number_of_targets(self, confirmed_only: Optional[bool] = None) -> int:
        """Return the number of currently extracted objects."""
        return len(self.get_object_estimates(confirmed_only=confirmed_only))

    def get_track_labels(
        self,
        extraction_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        confirmed_only: Optional[bool] = None,
    ) -> list[Optional[Hashable]]:
        """Return labels of extracted objects."""
        return [
            estimate.label
            for estimate in self.get_object_estimates(
                extraction_threshold=extraction_threshold,
                max_objects=max_objects,
                confirmed_only=confirmed_only,
            )
        ]

    def get_point_estimate(
        self,
        flatten_vector: bool = False,
        include_extent: bool = True,
        extraction_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        confirmed_only: Optional[bool] = None,
    ):
        """Return vectorized extracted-object estimates.

        With ``flatten_vector=False``, each column represents one object. With
        ``flatten_vector=True``, the matrix is flattened for compatibility with
        existing multi-target logging helpers.
        """
        estimates = self.get_object_estimates(
            extraction_threshold=extraction_threshold,
            max_objects=max_objects,
            confirmed_only=confirmed_only,
        )
        if not estimates:
            point_estimates = empty((0, 0))
        else:
            point_estimates = stack(
                [
                    self._vectorize_object_estimate(
                        estimate, include_extent=include_extent
                    )
                    for estimate in estimates
                ],
                axis=1,
            )
        if flatten_vector:
            return point_estimates.flatten()
        return point_estimates

    def get_point_estimate_kinematics(
        self,
        extraction_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        confirmed_only: Optional[bool] = None,
    ) -> list[Any]:
        """Return kinematic estimates only."""
        return [
            estimate.kinematics
            for estimate in self.get_object_estimates(
                extraction_threshold=extraction_threshold,
                max_objects=max_objects,
                confirmed_only=confirmed_only,
            )
        ]

    def get_point_estimate_extents(
        self,
        extraction_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        confirmed_only: Optional[bool] = None,
    ) -> list[Any]:
        """Return extent estimates only."""
        return [
            estimate.extent
            for estimate in self.get_object_estimates(
                extraction_threshold=extraction_threshold,
                max_objects=max_objects,
                confirmed_only=confirmed_only,
            )
        ]

    def get_measurement_rate_estimates(
        self,
        extraction_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
        confirmed_only: Optional[bool] = None,
        unavailable_value: Any = None,
    ) -> list[Any]:
        """Return measurement-rate estimates where available."""
        return [
            (
                unavailable_value
                if estimate.measurement_rate is None
                else estimate.measurement_rate
            )
            for estimate in self.get_object_estimates(
                extraction_threshold=extraction_threshold,
                max_objects=max_objects,
                confirmed_only=confirmed_only,
            )
        ]

    @abstractmethod
    def get_contour_points(
        self,
        n: int,
        labels: Optional[Sequence[Hashable]] = None,
        scaling_factor: float = 1.0,
        **kwargs,
    ) -> dict[Optional[Hashable], Any]:
        """Return drawable contour points for extracted objects."""

    def get_cardinality_distribution(self):
        """Return the cardinality distribution if represented explicitly."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a cardinality distribution."
        )

    def get_expected_number_of_targets(self) -> float:
        """Return expected cardinality when it can be inferred from estimates."""
        estimates = self.get_object_estimates()
        if not estimates:
            return 0.0

        existence_probabilities: list[float] = []
        for estimate in estimates:
            existence_probability = estimate.existence_probability
            if existence_probability is None:
                return float(len(estimates))
            existence_probabilities.append(existence_probability)
        return float(sum(existence_probabilities))

    def prune(self, *args, **kwargs) -> None:
        """Optional complexity-reduction hook."""
        del args, kwargs

    def merge(self, *args, **kwargs) -> None:
        """Optional component-merging hook."""
        del args, kwargs

    def cap(self, *args, **kwargs) -> None:
        """Optional component-capping hook."""
        del args, kwargs

    def reduce(self, *args, **kwargs) -> None:
        """Run optional pruning, merging, and capping hooks."""
        self.prune(*args, **kwargs)
        self.merge(*args, **kwargs)
        self.cap(*args, **kwargs)

    def store_prior_extents(self) -> None:
        """Record prior extent estimates."""
        self.prior_extents_over_time = self._record_estimates(
            "prior_extents",
            self._history_vector_from_values(self.get_point_estimate_extents()),
        )

    def store_posterior_extents(self) -> None:
        """Record posterior extent estimates."""
        self.posterior_extents_over_time = self._record_estimates(
            "posterior_extents",
            self._history_vector_from_values(self.get_point_estimate_extents()),
        )

    def store_prior_measurement_rates(self) -> None:
        """Record prior measurement-rate estimates."""
        self.prior_measurement_rates_over_time = self._record_estimates(
            "prior_measurement_rates",
            asarray(
                self.get_measurement_rate_estimates(unavailable_value=float("nan"))
            ),
        )

    def store_posterior_measurement_rates(self) -> None:
        """Record posterior measurement-rate estimates."""
        self.posterior_measurement_rates_over_time = self._record_estimates(
            "posterior_measurement_rates",
            asarray(
                self.get_measurement_rate_estimates(unavailable_value=float("nan"))
            ),
        )

    def clear_history(self, name=None):
        """Clear histories and keep MEOT-specific mirrors synchronized."""
        super().clear_history(name)
        if name is None or name == "prior_measurement_rates":
            if "prior_measurement_rates" in self.history:
                self.prior_measurement_rates_over_time = self.history[
                    "prior_measurement_rates"
                ]
        if name is None or name == "posterior_measurement_rates":
            if "posterior_measurement_rates" in self.history:
                self.posterior_measurement_rates_over_time = self.history[
                    "posterior_measurement_rates"
                ]
        if name is None or name == "cardinality":
            if "cardinality" in self.history:
                self.cardinality_over_time = self.history["cardinality"]
        if name is None or name == "associations":
            if "associations" in self.history:
                self.associations_over_time = self.history["associations"]

    def _vectorize_object_estimate(
        self, estimate: ExtendedObjectEstimate, include_extent: bool = True
    ):
        """Vectorize one object estimate for stacked point-estimate output."""
        kinematics = asarray(estimate.kinematics).flatten()
        if not include_extent or estimate.extent is None:
            return kinematics
        extent = asarray(estimate.extent).flatten()
        return concatenate((kinematics, extent))

    @staticmethod
    def _history_vector_from_values(values: Sequence[Any]):
        """Flatten and concatenate per-object values for padded histories."""
        if not values:
            return empty((0,))
        return concatenate([asarray(value).flatten() for value in values])

    def _get_cardinality_history_value(self):
        """Return an explicit cardinality PMF or fall back to expected count."""
        try:
            return self.get_cardinality_distribution()
        except NotImplementedError:
            return self.get_expected_number_of_targets()


__all__ = [
    "AbstractMultipleExtendedObjectTracker",
    "ExtendedObjectAssociationResult",
    "ExtendedObjectEstimate",
    "MultipleExtendedObjectStepResult",
]
