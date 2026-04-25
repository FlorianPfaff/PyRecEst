"""Feature-based probabilistic association models.

This module adds a lightweight learned association model that is useful for
cross-session neuron identity tracking. In settings such as Track2p or CaliAli,
pairwise association decisions are usually based on several cues at once, for
example registered centroid distance, ROI overlap, mask correlation, or trace
similarity. The :class:`LogisticPairwiseAssociationModel` learns how to fuse
those cues into calibrated match probabilities and assignment costs.

The API accepts either a standard design matrix with shape
``(n_samples, n_features)`` or a higher-order pairwise feature tensor with shape
``(..., n_features)`` together with a label array of shape ``(...)``. This is
convenient when ground-truth match matrices are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# pylint: disable=no-name-in-module,no-member,redefined-builtin
import numpy as _numpy
from numpy.linalg import LinAlgError
from pyrecest.backend import (
    abs,
    all,
    any,
    asarray,
    clip,
    concatenate,
    diag,
    exp,
    float64,
    int64,
    isfinite,
    linalg,
    log,
    max,
    ones,
    size,
    unique,
    where,
    zeros,
)

_COST_MODE = Literal["negative_log_probability", "one_minus_probability"]


@dataclass(frozen=True)
class _BinaryClassWeights:
    """Resolved weights for the negative and positive class."""

    negative: float
    positive: float

    def as_array(self, labels: Any) -> Any:
        return asarray(where(labels == 1, self.positive, self.negative), dtype=float64)


class LogisticPairwiseAssociationModel:  # pylint: disable=too-many-instance-attributes
    """Learn pairwise match probabilities from arbitrary association features.

    The model uses binary logistic regression with optional feature
    standardization, L2 regularization, and support for severe class imbalance.
    It is intentionally dependency-light and implemented with NumPy only so that
    it can be used inside PyRecEst without adding a machine-learning stack.

    Parameters
    ----------
    fit_intercept:
        Whether to learn an intercept term.
    l2_regularization:
        Non-negative L2 regularization strength applied to the coefficients.
        The intercept is not regularized.
    max_iterations:
        Maximum number of Newton / IRLS iterations.
    tolerance:
        Stop once the largest absolute parameter update falls below this value.
    standardize:
        Whether to z-score features before optimization.
    class_weight:
        ``None`` for unweighted fitting, ``"balanced"`` to automatically rebalance
        positive and negative examples, or a dictionary containing weights for
        labels ``0`` and ``1``.
    probability_clip:
        Numerical clipping applied to probabilities before taking logarithms.

    Notes
    -----
    The fit method accepts both a flattened feature matrix and a pairwise tensor.
    This makes it convenient to train directly from a ground-truth match matrix:

    >>> model = LogisticPairwiseAssociationModel(class_weight="balanced")
    >>> model.fit(pairwise_features, match_labels)
    >>> costs = model.pairwise_cost_matrix(pairwise_features)

    Here ``pairwise_features`` may contain, for example, centroid distance,
    overlap, and shape-correlation channels for every candidate neuron pair.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        fit_intercept: bool = True,
        l2_regularization: float = 1.0e-3,
        max_iterations: int = 100,
        tolerance: float = 1.0e-8,
        standardize: bool = True,
        class_weight: str | dict[int, float] | None = "balanced",
        probability_clip: float = 1.0e-12,
    ) -> None:
        if l2_regularization < 0.0:
            raise ValueError("l2_regularization must be non-negative")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if not 0.0 < probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, 0.5)")
        if (
            class_weight != "balanced"
            and class_weight is not None
            and not isinstance(class_weight, dict)
        ):
            raise ValueError(
                "class_weight must be None, 'balanced', or a dictionary with labels 0 and 1"
            )

        self.fit_intercept = fit_intercept
        self.l2_regularization = float(l2_regularization)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.standardize = bool(standardize)
        self.class_weight = class_weight
        self.probability_clip = float(probability_clip)

        self.n_features_in_: int | None = None
        self.feature_mean_: Any | None = None
        self.feature_scale_: Any | None = None
        self.coefficients_: Any | None = None
        self.intercept_: float | None = None
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.class_weights_: _BinaryClassWeights | None = None

    @staticmethod
    def _sigmoid(values: Any) -> Any:
        """Numerically stable sigmoid."""
        values = asarray(values, dtype=float64)
        exp_neg = exp(-abs(values))
        return where(values >= 0.0, 1.0 / (1.0 + exp_neg), exp_neg / (1.0 + exp_neg))

    @staticmethod
    def _ensure_binary_labels(labels: Any) -> Any:
        labels = asarray(labels).reshape(-1)
        if size(labels) == 0:
            raise ValueError("At least one labeled example is required")

        unique_labels = unique(labels)
        if not all((unique_labels == 0) | (unique_labels == 1)):
            raise ValueError("labels must only contain binary values 0/1 or False/True")

        labels = asarray(labels, dtype=int64)
        if size(unique_labels) != 2:
            raise ValueError("Both negative and positive examples are required")
        return labels

    @staticmethod
    def _prepare_training_features(features: Any) -> Any:
        features = asarray(features, dtype=float64)
        if features.ndim == 1:
            features = features[:, None]
        if features.ndim < 2:
            raise ValueError("features must be at least one-dimensional")

        flattened = features.reshape(-1, features.shape[-1])
        if flattened.shape[0] == 0:
            raise ValueError("At least one feature vector is required")
        if not all(isfinite(flattened)):
            raise ValueError("features must be finite")
        return flattened

    @staticmethod
    def _prepare_prediction_features(
        features: Any,
        expected_feature_dimension: int,
    ) -> tuple[Any, tuple[int, ...]]:
        features = asarray(features, dtype=float64)
        if features.ndim == 1:
            if features.shape[0] != expected_feature_dimension:
                raise ValueError(
                    "A one-dimensional prediction input must contain exactly one feature vector"
                )
            original_shape = ()
            flattened = features[None, :]
        else:
            if features.shape[-1] != expected_feature_dimension:
                raise ValueError(
                    "The last axis of features must equal the number of fitted features"
                )
            original_shape = features.shape[:-1]
            flattened = features.reshape(-1, expected_feature_dimension)

        if not all(isfinite(flattened)):
            raise ValueError("features must be finite")
        return flattened, original_shape

    def _resolve_class_weights(self, labels: Any) -> _BinaryClassWeights:
        if self.class_weight is None:
            return _BinaryClassWeights(negative=1.0, positive=1.0)

        if self.class_weight == "balanced":
            n_total = labels.shape[0]
            n_pos = int(labels.sum())
            n_neg = n_total - n_pos
            if n_neg == 0 or n_pos == 0:
                raise ValueError(
                    "Both classes must be present when using balanced class weights"
                )
            total_count = float(n_total)
            return _BinaryClassWeights(
                negative=total_count / (2.0 * n_neg),
                positive=total_count / (2.0 * n_pos),
            )

        if 0 not in self.class_weight or 1 not in self.class_weight:
            raise ValueError(
                "class_weight dictionaries must contain entries for both 0 and 1"
            )
        negative_weight = float(self.class_weight[0])
        positive_weight = float(self.class_weight[1])
        if negative_weight <= 0.0 or positive_weight <= 0.0:
            raise ValueError("class weights must be positive")
        return _BinaryClassWeights(negative=negative_weight, positive=positive_weight)

    @staticmethod
    def _flatten_sample_weight(sample_weight: Any | None, labels: Any) -> Any | None:
        if sample_weight is None:
            return None

        flattened_sample_weight = asarray(sample_weight, dtype=float64).reshape(-1)
        if flattened_sample_weight.shape[0] != labels.shape[0]:
            raise ValueError("sample_weight must match the number of flattened labels")
        return flattened_sample_weight

    def _build_effective_sample_weights(
        self,
        labels: Any,
        sample_weight: Any | None,
    ) -> Any:
        weights = ones(labels.shape[0], dtype=float64)
        if sample_weight is not None:
            sample_weight = asarray(sample_weight, dtype=float64)
            if sample_weight.shape != labels.shape:
                raise ValueError("sample_weight must match the flattened label shape")
            if any(sample_weight < 0.0):
                raise ValueError("sample_weight must be non-negative")
            weights = weights * sample_weight

        class_weights = self._resolve_class_weights(labels)
        self.class_weights_ = class_weights
        weights = weights * class_weights.as_array(labels)

        if not any(weights > 0.0):
            raise ValueError("At least one example must receive positive weight")
        return weights

    def _fit_standardization(self, features: Any) -> Any:
        if self.standardize:
            feature_mean = features.mean(axis=0)
            feature_scale = features.std(axis=0)
            feature_scale = where(feature_scale > 0.0, feature_scale, 1.0)
        else:
            feature_mean = zeros(features.shape[1], dtype=float64)
            feature_scale = ones(features.shape[1], dtype=float64)

        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        return (features - feature_mean) / feature_scale

    def _transform_features(self, features: Any) -> Any:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("The association model is not fitted")
        return (features - self.feature_mean_) / self.feature_scale_

    def _design_matrix(self, features: Any) -> Any:
        if self.fit_intercept:
            intercept_column = ones((features.shape[0], 1), dtype=float64)
            return concatenate((intercept_column, features), axis=1)
        return features

    def _require_fitted(self) -> None:
        if (
            self.n_features_in_ is None
            or self.coefficients_ is None
            or self.intercept_ is None
        ):
            raise RuntimeError("The association model must be fitted before use")

    def _prepare_fit_inputs(
        self,
        flattened_features: Any,
        labels: Any,
        sample_weight: Any | None,
    ) -> tuple[Any, Any]:
        self.n_features_in_ = flattened_features.shape[1]
        standardized_features = self._fit_standardization(flattened_features)
        design_matrix = self._design_matrix(standardized_features)
        effective_weights = self._build_effective_sample_weights(labels, sample_weight)
        return design_matrix, effective_weights

    def _regularization_mask(self, n_params: int) -> Any:
        if self.fit_intercept:
            return concatenate(
                (zeros(1, dtype=float64), ones(n_params - 1, dtype=float64))
            )
        return ones(n_params, dtype=float64)

    @staticmethod
    def _effective_tolerance(
        parameter_vector: Any, requested_tolerance: float
    ) -> float:
        try:
            dtype_eps = float(_numpy.finfo(parameter_vector.dtype).eps)
        except (AttributeError, TypeError, ValueError):
            dtype_eps = float(_numpy.finfo(float).eps)
        dtype_threshold = dtype_eps * 1000.0
        return (
            dtype_threshold
            if dtype_threshold > requested_tolerance
            else requested_tolerance
        )

    def _store_fitted_parameters(self, parameter_vector: Any) -> None:
        if self.fit_intercept:
            self.intercept_ = float(parameter_vector[0])
            self.coefficients_ = parameter_vector[1:]
            return
        self.intercept_ = 0.0
        self.coefficients_ = parameter_vector

    def fit(  # pylint: disable=too-many-locals
        self,
        features: Any,
        labels: Any,
        sample_weight: Any | None = None,
    ) -> "LogisticPairwiseAssociationModel":
        """Fit the association model."""
        flattened_features = self._prepare_training_features(features)
        labels = self._ensure_binary_labels(labels)
        if flattened_features.shape[0] != labels.shape[0]:
            raise ValueError(
                "The number of feature vectors must equal the number of labels after flattening"
            )

        sample_weight = self._flatten_sample_weight(sample_weight, labels)
        design_matrix, effective_weights = self._prepare_fit_inputs(
            flattened_features, labels, sample_weight
        )
        parameter_vector = zeros(design_matrix.shape[1], dtype=float64)
        regularization_mask = self._regularization_mask(design_matrix.shape[1])

        self.converged_ = False
        for iteration in range(1, self.max_iterations + 1):
            logits = design_matrix @ parameter_vector
            probabilities = self._sigmoid(logits)
            residual = (probabilities - labels) * effective_weights
            gradient = design_matrix.T @ residual
            if self.l2_regularization > 0.0:
                regularization = self.l2_regularization * regularization_mask
                gradient = gradient + regularization * parameter_vector

            curvature = effective_weights * probabilities * (1.0 - probabilities)
            hessian = design_matrix.T @ (design_matrix * curvature[:, None])
            if self.l2_regularization > 0.0:
                hessian = hessian + self.l2_regularization * diag(regularization_mask)

            try:
                step = linalg.solve(hessian, gradient)
            except LinAlgError:
                step = linalg.pinv(hessian) @ gradient

            parameter_vector = parameter_vector - step
            self.n_iter_ = iteration
            if max(abs(step)) <= self._effective_tolerance(
                parameter_vector, self.tolerance
            ):
                self.converged_ = True
                break

        self._store_fitted_parameters(parameter_vector)
        return self

    def decision_function(self, features: Any) -> Any:
        """Return posterior log-odds for the provided feature vectors."""
        self._require_fitted()
        assert self.n_features_in_ is not None
        flattened_features, original_shape = self._prepare_prediction_features(
            features, self.n_features_in_
        )
        standardized_features = self._transform_features(flattened_features)
        if self.fit_intercept:
            intercept = asarray([self.intercept_], dtype=float64)
            parameter_vector = concatenate((intercept, self.coefficients_))
        else:
            parameter_vector = self.coefficients_
        logits = self._design_matrix(standardized_features) @ parameter_vector
        if original_shape == ():
            return asarray(logits[0])
        return logits.reshape(original_shape)

    def predict_log_odds(self, features: Any) -> Any:
        """Alias for :meth:`decision_function`."""
        return self.decision_function(features)

    def predict_match_probability(self, features: Any) -> Any:
        """Return posterior match probabilities for the provided feature vectors."""
        return self._sigmoid(self.decision_function(features))

    def predict(self, features: Any, threshold: float = 0.5) -> Any:
        """Predict binary match decisions using the supplied threshold."""
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        return self.predict_match_probability(features) >= threshold

    def pairwise_cost_matrix(
        self,
        pairwise_features: Any,
        *,
        mode: _COST_MODE = "negative_log_probability",
    ) -> Any:
        """Convert pairwise features into an assignment cost matrix."""
        probabilities = clip(
            self.predict_match_probability(pairwise_features),
            self.probability_clip,
            1.0 - self.probability_clip,
        )
        if mode == "negative_log_probability":
            return -log(probabilities)
        if mode == "one_minus_probability":
            return 1.0 - probabilities
        raise ValueError(f"Unsupported cost mode: {mode}")
