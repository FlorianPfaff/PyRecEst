import collections
import copy
import warnings
from typing import Union

import numpy as np
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    asarray,
    int32,
    int64,
    ones,
    random,
    reshape,
    sum,
    zeros,
)

from .abstract_distribution_type import AbstractDistributionType
from .abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_BOOLEAN_TYPES = (bool, np.bool_)
_COMPLEX_TYPES = (complex, np.complexfloating)
_INVALID_SAMPLE_COUNT_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    np.datetime64,
    np.timedelta64,
)
_TEMPORAL_DTYPE_KINDS = {"M", "m"}
_TEMPORAL_SCALAR_TYPES = (np.datetime64, np.timedelta64)


def _validate_positive_sample_count(n) -> int:
    message = "n must be a positive integer"
    if np.ma.is_masked(n):
        raise ValueError(message)
    try:
        count_array = np.asarray(n)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if count_array.ndim != 0:
        raise ValueError(message)
    if getattr(count_array.dtype, "kind", None) in _TEMPORAL_DTYPE_KINDS:
        raise ValueError(message)

    count = count_array.item()
    if isinstance(count, _INVALID_SAMPLE_COUNT_TYPES):
        raise ValueError(message)

    try:
        count_int = int(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc

    try:
        is_exact_integer = count == count_int
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not bool(is_exact_integer):
        raise ValueError(message)
    if count_int <= 0:
        raise ValueError(message)
    return count_int


def _validate_explicit_weight_shape(weights, num_distributions: int):
    """Return explicit mixture weights without silently flattening matrices."""
    _validate_mixture_weight_values(weights)
    if isinstance(weights, np.ma.MaskedArray):
        weights = np.ma.getdata(weights)
    weights = pyrecest.backend.copy(asarray(weights))
    if weights.ndim == 0:
        if num_distributions != 1:
            raise ValueError("Mixture weights must be one-dimensional")
        return reshape(weights, (1,))
    if weights.ndim != 1:
        raise ValueError("Mixture weights must be one-dimensional")
    return weights


def _validate_mixture_weight_values(weights) -> None:
    """Reject invalid mixture weights before backend scalar comparisons."""
    if np.ma.is_masked(weights):
        raise ValueError("Mixture weights must not contain masked values")
    if isinstance(weights, np.ma.MaskedArray):
        weights = np.ma.getdata(weights)
    try:
        converted_weights = pyrecest.backend.to_numpy(weights)
        native_weight_values = np.asarray(converted_weights)
    except Exception as exc:  # pragma: no cover - backend-specific conversion type
        raise ValueError("Mixture weights must be real-valued numeric") from exc

    if getattr(native_weight_values.dtype, "kind", None) in _TEMPORAL_DTYPE_KINDS:
        raise ValueError("Mixture weights must be real-valued numeric")

    try:
        weight_values = np.asarray(converted_weights, dtype=object)
    except Exception as exc:  # pragma: no cover - NumPy object-conversion failure
        raise ValueError("Mixture weights must be real-valued numeric") from exc

    for weight in weight_values.reshape(-1):
        if isinstance(weight, _BOOLEAN_TYPES):
            raise ValueError("Mixture weights must be real-valued numeric, not boolean")
        if isinstance(weight, _TEXT_TYPES):
            raise ValueError("Mixture weights must be real-valued numeric")
        if isinstance(weight, _TEMPORAL_SCALAR_TYPES):
            raise ValueError("Mixture weights must be real-valued numeric")
        if isinstance(weight, _COMPLEX_TYPES):
            raise ValueError("Mixture weights must be real-valued numeric")

        try:
            parsed_weight = float(weight)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Mixture weights must be real-valued numeric") from exc

        if not np.isfinite(parsed_weight):
            raise ValueError("Mixture weights must be finite")
        if parsed_weight < 0.0:
            raise ValueError("Mixture weights must be nonnegative")


def _normalize_mixture_weights_on_host(host_weights):
    """Normalize finite nonnegative weights when backend reductions lose scale."""
    weight_scale = float(np.max(host_weights))
    if not np.isfinite(weight_scale) or weight_scale <= 0.0:
        raise ValueError("Mixture weights must have positive finite total mass")

    scaled_weights = host_weights / weight_scale
    scaled_weight_sum = float(np.sum(scaled_weights))
    if not np.isfinite(scaled_weight_sum) or scaled_weight_sum <= 0.0:
        raise ValueError("Mixture weights must have positive finite total mass")

    return asarray(scaled_weights / scaled_weight_sum)


class AbstractMixture(AbstractDistributionType):
    """
    Abstract base class for mixture distributions.
    """

    def __init__(
        self,
        dists: collections.abc.Sequence[AbstractManifoldSpecificDistribution],
        weights=None,
    ):
        AbstractDistributionType.__init__(self)
        dists = copy.deepcopy(dists)  # To prevent modifying the original object
        num_distributions = len(dists)
        if num_distributions == 0:
            raise ValueError("Mixture must contain at least one distribution")

        if weights is None:
            weights = ones(num_distributions) / num_distributions
        else:
            weights = _validate_explicit_weight_shape(weights, num_distributions)

        if num_distributions != weights.shape[0]:
            raise ValueError("Sizes of distributions and weights must be equal")

        _validate_mixture_weight_values(weights)

        if not all(dists[0].dim == dist.dim for dist in dists):
            raise ValueError("All distributions must have the same dimension")
        if not all(dists[0].input_dim == dist.input_dim for dist in dists):
            raise ValueError("All distributions must have the same input dimension")

        # XLA may flush subnormal operands during comparisons and reductions even
        # though device_get preserves their stored float64 values. Determine exact
        # zero support from the already validated host representation so valid tiny
        # components are not pruned before normalization.
        host_weights = np.asarray(pyrecest.backend.to_numpy(weights))
        non_zero_indices = np.flatnonzero(host_weights != 0.0).tolist()

        if len(non_zero_indices) == 0:
            raise ValueError("At least one mixture weight must be nonzero")

        try:
            weight_sum = sum(weights)
        except FloatingPointError:
            weight_sum = None
        if (
            weight_sum is not None
            and bool(pyrecest.backend.isfinite(weight_sum))
            and bool(weight_sum > 0.0)
        ):
            normalized_weights = weights / weight_sum
            weights_sum_to_one = bool(abs(weight_sum - 1.0) <= 1e-10)
        else:
            weight_scale = pyrecest.backend.max(weights)
            scaled_weight_sum = None
            if bool(pyrecest.backend.isfinite(weight_scale)) and bool(
                weight_scale > 0.0
            ):
                scale_root = pyrecest.backend.sqrt(weight_scale)
                # JAX may lower ``weights / weight_scale`` to multiplication by an
                # underflowed reciprocal when ``weight_scale`` is near float64.max.
                # Splitting the division across two square-root-sized factors keeps
                # the scaled weights and their sum finite without changing ratios.
                scaled_weights = (weights / scale_root) / scale_root
                try:
                    scaled_weight_sum = sum(scaled_weights)
                except FloatingPointError:
                    scaled_weight_sum = None

            if (
                scaled_weight_sum is not None
                and bool(pyrecest.backend.isfinite(scaled_weight_sum))
                and bool(scaled_weight_sum > 0.0)
            ):
                normalized_weights = scaled_weights / scaled_weight_sum
            else:
                # JAX/XLA can flush subnormal values to zero during arithmetic.
                # Normalize those rare inputs on the host, where their stored
                # float64 ratios are still available, then return ordinary-sized
                # probabilities to the active backend.
                normalized_weights = _normalize_mixture_weights_on_host(host_weights)
            weights_sum_to_one = False

        if len(non_zero_indices) < len(weights):
            warnings.warn(
                "Elements with zero weights detected. Pruning elements in mixture with weight zero."
            )
            dists = [dists[i] for i in non_zero_indices]
            normalized_weights = normalized_weights[
                array(non_zero_indices, dtype=int64)
            ]

        self.dists = dists

        if not weights_sum_to_one:
            warnings.warn("Weights of mixture do not sum to one.")
        self.w = normalized_weights

    @property
    def input_dim(self) -> int:
        return self.dists[0].input_dim

    def _as_sample_matrix(self, samples, n_samples: int):
        samples = asarray(samples)

        if self.input_dim == 1 and samples.ndim == 0:
            samples = reshape(samples, (1, 1))
        elif self.input_dim == 1 and samples.ndim == 1:
            samples = reshape(samples, (-1, 1))
        else:
            samples = pyrecest.backend.atleast_2d(samples)

        expected_shape = (n_samples, self.input_dim)
        if tuple(samples.shape) != expected_shape:
            raise ValueError(
                "Mixture component sample output must have shape "
                f"{expected_shape}, got {tuple(samples.shape)}"
            )
        return samples

    def _sample_component_matrix(self, component_index: int, n_samples: int):
        try:
            sample_i = self.dists[component_index].sample(n_samples)
        except (NotImplementedError, AssertionError, ValueError, TypeError):
            if pyrecest.backend.__backend_name__ != "jax":
                raise
            sample_i = self.dists[component_index].sample_metropolis_hastings(n_samples)
        return self._as_sample_matrix(sample_i, n_samples)

    def sample(self, n: Union[int, int32, int64]):
        """Draw iid mixture samples without biasing output positions.

        Sampling only multinomial component counts gives the correct unordered
        bag of component labels, but filling the output array with component-wise
        blocks makes early output positions more likely to come from early
        components. Draw one component label per requested sample and place each
        component's samples back into those sampled positions.
        """
        n = _validate_positive_sample_count(n)
        component_indices = random.choice(len(self.dists), size=n, p=self.w)
        component_indices_np = pyrecest.backend.to_numpy(component_indices).reshape(-1)

        if pyrecest.backend.__backend_name__ == "jax":
            samples = [
                self._sample_component_matrix(int(component_index), 1)
                for component_index in component_indices_np
            ]
            return pyrecest.backend.concatenate(samples, axis=0)

        sample_blocks = []
        position_blocks = []
        for component_index in range(len(self.dists)):
            positions = np.flatnonzero(component_indices_np == component_index)
            occ_val = int(positions.size)
            if occ_val == 0:
                continue
            sample_blocks.append(
                self._sample_component_matrix(component_index, occ_val)
            )
            position_blocks.append(positions)

        sample_blocks = pyrecest.backend.convert_to_wider_dtype(sample_blocks)
        samples_by_component = pyrecest.backend.concatenate(sample_blocks, axis=0)
        positions_by_component = np.concatenate(position_blocks)
        sampled_position_order = array(
            np.argsort(positions_by_component),
            dtype=int64,
        )
        return samples_by_component[sampled_position_order]

    def pdf(self, xs):
        xs = asarray(xs)

        if self.input_dim == 1 and xs.ndim <= 1:
            # For one-dimensional distributions, a flat array represents a batch
            # of scalar evaluation points. This matches GaussianDistribution.pdf
            # and avoids rejecting natural inputs such as xs.shape == (n,).
            p = zeros(xs.shape)
        else:
            if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
                raise ValueError("Dimension mismatch")
            p = zeros(1) if xs.ndim == 1 else zeros(xs.shape[:-1])

        for i, dist in enumerate(self.dists):
            p += self.w[i] * dist.pdf(xs)
        return p
