"""Information-form distributed Kalman filtering utilities."""

from __future__ import annotations

import copy as pycopy
import hashlib
from collections.abc import Hashable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    atleast_1d,
    atleast_2d,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    eye,
    linalg,
    zeros,
)
from pyrecest.distributions import GaussianDistribution
from pyrecest.protocols.common import BackendArray

from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin

_SENTINEL = object()


def _extend_hash(previous_hash: str | None, operation_name: str) -> str:
    previous = "" if previous_hash is None else previous_hash
    return hashlib.sha256(f"{previous}|{operation_name}".encode("utf-8")).hexdigest()


def _copy_array(value):
    if value is None:
        return None
    return backend_copy(value)


def _required(model, *names):
    for name in names:
        value = getattr(model, name, _SENTINEL)
        if value is not _SENTINEL:
            return value
    options = ", ".join(f"`{name}`" for name in names)
    raise AttributeError(f"{type(model).__name__} must expose one of: {options}.")


def _optional(model, *names):
    for name in names:
        value = getattr(model, name, _SENTINEL)
        if value is not _SENTINEL:
            return value
    return None


@dataclass(frozen=True)
class ContributionKey:
    """Identity of one additive information contribution."""

    origin_node_id: Hashable
    track_id: Hashable | None = None


@dataclass
class InformationContribution:
    """Additive IDKF information-vector contribution."""

    key: ContributionKey
    y: BackendArray
    epoch: int = 0
    operation_count: int = 0
    operation_hash: str | None = None

    def copy(self) -> "InformationContribution":
        return InformationContribution(
            key=self.key,
            y=_copy_array(self.y),
            epoch=self.epoch,
            operation_count=self.operation_count,
            operation_hash=self.operation_hash,
        )


@dataclass
class GlobalInformationState:
    """Shared information matrix and operation metadata."""

    Y: BackendArray
    transform_to_end: BackendArray | None = None
    epoch: int = 0
    operation_count: int = 0
    operation_hash: str | None = None

    def copy(self) -> "GlobalInformationState":
        return GlobalInformationState(
            Y=_copy_array(self.Y),
            transform_to_end=_copy_array(self.transform_to_end),
            epoch=self.epoch,
            operation_count=self.operation_count,
            operation_hash=self.operation_hash,
        )


class IdkfNode(AbstractFilter, EuclideanFilterMixin):
    """Node-level information-form distributed Kalman filter.

    Bank entries are additive information-vector contributions.  The aggregate
    Gaussian is obtained from Y^{-1} sum_i y_i.  Individual bank entries are not
    treated as independent posterior Gaussians.
    """

    def __init__(
        self,
        node_id,
        own_contribution: InformationContribution,
        global_information_state: GlobalInformationState,
        *,
        contribution_bank: (
            Mapping[ContributionKey, InformationContribution] | None
        ) = None,
        measurement_matrix=None,
        meas_noise=None,
        measurement_models: Sequence | Mapping | None = None,
        local_filter_bank: MutableMapping[Hashable, AbstractFilter] | None = None,
        compatibility_rtol: float = 1e-7,
        compatibility_atol: float = 1e-10,
    ):
        EuclideanFilterMixin.__init__(self)
        AbstractFilter.__init__(self, None)
        if own_contribution is None:
            raise ValueError("own_contribution must be provided")
        if global_information_state is None:
            raise ValueError("global_information_state must be provided")

        self.node_id = node_id
        self.own_contribution_key = own_contribution.key
        self.global_information_state = global_information_state.copy()
        self.measurement_matrix = (
            None if measurement_matrix is None else atleast_2d(measurement_matrix)
        )
        self.meas_noise = None if meas_noise is None else atleast_2d(meas_noise)
        self.measurement_models = measurement_models
        self.local_filter_bank = (
            {} if local_filter_bank is None else dict(local_filter_bank)
        )
        self.compatibility_rtol = compatibility_rtol
        self.compatibility_atol = compatibility_atol

        self.contribution_bank: dict[ContributionKey, InformationContribution] = {}
        if contribution_bank is not None:
            for key, contribution in contribution_bank.items():
                if key != contribution.key:
                    raise ValueError(
                        "contribution_bank keys must match contribution.key"
                    )
                self.contribution_bank[key] = contribution.copy()
        self.contribution_bank[self.own_contribution_key] = own_contribution.copy()
        self.seen_contribution_ids = {
            self._message_identity(contribution)
            for contribution in self.contribution_bank.values()
        }

    @classmethod
    def from_local_gaussian(
        cls,
        node_id,
        initial_state,
        all_prior_covariances,
        *,
        measurement_matrix=None,
        meas_noise=None,
        measurement_models: Sequence | Mapping | None = None,
        track_id: Hashable | None = None,
        epoch: int = 0,
        transform_to_end=None,
        **kwargs,
    ) -> "IdkfNode":
        state = cls._coerce_gaussian(initial_state)
        key = ContributionKey(node_id, track_id)
        own_contribution = InformationContribution(
            key=key,
            y=linalg.solve(atleast_2d(state.C), atleast_1d(state.mu)),
            epoch=epoch,
        )
        global_state = GlobalInformationState(
            Y=cls._sum_precision_matrices(all_prior_covariances),
            transform_to_end=transform_to_end,
            epoch=epoch,
        )
        return cls(
            node_id=node_id,
            own_contribution=own_contribution,
            global_information_state=global_state,
            measurement_matrix=measurement_matrix,
            meas_noise=meas_noise,
            measurement_models=measurement_models,
            **kwargs,
        )

    @staticmethod
    def _coerce_gaussian(state) -> GaussianDistribution:
        if isinstance(state, GaussianDistribution):
            return GaussianDistribution(
                atleast_1d(state.mu), atleast_2d(state.C), check_validity=False
            )
        if isinstance(state, tuple) and len(state) == 2:
            mean, covariance = state
            return GaussianDistribution(
                atleast_1d(mean), atleast_2d(covariance), check_validity=False
            )
        raise ValueError(
            "state must be a GaussianDistribution or a (mean, covariance) tuple"
        )

    @staticmethod
    def _sum_precision_matrices(covariances):
        iterable = (
            covariances.values() if isinstance(covariances, Mapping) else covariances
        )
        total = None
        for covariance in iterable:
            precision = linalg.inv(atleast_2d(covariance))
            total = precision if total is None else total + precision
        if total is None:
            raise ValueError("all_prior_covariances must contain at least one matrix")
        return total

    @property
    def dim(self) -> int:
        return self.global_information_state.Y.shape[0]

    @property
    def y(self):
        """Return this node's own information-vector contribution."""
        return self.contribution_bank[self.own_contribution_key].y

    @y.setter
    def y(self, value):
        self.contribution_bank[self.own_contribution_key].y = atleast_1d(value)

    @property
    def Ymat_glob(self):
        """Compatibility alias for the global information matrix."""
        return self.global_information_state.Y

    @Ymat_glob.setter
    def Ymat_glob(self, value):
        self.global_information_state.Y = atleast_2d(value)

    @property
    def global_information_matrix(self):
        return self.global_information_state.Y

    @property
    def current_contribution_keys(self) -> frozenset[ContributionKey]:
        return frozenset(
            key
            for key, contribution in self.contribution_bank.items()
            if self.is_contribution_current(contribution)
        )

    @property
    def stale_contribution_keys(self) -> frozenset[ContributionKey]:
        return frozenset(
            key
            for key, contribution in self.contribution_bank.items()
            if not self.is_contribution_current(contribution)
        )

    @property
    def filter_state(self) -> GaussianDistribution:
        Y = self.global_information_state.Y
        y_sum = self.information_vector_sum(include_stale=False)
        return GaussianDistribution(
            linalg.solve(Y, y_sum), linalg.inv(Y), check_validity=False
        )

    @filter_state.setter
    def filter_state(self, _new_state):
        raise AttributeError("IDKF filter_state is derived from the contribution bank")

    def get_point_estimate(self):
        return self.filter_state.mu

    def information_vector_sum(self, *, include_stale: bool = False):
        total = None
        for contribution in self.contribution_bank.values():
            if include_stale or self.is_contribution_current(contribution):
                total = contribution.y if total is None else total + contribution.y
        return zeros((self.dim,)) if total is None else total

    def is_contribution_current(self, contribution: InformationContribution) -> bool:
        state = self.global_information_state
        return (
            contribution.epoch == state.epoch
            and contribution.operation_count == state.operation_count
            and contribution.operation_hash == state.operation_hash
        )

    def predict_identity(self, sys_noise_cov, sys_input=None, **kwargs):
        self.predict_linear(eye(self.dim), sys_noise_cov, sys_input, **kwargs)

    def predict_linear(
        self,
        system_matrix,
        sys_noise_cov,
        sys_input=None,
        *,
        input_by_node: Mapping | None = None,
        assume_zero_remote_inputs: bool = False,
    ):
        for contribution in self.contribution_bank.values():
            if not self.is_contribution_current(contribution):
                raise ValueError(
                    "Cannot predict with stale IDKF contributions in the bank"
                )

        A = atleast_2d(system_matrix)
        Q = atleast_2d(sys_noise_cov)
        Y_old = self.global_information_state.Y
        predicted_covariance = A @ linalg.solve(Y_old, A.T) + Q
        operation_hash = _extend_hash(
            self.global_information_state.operation_hash, "predict_linear"
        )

        for contribution in self.contribution_bank.values():
            explicit_input = self._input_for_contribution(
                contribution.key,
                sys_input,
                input_by_node,
                assume_zero_remote_inputs,
            )
            predicted_component = (
                A @ linalg.solve(Y_old, contribution.y) + explicit_input
            )
            contribution.y = linalg.solve(predicted_covariance, predicted_component)
            contribution.epoch += 1
            contribution.operation_count += 1
            contribution.operation_hash = operation_hash
            self.seen_contribution_ids.add(self._message_identity(contribution))

        state = self.global_information_state
        if state.transform_to_end is not None:
            state.transform_to_end = (
                state.transform_to_end @ Y_old @ linalg.inv(A) @ predicted_covariance
            )
        state.Y = linalg.inv(predicted_covariance)
        state.epoch += 1
        state.operation_count += 1
        state.operation_hash = operation_hash

    def predict_model(self, transition_model, **kwargs):
        system_matrix = _required(transition_model, "system_matrix")
        sys_noise_cov = _required(transition_model, "system_noise_cov", "sys_noise_cov")
        sys_input = _optional(transition_model, "sys_input", "system_input")
        self.predict_linear(system_matrix, sys_noise_cov, sys_input, **kwargs)

    def update_linear(
        self,
        measurement,
        measurement_matrix=None,
        meas_noise=None,
        *,
        measurement_models: Sequence | Mapping | None = None,
        contribution_key: ContributionKey | Hashable | None = None,
    ):
        key = (
            self.own_contribution_key
            if contribution_key is None
            else self._coerce_contribution_key(contribution_key)
        )
        if key not in self.contribution_bank:
            raise KeyError(f"Unknown IDKF contribution key: {key!r}")
        contribution = self.contribution_bank[key]
        if not self.is_contribution_current(contribution):
            raise ValueError("Cannot update a stale IDKF contribution")

        H, R = self._resolve_measurement_model(measurement_matrix, meas_noise)
        contribution.y = contribution.y + H.T @ linalg.solve(R, atleast_1d(measurement))

        all_models = (
            measurement_models
            if measurement_models is not None
            else self.measurement_models
        )
        if all_models is None:
            all_models = ((H, R),)
        state = self.global_information_state
        state.Y = state.Y + self._measurement_information_sum(all_models)
        state.operation_count += 1
        state.operation_hash = _extend_hash(state.operation_hash, "update_linear")

        contribution.operation_count = state.operation_count
        contribution.operation_hash = state.operation_hash
        contribution.epoch = state.epoch
        self.seen_contribution_ids.add(self._message_identity(contribution))

    def update_model(self, measurement_model, measurement, **kwargs):
        measurement_matrix, meas_noise = self._measurement_model_parts(
            measurement_model
        )
        self.update_linear(measurement, measurement_matrix, meas_noise, **kwargs)

    def export_contribution(
        self, key: ContributionKey | Hashable | None = None
    ) -> InformationContribution:
        contribution_key = (
            self.own_contribution_key
            if key is None
            else self._coerce_contribution_key(key)
        )
        return self.contribution_bank[contribution_key].copy()

    def receive_contribution(self, source) -> bool:
        if isinstance(source, IdkfNode):
            self._assert_global_information_compatible(source.global_information_state)
            changed = False
            for contribution in source.contribution_bank.values():
                if source.is_contribution_current(contribution):
                    changed = self.receive_contribution(contribution) or changed
            return changed

        if not isinstance(source, InformationContribution):
            raise TypeError("source must be an InformationContribution or IdkfNode")
        self._assert_contribution_compatible(source)
        identity = self._message_identity(source)
        current = self.contribution_bank.get(source.key)
        if identity in self.seen_contribution_ids or (
            current is not None and identity == self._message_identity(current)
        ):
            self.seen_contribution_ids.add(identity)
            return False
        self.contribution_bank[source.key] = source.copy()
        self.seen_contribution_ids.add(identity)
        return True

    def fuse_with(self, other: "IdkfNode") -> "IdkfNode":
        self.receive_contribution(other)
        return self

    def fused_copy(self, other: "IdkfNode") -> "IdkfNode":
        result = pycopy.deepcopy(self)
        result.fuse_with(other)
        return result

    def drop_stale_contributions(
        self, *, keep_own: bool = True
    ) -> frozenset[ContributionKey]:
        dropped = set()
        for key, contribution in list(self.contribution_bank.items()):
            if keep_own and key == self.own_contribution_key:
                continue
            if not self.is_contribution_current(contribution):
                dropped.add(key)
                del self.contribution_bank[key]
        return frozenset(dropped)

    def record_idkf_state(self, history_name="idkf_state"):
        snapshot = {
            "global_information_state": self.global_information_state.copy(),
            "contribution_bank": {
                key: contribution.copy()
                for key, contribution in self.contribution_bank.items()
            },
            "current_contribution_keys": self.current_contribution_keys,
            "stale_contribution_keys": self.stale_contribution_keys,
        }
        return self.record_history(history_name, snapshot)

    def _input_for_contribution(
        self,
        key: ContributionKey,
        sys_input,
        input_by_node: Mapping | None,
        assume_zero_remote_inputs: bool,
    ):
        if input_by_node is not None:
            if key in input_by_node:
                return atleast_1d(input_by_node[key])
            if key.origin_node_id in input_by_node:
                return atleast_1d(input_by_node[key.origin_node_id])
            return zeros((self.dim,))
        if key == self.own_contribution_key:
            return zeros((self.dim,)) if sys_input is None else atleast_1d(sys_input)
        if sys_input is not None and not assume_zero_remote_inputs:
            raise ValueError(
                "Provide input_by_node or set assume_zero_remote_inputs=True for remote contributions"
            )
        return zeros((self.dim,))

    def _resolve_measurement_model(self, measurement_matrix, meas_noise):
        H = (
            self.measurement_matrix
            if measurement_matrix is None
            else measurement_matrix
        )
        R = self.meas_noise if meas_noise is None else meas_noise
        if H is None or R is None:
            raise ValueError("measurement_matrix and meas_noise must be provided")
        return atleast_2d(H), atleast_2d(R)

    def _measurement_information_sum(self, models: Sequence | Mapping):
        total = None
        for H, R in self._iter_measurement_models(models):
            info = H.T @ linalg.solve(R, H)
            total = info if total is None else total + info
        return zeros((self.dim, self.dim)) if total is None else total

    def _iter_measurement_models(self, models: Sequence | Mapping):
        iterable = models.values() if isinstance(models, Mapping) else models
        for model in iterable:
            yield self._measurement_model_parts(model)

    @staticmethod
    def _measurement_model_parts(model):
        if isinstance(model, tuple) and len(model) == 2:
            return atleast_2d(model[0]), atleast_2d(model[1])
        if isinstance(model, Mapping):
            H = model.get("measurement_matrix", model.get("H"))
            R = model.get(
                "meas_noise", model.get("measurement_noise_cov", model.get("R"))
            )
            if H is None or R is None:
                raise AttributeError(
                    "Measurement model mappings must expose H/measurement_matrix and R/meas_noise"
                )
            return atleast_2d(H), atleast_2d(R)
        H = _required(model, "measurement_matrix", "H")
        R = _required(model, "meas_noise", "measurement_noise_cov", "R")
        return atleast_2d(H), atleast_2d(R)

    def _coerce_contribution_key(
        self, key: ContributionKey | Hashable
    ) -> ContributionKey:
        if isinstance(key, ContributionKey):
            return key
        return ContributionKey(key, self.own_contribution_key.track_id)

    @staticmethod
    def _message_identity(contribution: InformationContribution):
        return (
            contribution.key,
            contribution.epoch,
            contribution.operation_count,
            contribution.operation_hash,
        )

    def _assert_contribution_compatible(self, contribution: InformationContribution):
        state = self.global_information_state
        if contribution.epoch != state.epoch:
            raise ValueError("Cannot fuse IDKF contributions from different epochs")
        if contribution.operation_count != state.operation_count:
            raise ValueError(
                "Cannot fuse IDKF contributions with different operation counts"
            )
        if contribution.operation_hash != state.operation_hash:
            raise ValueError(
                "Cannot fuse IDKF contributions with different operation histories"
            )

    def _assert_global_information_compatible(
        self, other_state: GlobalInformationState
    ):
        self._assert_contribution_compatible(
            InformationContribution(
                key=self.own_contribution_key,
                y=zeros((self.dim,)),
                epoch=other_state.epoch,
                operation_count=other_state.operation_count,
                operation_hash=other_state.operation_hash,
            )
        )
        if not allclose(
            other_state.Y,
            self.global_information_state.Y,
            rtol=self.compatibility_rtol,
            atol=self.compatibility_atol,
        ):
            raise ValueError(
                "Cannot fuse IDKF nodes with different global information matrices"
            )


InformationFormDistributedKalmanNode = IdkfNode
IDKFNode = IdkfNode

__all__ = [
    "ContributionKey",
    "GlobalInformationState",
    "IDKFNode",
    "IdkfNode",
    "InformationContribution",
    "InformationFormDistributedKalmanNode",
]
