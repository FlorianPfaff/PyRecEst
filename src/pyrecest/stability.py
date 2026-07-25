"""Runtime metadata helpers for public API stability."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from operator import index as _operator_index
from typing import Final, Literal, ParamSpec, TypeVar

from pyrecest.backend_support._jax_assignment_numpy_index_contract import (
    patch_jax_assignment_numpy_index_contract as _patch_jax_assignment_numpy_index_contract,
)
from pyrecest.backend_support._pytorch_allclose_device_contract import (
    patch_pytorch_allclose_device_contract as _patch_pytorch_allclose_device_contract,
)
from pyrecest.backend_support._pytorch_dot_outer_device_contract import (
    patch_pytorch_dot_outer_device_contract as _patch_pytorch_dot_outer_device_contract,
)
from pyrecest.backend_support._pytorch_matmul_device_contract import (
    patch_pytorch_matmul_device_contract as _patch_pytorch_matmul_device_contract,
)
from pyrecest.backend_support._pytorch_minmax_device_contract import (
    patch_pytorch_minmax_device_contract as _patch_pytorch_minmax_device_contract,
)
from pyrecest.backend_support._pytorch_one_hot_scalar_contract import (
    patch_pytorch_one_hot_scalar_contract as _patch_pytorch_one_hot_scalar_contract,
)
from pyrecest.backend_support._pytorch_stack_helpers_device_contract import (
    patch_pytorch_stack_helpers_device_contract as _patch_pytorch_stack_helpers_device_contract,
)
from pyrecest.backend_support._pytorch_trapezoid_numpy_contract import (
    patch_pytorch_trapezoid_numpy_contract as _patch_pytorch_trapezoid_numpy_contract,
)
from pyrecest.backend_support._pytorch_where_device_contract import (
    patch_pytorch_where_device_contract as _patch_pytorch_where_device_contract,
)


def _patch_pytorch_raw_comparison_arraylike_contract() -> None:
    """Patch raw/public PyTorch comparisons to accept array-like operands."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    helper_names = ("greater", "less", "logical_or")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_arraylike_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def _preferred_pytorch_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _coerce_binary_args(x, y):
        device = _preferred_pytorch_device(x, y)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        return x, y

    def _wrap_comparison(helper_name, torch_func):
        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = helper_name
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        comparison._pyrecest_arraylike_contract = True
        return comparison

    helpers = {
        "greater": _wrap_comparison("greater", torch.greater),
        "less": _wrap_comparison("less", torch.less),
        "logical_or": _wrap_comparison("logical_or", torch.logical_or),
    }
    for helper_name, helper in helpers.items():
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_pytorch_array_equal_equal_nan_contract() -> None:
    """Patch raw/public PyTorch ``array_equal`` to accept ``equal_nan``."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_array_equal = getattr(raw_pytorch, "array_equal", None)
    if original_array_equal is None:
        return
    if getattr(original_array_equal, "_pyrecest_equal_nan_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.array_equal = original_array_equal
        return

    def _preferred_pytorch_device(*values):
        for value in values:
            if torch.is_tensor(value) and value.device.type != "cpu":
                return value.device
        for value in values:
            if torch.is_tensor(value):
                return value.device
        return None

    def _coerce_array_equal_arg(value, *, device):
        if not torch.is_tensor(value):
            return torch.as_tensor(value, device=device)
        if device is not None and value.device != device:
            return value.to(device=device)
        return value

    def array_equal(a, b, equal_nan=False):
        device = _preferred_pytorch_device(a, b)
        a = _coerce_array_equal_arg(a, device=device)
        b = _coerce_array_equal_arg(b, device=device)
        if tuple(a.shape) != tuple(b.shape):
            return False

        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        if not equal_nan:
            return torch.equal(a, b)

        comparison = torch.eq(a, b)
        if dtype.is_floating_point or dtype.is_complex:
            comparison = comparison | (torch.isnan(a) & torch.isnan(b))
        return bool(torch.all(comparison))

    array_equal.__name__ = getattr(original_array_equal, "__name__", "array_equal")
    array_equal.__doc__ = getattr(original_array_equal, "__doc__", None)
    array_equal._pyrecest_equal_nan_contract = True
    array_equal._pyrecest_numpy_contract = True
    raw_pytorch.array_equal = array_equal
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.array_equal = array_equal


def _patch_pytorch_diag_numpy_contract() -> None:
    """Patch raw/public PyTorch ``diag`` to accept NumPy-style inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_diag = getattr(raw_pytorch, "diag", None)
    if original_diag is None:
        return
    if getattr(original_diag, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.diag = original_diag
        return

    def diag(v, k=0):
        return torch.diag(raw_pytorch.array(v), diagonal=k)

    diag.__name__ = getattr(original_diag, "__name__", "diag")
    diag.__doc__ = getattr(original_diag, "__doc__", None)
    diag._pyrecest_numpy_contract = True
    raw_pytorch.diag = diag
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.diag = diag


def _patch_pytorch_vec_to_diag_numpy_contract() -> None:
    """Patch raw/public PyTorch ``vec_to_diag`` to accept array-like inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_vec_to_diag = getattr(raw_pytorch, "vec_to_diag", None)
    if original_vec_to_diag is None:
        return
    if getattr(original_vec_to_diag, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.vec_to_diag = original_vec_to_diag
        return

    def vec_to_diag(vec):
        return original_vec_to_diag(raw_pytorch.array(vec))

    vec_to_diag.__name__ = getattr(original_vec_to_diag, "__name__", "vec_to_diag")
    vec_to_diag.__doc__ = getattr(original_vec_to_diag, "__doc__", None)
    vec_to_diag._pyrecest_numpy_contract = True
    raw_pytorch.vec_to_diag = vec_to_diag
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.vec_to_diag = vec_to_diag


def _patch_pytorch_triangular_vec_numpy_contract() -> None:
    """Patch raw/public PyTorch triangular-vector helpers for array-like inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    def _make_triangular_to_vec(helper_name, index_helper, original_helper):
        def triangular_to_vec(x, k=0):
            x = raw_pytorch.array(x)
            n = x.shape[-1]
            rows, cols = index_helper(n, k=k)
            rows = rows.to(device=x.device)
            cols = cols.to(device=x.device)
            return x[..., rows, cols]

        triangular_to_vec.__name__ = getattr(original_helper, "__name__", helper_name)
        triangular_to_vec.__doc__ = getattr(original_helper, "__doc__", None)
        triangular_to_vec._pyrecest_numpy_contract = True
        return triangular_to_vec

    for helper_name, index_helper_name in (
        ("tril_to_vec", "tril_indices"),
        ("triu_to_vec", "triu_indices"),
    ):
        original_helper = getattr(raw_pytorch, helper_name, None)
        index_helper = getattr(raw_pytorch, index_helper_name, None)
        if original_helper is None or index_helper is None:
            continue
        if getattr(original_helper, "_pyrecest_numpy_contract", False):
            if active_pytorch_backend:
                setattr(backend, helper_name, original_helper)
            continue

        helper = _make_triangular_to_vec(helper_name, index_helper, original_helper)
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)


def _patch_pytorch_arctan_numpy_contract() -> None:
    """Patch raw/public PyTorch ``arctan`` to accept NumPy-style inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_arctan = getattr(raw_pytorch, "arctan", None)
    if original_arctan is None:
        return
    if getattr(original_arctan, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.arctan = original_arctan
        return

    def arctan(a, *args, **kwargs):
        return torch.arctan(raw_pytorch.array(a), *args, **kwargs)

    arctan.__name__ = getattr(original_arctan, "__name__", "arctan")
    arctan.__doc__ = getattr(original_arctan, "__doc__", None)
    arctan._pyrecest_numpy_contract = True
    raw_pytorch.arctan = arctan
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.arctan = arctan


def _patch_pytorch_special_arraylike_contract() -> None:
    """Patch raw/public PyTorch special helpers to accept array-like inputs."""
    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    helper_names = ("erf", "gammaln", "gamma", "polygamma")
    if all(
        getattr(
            getattr(raw_pytorch, helper_name, None),
            "_pyrecest_arraylike_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if active_pytorch_backend:
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_pytorch, helper_name))
        return

    def _return_or_store_out(result, out):
        if out is None:
            return result
        copy_ = getattr(out, "copy_", None)
        if copy_ is not None:
            copy_(result)
        else:
            out[...] = raw_pytorch.to_numpy(result)
        return out

    def _as_real_gamma_input(a):
        values = raw_pytorch.array(a)
        if raw_pytorch.is_floating(values):
            return values
        if raw_pytorch.is_complex(values):
            raise TypeError("gamma is only supported for real-valued PyTorch inputs")
        return raw_pytorch.cast(values, dtype=raw_pytorch.get_default_dtype())

    def erf(a, out=None):
        result = torch.erf(raw_pytorch.array(a))
        return _return_or_store_out(result, out)

    def gammaln(a, out=None):
        result = torch.special.gammaln(raw_pytorch.array(a))
        return _return_or_store_out(result, out)

    def gamma(a, out=None):
        values = _as_real_gamma_input(a)
        positive_branch = torch.exp(torch.special.gammaln(values))
        reflected_branch = torch.pi / (
            torch.sin(torch.pi * values) * torch.exp(torch.special.gammaln(1 - values))
        )
        result = torch.where(values < 0, reflected_branch, positive_branch)

        zero_mask = values == 0
        signed_zero_inf = torch.where(
            torch.signbit(values),
            torch.full_like(values, -torch.inf),
            torch.full_like(values, torch.inf),
        )
        result = torch.where(zero_mask, signed_zero_inf, result)

        negative_integer_mask = (values < 0) & (values == torch.floor(values))
        result = torch.where(
            negative_integer_mask,
            torch.full_like(values, torch.nan),
            result,
        )
        return _return_or_store_out(result, out)

    def polygamma(n, a, out=None):
        result = torch.polygamma(_operator_index(n), raw_pytorch.array(a))
        return _return_or_store_out(result, out)

    helpers = {
        "erf": (erf, torch.erf),
        "gammaln": (gammaln, torch.special.gammaln),
        "gamma": (gamma, getattr(raw_pytorch, "gamma", None)),
        "polygamma": (polygamma, torch.polygamma),
    }
    for helper_name, (helper, original_helper) in helpers.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(original_helper, "__doc__", None)
        helper._pyrecest_arraylike_contract = True
        setattr(raw_pytorch, helper_name, helper)
        if active_pytorch_backend:
            setattr(backend, helper_name, helper)
    raw_pytorch._gammaln = gammaln  # pylint: disable=protected-access


def _jax_squeeze_axis(axis) -> int:
    """Return one NumPy-style integer squeeze axis."""
    if isinstance(axis, bool) or type(axis).__name__ == "bool_":
        raise TypeError("axis must be an integer or a tuple of integers")
    try:
        return _operator_index(axis)
    except TypeError as exc:
        raise TypeError("axis must be an integer or a tuple of integers") from exc


def _jax_squeeze_axes(axis, jnp) -> tuple[int, ...]:
    """Normalize NumPy-style squeeze axes for the JAX backend."""
    try:
        return (_jax_squeeze_axis(axis),)
    except TypeError:
        pass

    axis_array = jnp.asarray(axis)
    if axis_array.shape == ():
        return (_jax_squeeze_axis(axis_array.item()),)
    return tuple(_jax_squeeze_axis(one_axis) for one_axis in axis_array.tolist())


def _patch_jax_squeeze_numpy_contract() -> None:
    """Patch raw/public JAX ``squeeze`` to keep non-singleton explicit axes."""
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_squeeze = getattr(raw_jax, "squeeze", None)
    if original_squeeze is None:
        return
    if getattr(original_squeeze, "_pyrecest_numpy_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend.squeeze = original_squeeze
        return

    def squeeze(a, axis=None):
        a = jnp.asarray(a)
        if axis is None:
            return original_squeeze(a, axis=None)

        axes = _jax_squeeze_axes(axis, jnp)
        if not axes:
            return a

        normalized_axes = tuple(
            one_axis + a.ndim if one_axis < 0 else one_axis for one_axis in axes
        )
        for one_axis, normalized_axis in zip(axes, normalized_axes):
            if normalized_axis < 0 or normalized_axis >= a.ndim:
                raise ValueError(
                    f"axis {one_axis} is out of bounds for array of dimension {a.ndim}"
                )
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("duplicate value in 'axis'")
        if any(a.shape[one_axis] != 1 for one_axis in normalized_axes):
            return a

        squeeze_axis = (
            normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
        )
        return original_squeeze(a, axis=squeeze_axis)

    squeeze.__name__ = getattr(original_squeeze, "__name__", "squeeze")
    squeeze.__doc__ = getattr(original_squeeze, "__doc__", None)
    squeeze._pyrecest_numpy_contract = True
    raw_jax.squeeze = squeeze
    if getattr(backend, "__backend_name__", None) == "jax":
        backend.squeeze = squeeze


def _patch_shared_numpy_matrix_helpers_arraylike_contract() -> None:
    """Patch shared NumPy/autograd matrix helpers to accept array-like inputs."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import should be available
        return

    if getattr(backend, "__backend_name__", None) not in {"autograd", "numpy"}:
        return

    try:
        import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - shared backend should be available
        return

    original_helpers = {
        "vec_to_diag": getattr(shared_numpy, "vec_to_diag", None),
        "tril_to_vec": getattr(shared_numpy, "tril_to_vec", None),
        "triu_to_vec": getattr(shared_numpy, "triu_to_vec", None),
    }
    if any(helper is None for helper in original_helpers.values()):
        return
    if all(
        getattr(helper, "_pyrecest_arraylike_contract", False)
        for helper in original_helpers.values()
    ):
        for helper_name in original_helpers:
            setattr(backend, helper_name, getattr(shared_numpy, helper_name))
        return

    def vec_to_diag(vec):
        vec = shared_numpy.array(vec)
        d = vec.shape[-1]
        return vec[..., :, None] * shared_numpy.eye(d, dtype=vec.dtype)

    def tril_to_vec(x, k=0):
        x = shared_numpy.array(x)
        n = x.shape[-1]
        rows, cols = shared_numpy._np.tril_indices(n, k=k)
        return x[..., rows, cols]

    def triu_to_vec(x, k=0):
        x = shared_numpy.array(x)
        n = x.shape[-1]
        rows, cols = shared_numpy._np.triu_indices(n, k=k)
        return x[..., rows, cols]

    helpers = {
        "vec_to_diag": vec_to_diag,
        "tril_to_vec": tril_to_vec,
        "triu_to_vec": triu_to_vec,
    }
    for helper_name, helper in helpers.items():
        original_helper = original_helpers[helper_name]
        helper.__name__ = getattr(original_helper, "__name__", helper_name)
        helper.__doc__ = getattr(original_helper, "__doc__", None)
        helper._pyrecest_arraylike_contract = True
        setattr(shared_numpy, helper_name, helper)
        setattr(backend, helper_name, helper)


def _patch_jax_triangular_vector_helpers_arraylike_contract() -> None:
    """Patch raw/public JAX triangular vector helpers for array-like matrices."""
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    helper_names = ("tril_to_vec", "triu_to_vec")
    if all(
        getattr(
            getattr(raw_jax, helper_name, None),
            "_pyrecest_arraylike_contract",
            False,
        )
        for helper_name in helper_names
    ):
        if getattr(backend, "__backend_name__", None) == "jax":
            for helper_name in helper_names:
                setattr(backend, helper_name, getattr(raw_jax, helper_name))
        return

    original_helpers = {
        helper_name: getattr(raw_jax, helper_name, None) for helper_name in helper_names
    }
    if any(helper is None for helper in original_helpers.values()):
        return

    def tril_to_vec(x, k=0):
        x = jnp.asarray(x)
        n = x.shape[-1]
        rows, cols = jnp.tril_indices(n, k=k)
        return x[..., rows, cols]

    def triu_to_vec(x, k=0):
        x = jnp.asarray(x)
        n = x.shape[-1]
        rows, cols = jnp.triu_indices(n, k=k)
        return x[..., rows, cols]

    helpers = {
        "tril_to_vec": tril_to_vec,
        "triu_to_vec": triu_to_vec,
    }
    for helper_name, helper in helpers.items():
        original_helper = original_helpers[helper_name]
        helper.__name__ = getattr(original_helper, "__name__", helper_name)
        helper.__doc__ = getattr(original_helper, "__doc__", None)
        helper._pyrecest_arraylike_contract = True
        setattr(raw_jax, helper_name, helper)
        if getattr(backend, "__backend_name__", None) == "jax":
            setattr(backend, helper_name, helper)


_patch_jax_assignment_numpy_index_contract()
_patch_pytorch_allclose_device_contract()
_patch_pytorch_diag_numpy_contract()
_patch_pytorch_vec_to_diag_numpy_contract()
_patch_pytorch_triangular_vec_numpy_contract()
_patch_pytorch_arctan_numpy_contract()
_patch_pytorch_special_arraylike_contract()
_patch_pytorch_raw_comparison_arraylike_contract()
_patch_pytorch_array_equal_equal_nan_contract()
_patch_pytorch_dot_outer_device_contract()
_patch_pytorch_matmul_device_contract()
_patch_pytorch_minmax_device_contract()
_patch_pytorch_one_hot_scalar_contract()
_patch_shared_numpy_matrix_helpers_arraylike_contract()
_patch_pytorch_stack_helpers_device_contract()
_patch_pytorch_trapezoid_numpy_contract()
_patch_pytorch_where_device_contract()
_patch_jax_squeeze_numpy_contract()
_patch_jax_triangular_vector_helpers_arraylike_contract()

P = ParamSpec("P")
R = TypeVar("R")

StabilityLevel = Literal[
    "stable", "experimental", "deprecated", "backend-specific", "internal"
]
STABILITY_LEVELS: Final = (
    "stable",
    "experimental",
    "deprecated",
    "backend-specific",
    "internal",
)


@dataclass(frozen=True)
class PublicAPIStatus:
    """Stability metadata for a public API entry."""

    name: str
    level: StabilityLevel
    since: str | None = None
    remove_in: str | None = None
    replacement: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.level not in STABILITY_LEVELS:
            raise ValueError(f"Unknown stability level: {self.level!r}")

    def to_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation."""
        return asdict(self)


_PUBLIC_API_STATUS: Final[dict[str, PublicAPIStatus]] = {
    "KalmanFilter": PublicAPIStatus(
        "KalmanFilter", "stable", since="2.2.0", notes="Core linear Gaussian filter."
    ),
    "GaussianDistribution": PublicAPIStatus(
        "GaussianDistribution",
        "stable",
        since="2.2.0",
        notes="Core Euclidean distribution.",
    ),
    "pyrecest.backend": PublicAPIStatus(
        "pyrecest.backend",
        "backend-specific",
        since="2.2.0",
        notes="Support depends on the backend capability matrix.",
    ),
    "UKFOnManifolds": PublicAPIStatus(
        "UKFOnManifolds",
        "backend-specific",
        since="2.2.0",
        notes="Backend exclusions are documented in the backend API matrix.",
    ),
}


def stability(
    level: StabilityLevel,
    *,
    since: str | None = None,
    remove_in: str | None = None,
    replacement: str | None = None,
    notes: str = "",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach stability metadata to a function, method, or class."""
    if level not in STABILITY_LEVELS:
        raise ValueError(f"Unknown stability level: {level!r}")

    def decorator(obj: Callable[P, R]) -> Callable[P, R]:
        status = PublicAPIStatus(
            name=f"{obj.__module__}.{obj.__qualname__}",
            level=level,
            since=since,
            remove_in=remove_in,
            replacement=replacement,
            notes=notes,
        )
        setattr(obj, "__pyrecest_stability__", status)
        return obj

    return decorator


def get_public_api_status(name: object) -> PublicAPIStatus | None:
    """Return registered stability metadata for a public API name."""
    if not isinstance(name, str):
        return None
    return _PUBLIC_API_STATUS.get(name)


def iter_public_api_status() -> Iterable[PublicAPIStatus]:
    """Iterate registered public API stability rows in stable name order."""
    return tuple(_PUBLIC_API_STATUS[name] for name in sorted(_PUBLIC_API_STATUS))


__all__ = [
    "PublicAPIStatus",
    "STABILITY_LEVELS",
    "StabilityLevel",
    "get_public_api_status",
    "iter_public_api_status",
    "stability",
]
