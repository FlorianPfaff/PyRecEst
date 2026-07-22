from importlib.metadata import PackageNotFoundError, version
from operator import index as _operator_index

import pyrecest._backend  # noqa
from pyrecest._backend_submodules import (  # noqa: F401
    register_backend_submodules as _register_backend_submodules,
)
from pyrecest.backend import copy  # noqa: F401

_register_backend_submodules()


def _patch_shared_numpy_copy_facade() -> None:
    """Make shared NumPy backend copy accept scalar and array-like inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) not in {"autograd", "numpy"}:
        return

    original_copy = backend.copy

    def copy_arraylike(x):
        return original_copy(backend.array(x))

    copy_arraylike.__name__ = getattr(original_copy, "__name__", "copy")
    copy_arraylike.__doc__ = getattr(original_copy, "__doc__", None)
    backend.copy = copy_arraylike
    globals()["copy"] = backend.copy


def _patch_shared_numpy_squeeze_facade() -> None:
    """Make shared NumPy squeeze reject out-of-bounds axes before shape access."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) not in {"autograd", "numpy"}:
        return

    import pyrecest._backend._shared_numpy as shared_numpy  # pylint: disable=import-outside-toplevel

    original_squeeze = shared_numpy.squeeze
    np_module = shared_numpy._np

    def _normalize_squeeze_axes(axis):
        if isinstance(axis, (int, np_module.integer)):
            return (int(axis),)
        axis_array = np_module.asarray(axis)
        if axis_array.shape == ():
            try:
                return (_operator_index(axis_array),)
            except TypeError as exc:
                raise TypeError(
                    "only integer scalar arrays can be converted to a scalar index"
                ) from exc
        return tuple(axis)

    shared_numpy._normalize_squeeze_axes = _normalize_squeeze_axes

    def _axis_out_of_bounds_error(axis, ndim):
        axis_error = getattr(getattr(np_module, "exceptions", None), "AxisError", None)
        if axis_error is None:
            axis_error = getattr(np_module, "AxisError", None)
        if axis_error is None:
            return ValueError(
                f"axis {axis} is out of bounds for array of dimension {ndim}"
            )
        try:
            return axis_error(axis, ndim=ndim)
        except TypeError:  # pragma: no cover - compatibility with older NumPy APIs
            return axis_error(axis, ndim)

    def squeeze(x, axis=None):
        x_arr = np_module.asarray(x)
        if axis is None:
            return original_squeeze(x_arr, axis=None)

        axes = _normalize_squeeze_axes(axis)
        if not axes:
            return x_arr

        normalized_axes = []
        for one_axis in axes:
            if isinstance(one_axis, (int, np_module.integer)):
                one_axis = int(one_axis)
                normalized_axis = one_axis + x_arr.ndim if one_axis < 0 else one_axis
                if normalized_axis < 0 or normalized_axis >= x_arr.ndim:
                    raise _axis_out_of_bounds_error(one_axis, x_arr.ndim)
                normalized_axes.append(normalized_axis)
            else:
                normalized_axes.append(one_axis)
        normalized_axes = tuple(normalized_axes)

        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("duplicate value in 'axis'")
        if any(x_arr.shape[one_axis] != 1 for one_axis in normalized_axes):
            return x_arr
        squeeze_axis = (
            normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
        )
        return np_module.squeeze(x_arr, axis=squeeze_axis)

    squeeze.__name__ = getattr(original_squeeze, "__name__", "squeeze")
    squeeze.__doc__ = getattr(original_squeeze, "__doc__", None)
    shared_numpy.squeeze = squeeze
    backend.squeeze = squeeze


def _patch_set_diag_arraylike_facade() -> None:
    """Make public and raw PyTorch set_diag accept array-like matrix inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    original_set_diag = backend.set_diag

    def set_diag(x, new_diag):
        if not backend.is_array(x):
            x = backend.array(x)
        return original_set_diag(x, new_diag)

    set_diag.__name__ = getattr(original_set_diag, "__name__", "set_diag")
    set_diag.__doc__ = getattr(original_set_diag, "__doc__", None)
    backend.set_diag = set_diag

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    raw_set_diag = getattr(pytorch_backend, "set_diag", None)
    if raw_set_diag is None:
        return
    if getattr(raw_set_diag, "_pyrecest_arraylike_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.set_diag = raw_set_diag
        return

    def raw_pytorch_set_diag(x, new_diag):
        x = pytorch_backend.array(x)
        diag_len = min(x.shape[-2], x.shape[-1])
        result = x.clone()
        diag_indices = _torch.arange(diag_len, device=x.device)
        values = _torch.as_tensor(new_diag, dtype=x.dtype, device=x.device)
        result[..., diag_indices, diag_indices] = values
        return result

    raw_pytorch_set_diag.__name__ = getattr(raw_set_diag, "__name__", "set_diag")
    raw_pytorch_set_diag.__doc__ = getattr(raw_set_diag, "__doc__", None)
    raw_pytorch_set_diag._pyrecest_arraylike_contract = True
    pytorch_backend.set_diag = raw_pytorch_set_diag
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.set_diag = raw_pytorch_set_diag


def _patch_pytorch_one_hot_integer_label_facade() -> None:
    """Make PyTorch one_hot accept all integer label tensor dtypes."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    original_one_hot = getattr(pytorch_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_integer_label_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.one_hot = original_one_hot
        return

    def one_hot(labels, num_classes):
        if _torch.is_tensor(labels):
            if (
                labels.dtype == _torch.bool
                or labels.dtype.is_floating_point
                or labels.dtype.is_complex
            ):
                return original_one_hot(labels, _operator_index(num_classes))
            labels = labels.to(dtype=_torch.long)
        else:
            labels = _torch.LongTensor(labels)
        return _torch.nn.functional.one_hot(
            labels,
            _operator_index(num_classes),
        ).type(_torch.uint8)

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_integer_label_contract = True
    pytorch_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.one_hot = one_hot


_patch_shared_numpy_copy_facade()
_patch_shared_numpy_squeeze_facade()
_patch_set_diag_arraylike_facade()
_patch_pytorch_one_hot_integer_label_facade()


def _patch_pytorch_comparison_facade() -> None:
    """Make public and raw PyTorch comparison helpers accept array-like inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    def _coerce_binary_args(x, y):
        device = next(
            (value.device for value in (x, y) if _torch.is_tensor(value)),
            None,
        )
        if not _torch.is_tensor(x):
            x = _torch.as_tensor(x, device=device)
        elif device is not None and x.device != device:
            x = x.to(device=device)
        if not _torch.is_tensor(y):
            y = _torch.as_tensor(y, device=device)
        elif device is not None and y.device != device:
            y = y.to(device=device)
        return x, y

    def _wrap_comparison(torch_func):
        def comparison(x, y, **kwargs):
            x, y = _coerce_binary_args(x, y)
            return torch_func(x, y, **kwargs)

        comparison.__name__ = getattr(torch_func, "__name__", "comparison")
        comparison.__doc__ = getattr(torch_func, "__doc__", None)
        return comparison

    greater = _wrap_comparison(_torch.greater)
    less = _wrap_comparison(_torch.less)
    logical_or = _wrap_comparison(_torch.logical_or)

    backend.greater = raw_pytorch.greater = greater
    backend.less = raw_pytorch.less = less
    backend.logical_or = raw_pytorch.logical_or = logical_or


def _patch_pytorch_clip_facade() -> None:
    """Make public and raw PyTorch ``clip`` accept array-like inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    def _clip_bound(value, *, device):
        if value is None:
            return None
        if _torch.is_tensor(value):
            return value.to(device=device)
        return _torch.as_tensor(value, device=device)

    def clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None):
        if min is not None:
            if a_min is not None:
                raise TypeError("clip() got both 'a_min' and 'min'")
            a_min = min
        if max is not None:
            if a_max is not None:
                raise TypeError("clip() got both 'a_max' and 'max'")
            a_max = max
        if a_min is None and a_max is None:
            raise ValueError("One of max or min must be given")

        x = backend.array(a)
        result = _torch.clip(
            x,
            min=_clip_bound(a_min, device=x.device),
            max=_clip_bound(a_max, device=x.device),
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    clip.__name__ = getattr(_torch.clip, "__name__", "clip")
    clip.__doc__ = getattr(_torch.clip, "__doc__", None)
    pytorch_backend.clip = clip
    backend.clip = clip


def _pytorch_tile_repetition(repetition) -> int:
    """Return one NumPy-style tile repetition as an integer."""

    try:
        return _operator_index(repetition)
    except TypeError as exc:
        raise TypeError("tile repetitions must be integers") from exc


def _pytorch_tile_repetitions(reps, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style tile repetitions for ``torch.Tensor.repeat``."""

    if torch_module.is_tensor(reps):
        reps = reps.detach().cpu().numpy()
    reps_array = numpy_module.asarray(reps)
    if reps_array.shape == ():
        repetitions = (_pytorch_tile_repetition(reps_array.item()),)
    else:
        repetitions = tuple(
            _pytorch_tile_repetition(one_repetition)
            for one_repetition in reps_array.tolist()
        )
    if any(one_repetition < 0 for one_repetition in repetitions):
        raise ValueError("negative dimensions are not allowed")
    return repetitions


def _patch_pytorch_tile_facade() -> None:
    """Make public and raw PyTorch ``tile`` follow NumPy repetition semantics."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as _pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    def tile(x, reps):
        x = _pytorch_backend.array(x)
        repetitions = _pytorch_tile_repetitions(reps, _np, _torch)
        if not repetitions:
            return x.clone()
        if x.ndim < len(repetitions):
            x = x.reshape((1,) * (len(repetitions) - x.ndim) + tuple(x.shape))
        elif x.ndim > len(repetitions):
            repetitions = (1,) * (x.ndim - len(repetitions)) + repetitions
        return x.repeat(repetitions)

    tile.__name__ = "tile"
    tile.__doc__ = getattr(_np.tile, "__doc__", None)
    _pytorch_backend.tile = tile
    if active_pytorch_backend:
        backend.tile = tile


def _patch_pytorch_stack_helpers_facade() -> None:
    """Make public PyTorch stack helpers accept NumPy-style array-like inputs."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import numpy as _np  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    def _tensor_sequence(tup):
        return [backend.array(item) for item in tup]

    def hstack(tup):
        tensors = [_torch.atleast_1d(tensor) for tensor in _tensor_sequence(tup)]
        if not tensors:
            return _torch.cat(tensors, dim=0)
        return _torch.cat(tensors, dim=0 if tensors[0].ndim == 1 else 1)

    def vstack(tup):
        tensors = [_torch.atleast_2d(tensor) for tensor in _tensor_sequence(tup)]
        return _torch.cat(tensors, dim=0)

    def column_stack(tup):
        tensors = []
        for tensor in _tensor_sequence(tup):
            if tensor.ndim < 2:
                tensor = tensor.reshape(-1, 1)
            tensors.append(tensor)
        return _torch.cat(tensors, dim=1)

    def dstack(tup):
        tensors = [_torch.atleast_3d(tensor) for tensor in _tensor_sequence(tup)]
        return _torch.cat(tensors, dim=2)

    for helper_name, helper in {
        "hstack": hstack,
        "vstack": vstack,
        "column_stack": column_stack,
        "dstack": dstack,
    }.items():
        helper.__name__ = helper_name
        helper.__doc__ = getattr(_np, helper_name).__doc__
        setattr(backend, helper_name, helper)


def _patch_pytorch_linear_helpers_facade() -> None:
    """Make public and raw PyTorch linear helpers follow backend contracts."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "pytorch":
        return

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import fails first
        return

    def _promoted_pair(a, b):
        a = pytorch_backend.array(a)
        b = pytorch_backend.array(b)
        return pytorch_backend.convert_to_wider_dtype([a, b])

    def dot(a, b):
        a, b = _promoted_pair(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return _torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return _torch.dot(a, b)
        if b.ndim == 1:
            return _torch.einsum("...i,i->...", a, b)
        if a.ndim == 1:
            return _torch.einsum("i,...i->...", a, b)
        return _torch.einsum("...i,...i->...", a, b)

    def outer(a, b):
        a, b = _promoted_pair(a, b)
        if a.ndim == 0 or b.ndim == 0:
            return _torch.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    dot.__name__ = "dot"
    dot.__doc__ = getattr(pytorch_backend.dot, "__doc__", None)
    outer.__name__ = "outer"
    outer.__doc__ = getattr(pytorch_backend.outer, "__doc__", None)
    backend.dot = pytorch_backend.dot = dot
    backend.outer = pytorch_backend.outer = outer


def _patch_raw_pytorch_cumulative_facade() -> None:
    """Make raw PyTorch cumulative helpers accept NumPy's ``out`` argument."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        from pyrecest._backend_submodules import (  # pylint: disable=import-outside-toplevel
            _copy_result_to_out,
        )
    except ModuleNotFoundError:  # pragma: no cover - only relevant without PyTorch
        return

    def _wrap_cumulative(cumulative):
        if getattr(cumulative, "_pyrecest_out_contract", False):
            return cumulative

        def wrapped_cumulative(x, axis=None, dtype=None, out=None):
            result = cumulative(x, axis=axis, dtype=dtype)
            if out is not None:
                return _copy_result_to_out(result, out)
            return result

        wrapped_cumulative.__name__ = getattr(cumulative, "__name__", "cumulative")
        wrapped_cumulative.__doc__ = getattr(cumulative, "__doc__", None)
        wrapped_cumulative._pyrecest_out_contract = True
        return wrapped_cumulative

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    selected_backend_is_pytorch = (
        getattr(backend, "__backend_name__", None) == "pytorch"
    )
    for helper_name in ("cumsum", "cumprod"):
        cumulative = getattr(pytorch_backend, helper_name, None)
        if cumulative is None:
            continue
        wrapped_cumulative = _wrap_cumulative(cumulative)
        setattr(pytorch_backend, helper_name, wrapped_cumulative)
        if selected_backend_is_pytorch:
            setattr(backend, helper_name, wrapped_cumulative)


def _patch_raw_pytorch_trace_facade() -> None:
    """Make raw PyTorch ``trace`` follow NumPy's trace signature."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - only relevant without PyTorch
        return

    original_trace = getattr(pytorch_backend, "trace", None)

    def trace(a, offset=0, axis1=-2, axis2=-1, dtype=None, out=None):
        values = pytorch_backend.array(a)
        if values.ndim < 2:
            raise ValueError("diag requires an array of at least two dimensions")
        diagonal = pytorch_backend.diagonal(
            values,
            offset=_operator_index(offset),
            axis1=_operator_index(axis1),
            axis2=_operator_index(axis2),
        )
        result = pytorch_backend.sum(diagonal, axis=-1, dtype=dtype)
        if out is not None:
            copy_ = getattr(out, "copy_", None)
            if copy_ is not None:
                copy_(result)
                return out
            out[...] = result
            return out
        return result

    trace.__name__ = getattr(original_trace, "__name__", "trace")
    trace.__doc__ = getattr(original_trace, "__doc__", None)
    pytorch_backend.trace = trace

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.trace = trace


def _patch_jax_std_out_facade() -> None:
    """Make public and raw JAX ``std`` accept NumPy's ``out`` argument."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "jax":
        return

    try:
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - backend import fails first
        raw_jax = None

    original_std = backend.std
    original_raw_std = getattr(raw_jax, "std", None) if raw_jax is not None else None

    def _return_or_store_out(result, out):
        if out is None:
            return result
        return backend.asarray(out).at[...].set(result)

    def _std_kwargs(axis, dtype, ddof, keepdims, correction):
        kwargs = {
            "axis": axis,
            "dtype": dtype,
            "out": None,
            "keepdims": keepdims,
        }
        if correction is None:
            kwargs["ddof"] = ddof
        elif ddof == 0:
            kwargs["correction"] = correction
        else:
            kwargs["ddof"] = ddof
            kwargs["correction"] = correction
        return kwargs

    def std(
        a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, correction=None
    ):
        result = original_std(
            a,
            **_std_kwargs(axis, dtype, ddof, keepdims, correction),
        )
        return _return_or_store_out(result, out)

    std.__name__ = getattr(original_std, "__name__", "std")
    std.__doc__ = getattr(original_std, "__doc__", None)
    backend.std = std

    if original_raw_std is None:
        return

    def raw_std(
        a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, correction=None
    ):
        result = original_raw_std(
            a,
            **_std_kwargs(axis, dtype, ddof, keepdims, correction),
        )
        return _return_or_store_out(result, out)

    raw_std.__name__ = getattr(original_raw_std, "__name__", "std")
    raw_std.__doc__ = getattr(original_raw_std, "__doc__", None)
    raw_jax.std = raw_std


def _patch_jax_matmul_out_facade() -> None:
    """Make public and raw JAX ``matmul`` honor NumPy's ``out`` contract."""

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if getattr(backend, "__backend_name__", None) != "jax":
        return

    try:
        import pyrecest._backend.jax as jax_backend  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - backend import fails first in practice
        return

    original_matmul = jax_backend.matmul

    def matmul(x, y, out=None):
        result = original_matmul(x, y, out=None)
        if out is None:
            return result
        return backend.asarray(out).at[...].set(result)

    matmul.__name__ = getattr(original_matmul, "__name__", "matmul")
    matmul.__doc__ = getattr(original_matmul, "__doc__", None)
    jax_backend.matmul = matmul
    backend.matmul = matmul


_patch_pytorch_comparison_facade()
_patch_pytorch_clip_facade()
_patch_pytorch_tile_facade()
_patch_pytorch_stack_helpers_facade()
_patch_pytorch_linear_helpers_facade()
_patch_raw_pytorch_cumulative_facade()
_patch_raw_pytorch_trace_facade()
_patch_jax_std_out_facade()
_patch_jax_matmul_out_facade()

from pyrecest.backend_support import (  # noqa: E402,F401
    backend_support,
    format_backend_support_markdown,
    get_backend_support,
)
from pyrecest.backend_tools import (  # noqa: E402,F401
    assert_backend,
    get_backend_name,
    is_backend,
    warn_if_backend_env_changed,
)
from pyrecest.evidence import (  # noqa: E402,F401
    EvidenceComputationMode,
    resolve_evidence_computation_mode,
)
from pyrecest.exceptions import (  # noqa: E402,F401
    BackendNotSupportedError,
    BackendSupportError,
    DimensionMismatchError,
    NumericalStabilityError,
    OptionalDependencyError,
    PyRecEstError,
    ShapeError,
    ValidationError,
)
from pyrecest.stability import (  # noqa: E402,F401
    get_public_api_status,
    iter_public_api_status,
    stability,
)

try:
    __version__ = version("pyrecest")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = [
    "BackendNotSupportedError",
    "BackendSupportError",
    "DimensionMismatchError",
    "EvidenceComputationMode",
    "NumericalStabilityError",
    "OptionalDependencyError",
    "PyRecEstError",
    "ShapeError",
    "ValidationError",
    "__version__",
    "assert_backend",
    "backend_support",
    "copy",
    "format_backend_support_markdown",
    "get_backend_name",
    "get_backend_support",
    "get_public_api_status",
    "is_backend",
    "iter_public_api_status",
    "stability",
    "warn_if_backend_env_changed",
    "resolve_evidence_computation_mode",
]
