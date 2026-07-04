"""Public accessors for backend support metadata."""

from __future__ import annotations

from operator import index as _operator_index

from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    iter_api_backend_capabilities,
)
from pyrecest.backend_support._torch_dtype_promotion_contract import (
    patch_pytorch_dtype_promotion_contract as _patch_pytorch_dtype_promotion_contract,
)


def _pytorch_scalar_tensor_index(index, torch_module):
    """Return Python int indices for scalar integer tensors."""

    if not torch_module.is_tensor(index) or index.ndim != 0:
        return index
    if (
        index.dtype in {torch_module.bool, torch_module.uint8}
        or index.dtype.is_floating_point
        or index.dtype.is_complex
    ):
        return index
    return _operator_index(index)


def _wrap_pytorch_assignment_helper(original_assignment, torch_module):
    """Normalize scalar tensor indices before assignment helper len() checks."""

    if getattr(original_assignment, "_pyrecest_scalar_tensor_index_contract", False):
        return original_assignment

    def assignment(x, values, indices, axis=0):
        indices = _pytorch_scalar_tensor_index(indices, torch_module)
        return original_assignment(x, values, indices, axis=axis)

    assignment.__name__ = getattr(original_assignment, "__name__", "assignment")
    assignment.__doc__ = getattr(original_assignment, "__doc__", None)
    assignment._pyrecest_scalar_tensor_index_contract = True
    return assignment


def _patch_raw_pytorch_assignment_scalar_tensor_indices() -> None:
    """Make raw PyTorch assignment helpers accept scalar integer tensor indices."""

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as _torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    raw_pytorch.assignment = _wrap_pytorch_assignment_helper(
        raw_pytorch.assignment,
        _torch,
    )
    raw_pytorch.assignment_by_sum = _wrap_pytorch_assignment_helper(
        raw_pytorch.assignment_by_sum,
        _torch,
    )
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.assignment = raw_pytorch.assignment
        backend.assignment_by_sum = raw_pytorch.assignment_by_sum


def _patch_pytorch_dot_numpy_contract() -> None:
    """Make PyTorch dot follow the backend batched inner-product contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return
    original_dot = raw_pytorch.dot
    if getattr(original_dot, "_pyrecest_numpy_contract", False):
        return

    def dot(a, b):
        a = raw_pytorch.array(a)
        b = raw_pytorch.array(b)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)

        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        if a.ndim == 1 and b.ndim == 1:
            return torch.dot(a, b)
        if b.ndim == 1:
            return torch.einsum("...i,i->...", a, b)
        if a.ndim == 1:
            return torch.einsum("i,...i->...", a, b)
        return torch.einsum("...i,...i->...", a, b)

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_contract = True
    raw_pytorch.dot = dot
    if active_pytorch_backend:
        backend.dot = dot


def _patch_pytorch_outer_numpy_contract() -> None:
    """Make PyTorch outer pair leading dimensions like the backend contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return
    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return
    original_outer = raw_pytorch.outer
    if getattr(original_outer, "_pyrecest_numpy_contract", False):
        return

    def outer(a, b):
        a = raw_pytorch.array(a)
        b = raw_pytorch.array(b)
        dtype = torch.promote_types(a.dtype, b.dtype)
        a = a.to(dtype=dtype)
        b = b.to(dtype=dtype)
        if a.ndim == 0 or b.ndim == 0:
            return torch.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    outer.__name__ = getattr(original_outer, "__name__", "outer")
    outer.__doc__ = getattr(original_outer, "__doc__", None)
    outer._pyrecest_numpy_contract = True
    raw_pytorch.outer = outer
    if active_pytorch_backend:
        backend.outer = outer


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


def _patch_pytorch_tile_numpy_contract() -> None:
    """Make PyTorch tile follow NumPy repetition semantics."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return
    original_tile = raw_pytorch.tile
    if getattr(original_tile, "_pyrecest_numpy_contract", False):
        return

    def tile(x, reps):
        x = raw_pytorch.array(x)
        repetitions = _pytorch_tile_repetitions(reps, np, torch)
        if not repetitions:
            return x.clone()
        if x.ndim < len(repetitions):
            x = x.reshape((1,) * (len(repetitions) - x.ndim) + tuple(x.shape))
        elif x.ndim > len(repetitions):
            repetitions = (1,) * (x.ndim - len(repetitions)) + repetitions
        return x.repeat(repetitions)

    tile.__name__ = getattr(original_tile, "__name__", "tile")
    tile.__doc__ = getattr(np.tile, "__doc__", None)
    tile._pyrecest_numpy_contract = True
    raw_pytorch.tile = tile
    if active_pytorch_backend:
        backend.tile = tile


def _patch_pytorch_copy_numpy_contract() -> None:
    """Make PyTorch copy return tensors for array-like inputs."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_copy = raw_pytorch.copy
    if getattr(original_copy, "_pyrecest_numpy_contract", False):
        return

    def copy(x):
        if raw_pytorch.is_array(x):
            return original_copy(x)
        return raw_pytorch.array(x)

    copy.__name__ = getattr(original_copy, "__name__", "copy")
    copy.__doc__ = getattr(original_copy, "__doc__", None)
    copy._pyrecest_numpy_contract = True
    raw_pytorch.copy = copy
    if active_pytorch_backend:
        backend.copy = copy


def _patch_pytorch_clip_numpy_contract() -> None:
    """Make PyTorch clip accept array-like inputs regardless of public backend."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_clip = raw_pytorch.clip
    if getattr(original_clip, "_pyrecest_numpy_contract", False):
        return

    def _clip_bound(value, *, device):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(device=device)
        return torch.as_tensor(value, device=device)

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

        x = raw_pytorch.array(a)
        result = torch.clip(
            x,
            min=_clip_bound(a_min, device=x.device),
            max=_clip_bound(a_max, device=x.device),
        )
        if out is not None:
            copy_ = getattr(out, "copy_", None)
            if copy_ is not None:
                copy_(result)
                return out
            out[...] = raw_pytorch.to_numpy(result)
            return out
        return result

    clip.__name__ = getattr(original_clip, "__name__", "clip")
    clip.__doc__ = getattr(original_clip, "__doc__", None)
    clip._pyrecest_numpy_contract = True
    raw_pytorch.clip = clip
    if active_pytorch_backend:
        backend.clip = clip


def _pytorch_broadcast_dimension(dimension, numpy_module) -> int:
    """Return one NumPy-style broadcast dimension as a non-boolean integer."""

    if isinstance(dimension, (bool, numpy_module.bool_)):
        raise TypeError("broadcast shape entries must be integers")
    try:
        return _operator_index(dimension)
    except TypeError as exc:
        raise TypeError("broadcast shape entries must be integers") from exc


def _pytorch_broadcast_shape(shape, numpy_module, torch_module) -> tuple[int, ...]:
    """Normalize NumPy-style broadcast shapes for ``torch.Tensor.expand``."""

    if torch_module.is_tensor(shape):
        shape = shape.detach().cpu().numpy()
    shape_array = numpy_module.asarray(shape)
    if shape_array.shape == ():
        broadcast_shape = (
            _pytorch_broadcast_dimension(shape_array.item(), numpy_module),
        )
    else:
        broadcast_shape = tuple(
            _pytorch_broadcast_dimension(one_dimension, numpy_module)
            for one_dimension in shape_array.tolist()
        )
    if any(one_dimension < 0 for one_dimension in broadcast_shape):
        raise ValueError("all elements of broadcast shape must be non-negative")
    return broadcast_shape


def _patch_pytorch_broadcast_to_numpy_contract() -> None:
    """Make PyTorch broadcast_to normalize shapes like NumPy."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except (
        ModuleNotFoundError
    ):  # pragma: no cover - PyTorch backend import failed earlier
        return

    original_broadcast_to = raw_pytorch.broadcast_to
    if getattr(original_broadcast_to, "_pyrecest_numpy_contract", False):
        return

    def broadcast_to(x, shape):
        x = raw_pytorch.array(x)
        result = x.expand(_pytorch_broadcast_shape(shape, np, torch))
        return result

    broadcast_to.__name__ = getattr(original_broadcast_to, "__name__", "broadcast_to")
    broadcast_to.__doc__ = getattr(original_broadcast_to, "__doc__", None)
    broadcast_to._pyrecest_numpy_contract = True
    raw_pytorch.broadcast_to = broadcast_to
    if active_pytorch_backend:
        backend.broadcast_to = broadcast_to


def _patch_jax_outer_numpy_contract() -> None:
    """Make JAX outer pair leading dimensions like the backend contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend import failed earlier
        return

    original_outer = raw_jax.outer
    if getattr(original_outer, "_pyrecest_numpy_contract", False):
        return

    def outer(a, b):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        if a.ndim == 0 or b.ndim == 0:
            return jnp.multiply(a, b)
        return a[..., :, None] * b[..., None, :]

    outer.__name__ = getattr(original_outer, "__name__", "outer")
    outer.__doc__ = getattr(jnp.outer, "__doc__", None)
    outer._pyrecest_numpy_contract = True
    raw_jax.outer = outer
    if active_jax_backend:
        backend.outer = outer


def _patch_jax_one_hot_backend_contract() -> None:
    """Make JAX one_hot accept the shared labels/num_classes contract."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"

    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend import failed earlier
        return

    original_one_hot = raw_jax.one_hot
    if getattr(original_one_hot, "_pyrecest_backend_contract", False):
        if active_jax_backend:
            backend.one_hot = original_one_hot
        return

    def one_hot(labels, num_classes=None, *, depth=None):
        if depth is not None:
            if num_classes is not None and num_classes != depth:
                raise TypeError("one_hot() got both 'num_classes' and 'depth'")
            num_classes = depth
        if num_classes is None:
            raise TypeError("one_hot() missing required argument 'num_classes'")
        return jnp.eye(num_classes, dtype=jnp.uint8)[jnp.asarray(labels)]

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_backend_contract = True
    raw_jax.one_hot = one_hot
    if active_jax_backend:
        backend.one_hot = one_hot


def _patch_jax_take_out_contract() -> None:
    """Make JAX take honor NumPy's out keyword without passing it to JAX."""
    try:
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - import fails before this module
        return

    active_jax_backend = getattr(backend, "__backend_name__", None) == "jax"

    try:
        import pyrecest._backend.jax as raw_jax  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend import failed earlier
        return

    original_take = raw_jax.take
    if getattr(original_take, "_pyrecest_out_contract", False):
        if active_jax_backend:
            backend.take = original_take
        return

    def take(
        a,
        indices,
        axis=None,
        out=None,
        mode=None,
        unique_indices=False,
        indices_are_sorted=False,
        fill_value=None,
    ):
        result = original_take(
            a,
            indices,
            axis=axis,
            out=None,
            mode=mode,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            fill_value=fill_value,
        )
        if out is None:
            return result
        return raw_jax.asarray(out).at[...].set(result)

    take.__name__ = getattr(original_take, "__name__", "take")
    take.__doc__ = getattr(original_take, "__doc__", None)
    take._pyrecest_out_contract = True
    raw_jax.take = take
    if active_jax_backend:
        backend.take = take


_patch_raw_pytorch_assignment_scalar_tensor_indices()
_patch_pytorch_dtype_promotion_contract()
_patch_pytorch_dot_numpy_contract()
_patch_pytorch_outer_numpy_contract()
_patch_pytorch_tile_numpy_contract()
_patch_pytorch_copy_numpy_contract()
_patch_pytorch_clip_numpy_contract()
_patch_pytorch_broadcast_to_numpy_contract()
_patch_jax_outer_numpy_contract()
_patch_jax_one_hot_backend_contract()
_patch_jax_take_out_contract()


def get_backend_support(
    api_name: str, *, backend: str | None = None
) -> dict[str, str] | str | None:
    """Return backend support metadata for a public API."""
    row = API_BACKEND_CAPABILITIES.get(api_name)
    if row is None:
        return None
    if backend is not None:
        return row.get(backend)
    return dict(row)


def backend_support(
    api_name: str, backend: str | None = None
) -> dict[str, str] | str | None:
    """Alias for :func:`get_backend_support` for concise user code."""
    return get_backend_support(api_name, backend=backend)


def _markdown_table_cell(value: object) -> str:
    escape = chr(92) + chr(124)
    return str(value).replace("\r", " ").replace("\n", "<br>").replace(chr(124), escape)


def _markdown_table_row(cells: list[str]) -> str:
    separator = chr(124)
    return f"{separator} " + f" {separator} ".join(cells) + f" {separator}"


def format_backend_support_markdown() -> str:
    """Render the public backend API matrix as a Markdown table."""
    lines = [
        _markdown_table_row(["API", "NumPy", "PyTorch", "JAX", "Notes"]),
        _markdown_table_row(["-----", "-------", "---------", "-----", "-------"]),
    ]
    for api_name, row in iter_api_backend_capabilities():
        lines.append(
            _markdown_table_row(
                [
                    f"`{_markdown_table_cell(api_name)}`",
                    _markdown_table_cell(row["numpy"]),
                    _markdown_table_cell(row["pytorch"]),
                    _markdown_table_cell(row["jax"]),
                    _markdown_table_cell(row.get("notes", "")),
                ]
            )
        )
    return "\n".join(lines)


__all__ = [
    "BACKEND_SUPPORT_LEVELS",
    "backend_support",
    "format_backend_support_markdown",
    "get_backend_support",
]
