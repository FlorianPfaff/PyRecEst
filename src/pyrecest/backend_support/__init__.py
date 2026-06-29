"""Public accessors for backend support metadata."""

from __future__ import annotations

from operator import index as _operator_index

from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    iter_api_backend_capabilities,
)


def _patch_pytorch_dot_numpy_contract() -> None:
    """Make PyTorch dot follow NumPy's contraction axes."""
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
            return torch.tensordot(a, b, dims=([-1], [0]))
        if a.ndim == 1:
            return torch.tensordot(a, b, dims=([0], [-2]))
        return torch.tensordot(a, b, dims=([-1], [-2]))

    dot.__name__ = getattr(original_dot, "__name__", "dot")
    dot.__doc__ = getattr(original_dot, "__doc__", None)
    dot._pyrecest_numpy_contract = True
    raw_pytorch.dot = dot
    if active_pytorch_backend:
        backend.dot = dot


def _patch_pytorch_outer_numpy_contract() -> None:
    """Make PyTorch outer flatten inputs like NumPy's outer."""
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
        return torch.outer(a.reshape(-1), b.reshape(-1))

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


_patch_pytorch_dot_numpy_contract()
_patch_pytorch_outer_numpy_contract()
_patch_pytorch_tile_numpy_contract()


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
