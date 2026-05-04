"""Shape-validation helpers for reusable estimation-model objects.

The functions in this module follow the public shape conventions documented in
``docs/conventions.md``.  They intentionally avoid importing filters or
concrete distributions so they can be used by model classes, filters, and tests
without creating dependency cycles.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

from pyrecest import backend

# pylint: disable=too-many-arguments


def _as_backend_array(value: Any, name: str):
    """Convert ``value`` to a backend array and raise a user-facing error."""
    try:
        return backend.asarray(value)
    except Exception as exc:  # pragma: no cover - backend-specific exception type
        raise ValueError(f"{name} must be convertible to a backend array.") from exc


def _shape_tuple(value: Any) -> tuple[int, ...]:
    """Return the backend shape as a plain Python tuple."""
    return tuple(int(dim) for dim in backend.shape(value))


def _format_shape(shape: tuple[int, ...]) -> str:
    return str(shape)


def _validate_expected_dim(
    actual_dim: int, expected_dim: int | None, name: str, dim_name: str
) -> None:
    if expected_dim is None:
        return
    if not isinstance(expected_dim, Integral):
        raise TypeError(f"{dim_name} must be an integer or None.")
    if int(expected_dim) <= 0:
        raise ValueError(f"{dim_name} must be positive.")
    if actual_dim != int(expected_dim):
        raise ValueError(
            f"{name} has dimension {actual_dim}, expected {int(expected_dim)}."
        )


def _to_python_bool(value: Any) -> bool:
    """Convert scalar backend boolean results to Python ``bool``."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def validate_vector(
    vector: Any,
    *,
    name: str = "vector",
    dim: int | None = None,
    allow_scalar: bool = False,
):
    """Validate and return a one-dimensional backend array.

    Parameters
    ----------
    vector : array-like
        Candidate vector.
    name : str, optional
        Name used in validation error messages.
    dim : int, optional
        Required vector dimension.
    allow_scalar : bool, optional
        If true, scalar input is reshaped to shape ``(1,)``.

    Returns
    -------
    backend array
        The validated vector as a backend array.
    """
    vector = _as_backend_array(vector, name)
    vector_ndim = backend.ndim(vector)

    if vector_ndim == 0 and allow_scalar:
        vector = backend.reshape(vector, (1,))
        vector_ndim = 1

    if vector_ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional with shape (dim,), got shape {_format_shape(_shape_tuple(vector))}."
        )

    actual_dim = _shape_tuple(vector)[0]
    _validate_expected_dim(actual_dim, dim, name, "dim")
    return vector


def validate_state_vector(
    state: Any,
    *,
    name: str = "state",
    state_dim: int | None = None,
    allow_scalar: bool = False,
):
    """Validate a state vector with shape ``(state_dim,)``.

    Scalar state vectors are rejected by default.  Set ``allow_scalar=True`` for
    one-dimensional scalar states that should be reshaped to ``(1,)``.
    """
    return validate_vector(state, name=name, dim=state_dim, allow_scalar=allow_scalar)


def validate_measurement_vector(
    measurement: Any,
    *,
    name: str = "measurement",
    meas_dim: int | None = None,
    allow_scalar: bool = False,
):
    """Validate a single-target measurement vector with shape ``(meas_dim,)``."""
    return validate_vector(
        measurement, name=name, dim=meas_dim, allow_scalar=allow_scalar
    )


def validate_matrix(
    matrix: Any,
    *,
    name: str = "matrix",
    rows: int | None = None,
    cols: int | None = None,
):
    """Validate and return a two-dimensional backend array.

    Parameters
    ----------
    matrix : array-like
        Candidate matrix.
    name : str, optional
        Name used in validation error messages.
    rows : int, optional
        Required number of rows.
    cols : int, optional
        Required number of columns.
    """
    matrix = _as_backend_array(matrix, name)

    if backend.ndim(matrix) != 2:
        raise ValueError(
            f"{name} must be two-dimensional, got shape {_format_shape(_shape_tuple(matrix))}."
        )

    actual_rows, actual_cols = _shape_tuple(matrix)
    _validate_expected_dim(actual_rows, rows, name, "rows")
    _validate_expected_dim(actual_cols, cols, name, "cols")
    return matrix


def validate_covariance_matrix(
    covariance: Any,
    *,
    name: str = "covariance",
    dim: int | None = None,
    allow_scalar: bool = False,
    check_symmetric: bool = False,
    symmetric_rtol: float = 1e-7,
    symmetric_atol: float = 1e-9,
):
    """Validate a square covariance matrix with shape ``(dim, dim)``.

    This function checks shape and, optionally, symmetry.  It deliberately does
    not check positive definiteness because that can be expensive and backend
    dependent.  Concrete distributions may still perform stronger validity
    checks when needed.
    """
    covariance = _as_backend_array(covariance, name)

    if backend.ndim(covariance) == 0 and allow_scalar:
        covariance = backend.reshape(covariance, (1, 1))

    covariance = validate_matrix(covariance, name=name)
    rows, cols = _shape_tuple(covariance)
    if rows != cols:
        raise ValueError(
            f"{name} must be square, got shape {_format_shape((rows, cols))}."
        )

    _validate_expected_dim(rows, dim, name, "dim")

    if check_symmetric and not _to_python_bool(
        backend.allclose(
            covariance,
            backend.transpose(covariance),
            rtol=symmetric_rtol,
            atol=symmetric_atol,
        )
    ):
        raise ValueError(f"{name} must be symmetric.")

    return covariance


def validate_transition_matrix(
    system_matrix: Any,
    *,
    name: str = "system_matrix",
    state_dim: int | None = None,
    pred_dim: int | None = None,
):
    """Validate a linear transition matrix with shape ``(pred_dim, state_dim)``."""
    return validate_matrix(system_matrix, name=name, rows=pred_dim, cols=state_dim)


def validate_measurement_matrix(
    measurement_matrix: Any,
    *,
    name: str = "measurement_matrix",
    state_dim: int | None = None,
    meas_dim: int | None = None,
):
    """Validate a linear measurement matrix with shape ``(meas_dim, state_dim)``."""
    return validate_matrix(measurement_matrix, name=name, rows=meas_dim, cols=state_dim)


def validate_noise_covariance(
    noise_covariance: Any,
    *,
    name: str = "noise_covariance",
    dim: int | None = None,
    allow_scalar: bool = False,
    check_symmetric: bool = False,
):
    """Validate a process- or measurement-noise covariance with shape ``(dim, dim)``."""
    return validate_covariance_matrix(
        noise_covariance,
        name=name,
        dim=dim,
        allow_scalar=allow_scalar,
        check_symmetric=check_symmetric,
    )


def _maybe_call(value: Any, *, allow_methods: bool) -> Any:
    if callable(value):
        if not allow_methods:
            raise ValueError(
                "Callable distribution attributes are disabled for this inference attempt."
            )
        return value()
    return value


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, Integral) and int(value) > 0:
        return int(value)
    return None


def infer_state_dim_from_distribution(
    distribution: Any, *, allow_methods: bool = True
) -> int:
    """Infer a state dimension from common PyRecEst distribution attributes.

    The helper first checks explicit dimension attributes such as ``dim`` and
    ``input_dim``.  If those are unavailable, it falls back to common storage
    attributes such as ``mu``/``m`` for means, ``C`` for covariance, and ``d``
    for row-oriented Dirac support locations.  Method calls such as
    ``mean()``/``covariance()`` are allowed by default but can be disabled with
    ``allow_methods=False`` to avoid expensive numerical fallbacks.
    """
    for attr_name in ("dim", "input_dim"):
        if hasattr(distribution, attr_name):
            attr_value = _maybe_call(
                getattr(distribution, attr_name), allow_methods=allow_methods
            )
            inferred_dim = _positive_int_or_none(attr_value)
            if inferred_dim is not None:
                return inferred_dim

    for attr_name in ("mu", "m", "mean"):
        if hasattr(distribution, attr_name):
            try:
                mean = _maybe_call(
                    getattr(distribution, attr_name), allow_methods=allow_methods
                )
                return _shape_tuple(
                    validate_state_vector(mean, name=f"distribution.{attr_name}")
                )[0]
            except (TypeError, ValueError):
                pass

    for attr_name in ("C", "covariance"):
        if hasattr(distribution, attr_name):
            try:
                covariance = _maybe_call(
                    getattr(distribution, attr_name), allow_methods=allow_methods
                )
                return _shape_tuple(
                    validate_covariance_matrix(
                        covariance, name=f"distribution.{attr_name}"
                    )
                )[0]
            except (TypeError, ValueError):
                pass

    if hasattr(distribution, "d"):
        try:
            support = validate_matrix(getattr(distribution, "d"), name="distribution.d")
            return _shape_tuple(support)[1]
        except (TypeError, ValueError):
            pass

    raise ValueError(
        "Could not infer a positive state dimension from the distribution."
    )


__all__ = [
    "infer_state_dim_from_distribution",
    "validate_covariance_matrix",
    "validate_matrix",
    "validate_measurement_matrix",
    "validate_measurement_vector",
    "validate_noise_covariance",
    "validate_state_vector",
    "validate_transition_matrix",
    "validate_vector",
]
