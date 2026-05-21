# Backend-Portable Workflows

This tutorial shows the conventions for code that should run under the NumPy,
PyTorch, and JAX backends without changing the estimator logic.

## 1. Select the backend before importing PyRecEst

Set `PYRECEST_BACKEND` before Python imports `pyrecest`:

```bash
PYRECEST_BACKEND=numpy python my_filter.py
PYRECEST_BACKEND=pytorch python my_filter.py
PYRECEST_BACKEND=jax JAX_ENABLE_X64=True python my_filter.py
```

For JAX workflows that compare numerical values against NumPy or PyTorch, enable
64-bit mode when the test tolerance assumes double precision.

## 2. Import arrays from `pyrecest.backend`

Use the backend facade for arrays, matrices, and common numerical helpers:

```python
from pyrecest.backend import array, diag, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter


initial = GaussianDistribution(
    array([0.0, 1.0]),
    diag(array([1.0, 0.25])),
    check_validity=False,
)
kf = KalmanFilter(initial)

system_matrix = array([[1.0, 1.0], [0.0, 1.0]])
process_noise = diag(array([0.05, 0.01]))
measurement_matrix = array([[1.0, 0.0]])
measurement_noise = array([[0.25]])

kf.predict_linear(system_matrix, process_noise)
diagnostics = kf.update_linear(
    array([0.9]),
    measurement_matrix,
    measurement_noise,
    return_diagnostics=True,
)

print(kf.get_point_estimate())
print(diagnostics["nis"])
```

Avoid importing NumPy, PyTorch, or JAX directly inside reusable estimator code
unless the API is intentionally backend-specific.

## 3. Keep shapes explicit

Backend differences usually appear first as shape, dtype, or scalar-conversion
issues. Prefer explicit one-dimensional vectors and two-dimensional matrices:

| Quantity               | Recommended shape |
|------------------------|-------------------|
| State mean             | `(n,)`            |
| State covariance       | `(n, n)`          |
| Measurement vector     | `(m,)`            |
| Measurement matrix     | `(m, n)`          |
| Measurement covariance | `(m, m)`          |

For a one-dimensional measurement, use `array([z])` rather than a scalar and
`array([[r]])` rather than `array([r])`.

## 4. Test the same script under each target backend

Use the backend matrix as a contract, not a promise that every advanced helper is
portable. For a compact smoke test, run:

```bash
for backend in numpy pytorch jax; do
  PYRECEST_BACKEND="$backend" python my_filter.py
done
```

If the workflow depends on backend metadata, inspect it directly:

```bash
pyrecest backends --format markdown
python scripts/check_backend_api_matrix.py
```

## 5. Document intentional backend restrictions

When an API cannot preserve backend semantics, update
`src/pyrecest/_backend/capabilities.py`, the backend API matrix, and a focused
test in the same patch. If an operation copies through NumPy or SciPy, document
whether gradients, device placement, or JAX tracing are preserved.
