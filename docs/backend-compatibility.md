# Backend Compatibility

PyRecEst uses `pyrecest.backend` as a backend facade. User code that should run
on more than one backend should import array constructors and numerical helpers
from `pyrecest.backend` instead of importing NumPy, PyTorch, or JAX directly.

```python
from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter
```

Set `PYRECEST_BACKEND` before Python imports `pyrecest`:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
PYRECEST_BACKEND=jax python examples/basic/kalman_filter.py
```

After import, the selected backend is available as
`pyrecest.backend.__backend_name__` and through
`pyrecest.backend.get_backend_name()`. Convenience helpers in
`pyrecest.backend_tools` and at package level expose `get_backend_name()`,
`is_backend(...)`, `assert_backend(...)`, and
`warn_if_backend_env_changed()` for backend-sensitive scripts.

## Support Summary

The test workflow runs the suite for the `numpy`, `pytorch`, and `jax` backends
across supported Python versions. Passing tests do not mean that every public
API has identical behavior on every backend. Some modules intentionally assert
or raise when a backend does not support the required operation. The declared
unsupported and partial backend capabilities live in
`src/pyrecest/_backend/capabilities.py` and are exercised by backend contract
tests. See the [backend support matrix](backend-support.md) for a generated,
smoke-test-backed snapshot of selected API capabilities.

Association-hypothesis helpers are currently NumPy/SciPy-oriented utilities.
They may coerce backend arrays to NumPy arrays and should not be assumed to
preserve PyTorch device placement, PyTorch gradients, or JAX tracing semantics
unless a focused backend test covers the exact helper being used.

| Backend | Best fit                                                                                                                  | Main limitations                                                                                                                                                          |
|---------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NumPy   | Default backend and broadest compatibility for examples, evaluation, plotting, tracking, and SciPy-based helpers.         | No automatic differentiation through `pyrecest.backend.autodiff`. Autodiff calls raise `AutodiffNotImplementedError`.                                                     |
| PyTorch | Tensor workflows and automatic differentiation where the required API is implemented.                                     | Some backend helpers are placeholders or bridge through NumPy/SciPy. `pyrecest.backend.signal.fftconvolve` and `pyrecest.backend.searchsorted` are not implemented.       |
| JAX     | JAX array and autodiff workflows for APIs that avoid unsupported mutable, assignment-heavy, or SciPy-specific operations. | Several package areas explicitly skip or reject JAX, including some sampling, tracking, assignment, evaluation, point-set registration, and manifold-specific operations. |

## Capability Metadata

Backend limitations are tracked in `src/pyrecest/_backend/capabilities.py`. The
metadata distinguishes unsupported facade functions from partial support, such
as PyTorch helpers that bridge through NumPy/SciPy and may not preserve device
placement or gradient behavior. Public API support rows are documented in
[Backend API Matrix](backend-api-matrix.md).

## NumPy Backend

NumPy is the default backend and should be the first choice when:

- using the examples without backend-specific requirements;
- using plotting, evaluation, tracking, and SciPy-adjacent helpers;
- investigating whether a failure is backend-specific.

The NumPy backend does not provide automatic differentiation. The functions in
`pyrecest.backend.autodiff` are placeholders that raise
`AutodiffNotImplementedError`. Use PyTorch, JAX, or an autograd-based workflow
when differentiating through backend operations is required.

## PyTorch Backend

Install PyTorch support with:

```bash
python -m pip install "pyrecest[pytorch_support]"
```

Use the PyTorch backend when tensor operations or PyTorch autodiff are part of
the workflow. Keep these caveats in mind:

- `pyrecest.backend.signal.fftconvolve` is not implemented.
- `pyrecest.backend.searchsorted` is a placeholder that raises
  `NotImplementedError`.
- Some linear algebra helpers use NumPy or SciPy internally, including matrix
  square root, fractional matrix powers, polar decomposition, quadratic
  assignment, and some Sylvester-equation paths. These helpers may copy data
  between tensor and NumPy representations and should not be assumed to preserve
  PyTorch device placement or gradient behavior like native PyTorch operations.
- `pyrecest.backend.random.choice` does not support weighted sampling without
  replacement.

When a workflow uses advanced tracking, evaluation, plotting, or SciPy-heavy
utilities, compare behavior against the NumPy backend before assuming full
PyTorch parity.

## JAX Backend

Install JAX support with:

```bash
python -m pip install "pyrecest[jax_support]"
```

Use the JAX backend when JAX arrays and JAX autodiff are needed. JAX has the
largest set of explicit exclusions in the current codebase:

- Backend dtype helpers `convert_to_wider_dtype`, `get_default_dtype`, and
  `get_default_cdtype` are not implemented.
- Some autodiff facade functions are not supported, including Hessian-vector
  and combined value/Jacobian/Hessian helpers.
- `UKFOnManifolds.predict` and `UKFOnManifolds.update` reject the JAX backend.
- Several distribution operations reject JAX, including selected custom
  distributions, SE(2) operations, spherical harmonics operations, some complex
  hyperspherical distributions, piecewise-constant distribution operations, and
  some von Mises-Fisher sampling paths.
- Several samplers, assignment utilities, point-set registration utilities,
  multi-session assignment utilities, evaluation helpers, and result summaries
  skip or reject JAX.

The JAX random backend maintains a global PRNG key for facade compatibility and
also supports explicit state passing in some random functions. Code that needs
fully functional JAX-style random handling should pass and manage state
deliberately instead of relying only on implicit global state.

## Representation Conversion

The distribution representation-conversion gateway is tested under the same
backend matrix as the rest of the suite. The portable baseline currently covers:

- Euclidean analytic-to-particle conversion, for example
  `GaussianDistribution -> LinearDiracDistribution` through sampling;
- particle-to-Gaussian conversion through moment matching;
- class-based and alias-based routes such as `LinearDiracDistribution` and
  `"particles"`;
- backend-independent argument validation for missing or unsupported conversion
  parameters.

Conversion results are expected to keep arrays in the active backend
representation when the source and target representation are backend-portable.
For example, running with `PYRECEST_BACKEND=pytorch` should produce PyTorch
tensors for linear Dirac particles, while `PYRECEST_BACKEND=jax` should produce
JAX arrays.

Backend portability is route-specific. A conversion that delegates to a target
class's `from_distribution(...)` method inherits that target's backend support.
Routes based on SciPy-heavy grids, manifold operations, plotting, or
backend-specific samplers may still be NumPy-only or have explicit PyTorch/JAX
limitations. When adding a new conversion route, add a focused backend test if
the route is intended to be portable, or document and test the explicit
restriction if it is not.

## Choosing A Backend

Start with NumPy when learning the library, reproducing examples, or using
tracking and evaluation helpers. Move to PyTorch or JAX when the workflow needs
their tensor or autodiff behavior and the target API has been checked on that
backend.

For backend-portable code:

- import arrays and numerical helpers from `pyrecest.backend`;
- set `PYRECEST_BACKEND` before importing `pyrecest`;
- keep array shapes and dtypes explicit;
- run the same focused test or example with each backend you intend to support;
- expect advanced utilities to require backend-specific checks until API-level
  support tables are added.

## Documenting New Backend Restrictions

When adding or changing an API with backend-specific behavior:

- raise a clear `NotImplementedError` or assertion with the unsupported backend
  named in the message;
- add or update tests that exercise the supported backend behavior;
- mention the restriction in the relevant tutorial, example, or API notes;
- prefer implementing missing backend facade functions over direct imports when
  the operation should be portable.
