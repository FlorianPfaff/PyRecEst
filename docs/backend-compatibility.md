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
`pyrecest.backend.get_backend_name()`.

## Support Summary

The test workflow runs the suite for the `numpy`, `pytorch`, and `jax` backends
across supported Python versions. Passing tests do not mean that every public
API has identical behavior on every backend. Some modules intentionally assert
or raise when a backend does not support the required operation.

| Backend | Best fit                                                                                                                  | Main limitations                                                                                                                                                          |
|---------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NumPy   | Default backend and broadest compatibility for examples, evaluation, plotting, tracking, and SciPy-based helpers.         | No automatic differentiation through `pyrecest.backend.autodiff`. Autodiff calls raise `AutodiffNotImplementedError`.                                                     |
| PyTorch | Tensor workflows and automatic differentiation where the required API is implemented.                                     | Some backend helpers are placeholders or bridge through NumPy/SciPy. `pyrecest.backend.signal.fftconvolve` and `pyrecest.backend.searchsorted` are not implemented.       |
| JAX     | JAX array and autodiff workflows for APIs that avoid unsupported mutable, assignment-heavy, or SciPy-specific operations. | Several package areas explicitly skip or reject JAX, including some sampling, tracking, assignment, evaluation, point-set registration, and manifold-specific operations. |

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
