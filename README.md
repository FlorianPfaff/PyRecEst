# PyRecEst

[![Tests](https://github.com/FlorianPfaff/PyRecEst/actions/workflows/tests.yml/badge.svg)](https://github.com/FlorianPfaff/PyRecEst/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyrecest.svg)](https://pypi.org/project/pyrecest/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyrecest.svg)](https://pypi.org/project/pyrecest/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://florianpfaff.github.io/PyRecEst/)

Recursive Bayesian Estimation for Python.

PyRecEst is a Python library for recursive Bayesian estimation on Euclidean
spaces and manifolds. It uses a NumPy backend by default and can also run with
PyTorch or JAX backends.

## Features

PyRecEst provides tools for:

- distributions and densities on Euclidean spaces and manifolds;
- recursive Bayesian estimators, filters, and trackers;
- multi-target tracking (MTT) and extended object tracking (EOT);
- evaluation of filters and trackers; and
- sampling distributions and generating grids.

## Which API Should I Start With?

| Task | Start with |
|------|------------|
| Linear Euclidean Gaussian filtering | `KalmanFilter` and `examples/basic/kalman_filter.py` |
| Reusable transition and measurement models | `examples/basic/kalman_filter_with_models.py` |
| Nonlinear filtering | `examples/basic/ukf_with_models.py` or particle-filter examples |
| Multi-target tracking with clutter or missed detections | `examples/basic/multi_target_tracking.py` |
| Circular, spherical, or manifold-valued states | circular, hypertoroidal, and hyperspherical distribution modules |
| Backend-portable code | `pyrecest.backend` imports plus focused tests under each backend |

For more detail, see `docs/choosing-an-api.md` and `docs/backend-compatibility.md`.

## Command Line And Reproducible Scenarios

After installation, use the lightweight CLI to inspect an environment or run a
reproducible scenario:

```bash
pyrecest info
pyrecest backends
pyrecest run-scenario scenarios/linear_gaussian_cv_1d/config.toml
```

The same scenario runner is available from a source checkout via
`scripts/run_scenario.py`. Scenario definitions live in `scenarios/` and include
TOML configuration plus JSON golden outputs.

The `pyrecest.diagnostics` module defines common diagnostics containers for
innovation statistics, particle-filter health, and association decisions.

## Installation

PyRecEst requires Python 3.11 or newer and earlier than Python 3.15.

Install the package from PyPI:

```bash
python -m pip install pyrecest
```

Optional backend and domain-specific dependencies can be installed with extras:

```bash
python -m pip install "pyrecest[pytorch_support]"
python -m pip install "pyrecest[jax_support]"
python -m pip install "pyrecest[healpy_support]"
```

For development from a source checkout, use Poetry or the provided conda
environment:

```bash
poetry install --with dev --all-extras
# or
conda env create -f environment.yml
```

## Quickstart

The following example runs a one-dimensional constant-velocity Kalman filter.
It uses the backend abstraction exposed by `pyrecest.backend`, so the same code
can run on supported numerical backends.

```python
from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter


dt = 1.0
system_matrix = array([[1.0, dt], [0.0, 1.0]])
measurement_matrix = array([[1.0, 0.0]])
system_noise_cov = diag(array([0.05, 0.01]))
measurement_noise_cov = array([[0.25]])
measurements = [0.9, 2.0, 3.1, 3.9, 5.2]

kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))

for measurement in measurements:
    kalman_filter.predict_linear(system_matrix, system_noise_cov)
    kalman_filter.update_linear(
        array([measurement]), measurement_matrix, measurement_noise_cov
    )
    print(kalman_filter.get_point_estimate())
```

Run the complete script with:

```bash
python examples/basic/kalman_filter.py
```

## Documentation

The `docs/` directory contains the first project documentation pages:

- [Getting started](docs/getting-started.md) covers installation, development
  setup, backend selection, and running examples.
- [API overview](docs/api-overview.md) maps the main packages and points to the
  most common public entry points.
- [Backend compatibility](docs/backend-compatibility.md) explains the NumPy,
  PyTorch, and JAX support model and known limitations.
- [API reference](docs/reference/index.md) contains generated package reference
  pages built with MkDocs and mkdocstrings.
- [Task tutorials](docs/tutorials/index.md) show common distribution, filtering,
  tracking, and evaluation workflows.
- [Shapes and conventions](docs/conventions.md) documents common vector,
  matrix, measurement-set, batch, and manifold-coordinate shapes.
- [Examples](examples/README.md) lists the executable examples and what each
  one demonstrates.

Build the documentation site locally with:

```bash
poetry install --with docs --without dev
poetry run mkdocs build --strict
```

## Backends

PyRecEst imports `pyrecest.backend` dynamically. The default backend is NumPy.
Set `PYRECEST_BACKEND` before Python imports `pyrecest` to select another
backend:

```bash
PYRECEST_BACKEND=pytorch python examples/basic/kalman_filter.py
PYRECEST_BACKEND=jax python examples/basic/kalman_filter.py
```

Install the matching optional extra before using a non-default backend.

## Examples and tests

- `examples/basic/kalman_filter.py` contains a small executable Kalman filter
  example.
- `tests/` contains additional usage examples for distributions, filters,
  smoothers, evaluation, sampling, metrics, and tracking utilities.

To run the test suite from a development environment:

```bash
python -m pytest
```

## Citation

If you use **PyRecEst** in your research, please cite:

<table>
  <tr>
    <th>BibTeX</th>
    <th>BibLaTeX</th>
  </tr>
  <tr>
    <td>
      <pre><code class="language-bibtex">@misc{pfaff_pyrecest_2023,
  author       = {Florian Pfaff},
  title        = {PyRecEst: Recursive Bayesian Estimation for Python},
  year         = {2023},
  howpublished = {\url{https://github.com/FlorianPfaff/PyRecEst}},
  note         = {MIT License}
}</code></pre>
    </td>
    <td>
      <pre><code class="language-biblatex">@software{pfaff_pyrecest_2023_software,
  author    = {Florian Pfaff},
  title     = {PyRecEst: Recursive Bayesian Estimation for Python},
  year      = {2023},
  url       = {https://github.com/FlorianPfaff/PyRecEst},
  license   = {MIT},
  keywords  = {Bayesian filtering; manifolds; tracking; Python; NumPy; PyTorch; JAX}
}</code></pre>
    </td>
  </tr>
</table>

## Credits

- Florian Pfaff (<pfaff@ias.uni-stuttgart.de>)

PyRecEst borrows its structure from libDirectional and follows its code closely
for many classes. libDirectional, a project to which Florian Pfaff contributed
extensively, is [available on GitHub](https://github.com/libDirectional). The
backend implementations are based on those of
[geomstats](https://github.com/geomstats/geomstats).

## License
`PyRecEst` is licensed under the MIT License.
