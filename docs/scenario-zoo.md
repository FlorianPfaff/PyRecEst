# Scenario Zoo

The scenario zoo is a collection of small, reproducible estimation problems that
can be reused as examples, regression tests, tutorials, benchmarks, and paper
artifacts.

A scenario should contain:

| File                       | Purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `README.md`                | Model summary and reproduction notes.                            |
| `config.toml`              | Model, measurement, initial-state, seed, and data configuration. |
| `expected.json`            | Golden outputs with tolerances.                                  |
| optional scripts/notebooks | Plots or scenario-specific analysis.                             |

The first scenario is `scenarios/linear_gaussian_cv_1d`, a constant-velocity
Kalman filtering problem. Run it with:

```bash
python scripts/run_scenario.py scenarios/linear_gaussian_cv_1d/config.toml   --expected scenarios/linear_gaussian_cv_1d/expected.json
```

Installed environments can use the console script:

```bash
pyrecest run-scenario scenarios/linear_gaussian_cv_1d/config.toml   --expected scenarios/linear_gaussian_cv_1d/expected.json
```

## Adding A Scenario

Prefer deterministic scenarios where possible. For stochastic scenarios, record
the seed, generator semantics, number of Monte Carlo runs, and tolerance bands.
Use backend-portable operations if the scenario is intended to run under NumPy,
PyTorch, and JAX.
