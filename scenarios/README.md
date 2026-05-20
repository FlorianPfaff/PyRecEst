# Scenario Zoo

The scenario zoo collects small, reproducible estimation problems with fixed
seeds, measurements, expected outputs, and optional plotting or benchmark
recipes. Scenarios are intentionally data-first so they can be used by
examples, regression tests, tutorials, and paper artifacts.

Each scenario should include:

- `README.md` with the mathematical model and intended use;
- `config.toml` with model, measurement, initial-state, and data sections;
- `expected.json` with tolerance-friendly golden outputs;
- optional scripts or notebooks for visualization.

Run a scenario from a development checkout with:

```bash
python scripts/run_scenario.py scenarios/linear_gaussian_cv_1d/config.toml \
  --expected scenarios/linear_gaussian_cv_1d/expected.json
```

Or, after installing the package with console scripts enabled:

```bash
pyrecest run-scenario scenarios/linear_gaussian_cv_1d/config.toml \
  --expected scenarios/linear_gaussian_cv_1d/expected.json
```
