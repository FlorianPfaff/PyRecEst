# Command Line Interface

PyRecEst provides a small CLI for environment inspection and reproducible
scenario runs.

```bash
pyrecest info
pyrecest backends
pyrecest run-scenario scenarios/linear_gaussian_cv_1d/config.toml
```

`pyrecest info` prints JSON with the installed PyRecEst version, active backend,
Python version, platform, and selected dependency versions. This is useful in bug
reports.

`pyrecest backends` prints the machine-readable backend capability metadata.

`pyrecest run-scenario` runs a TOML scenario and can compare the final estimate
against an `expected.json` file:

```bash
pyrecest run-scenario scenarios/linear_gaussian_cv_1d/config.toml \
  --expected scenarios/linear_gaussian_cv_1d/expected.json \
  --tolerance 1e-8
```
