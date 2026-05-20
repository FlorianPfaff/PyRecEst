# Task Tutorials

These tutorials focus on common PyRecEst tasks. Each page uses the public
package-level imports and backend-compatible arrays that are recommended for
user code.

## Tutorials

- [Use a distribution](use-a-distribution.md): create distributions, multiply
  Gaussian factors, and compare the result with the information-form update.
- [Write a filter loop](write-a-filter-loop.md): run a Kalman predict/update
  loop and inspect the posterior state.
- [Robust Kalman updates](robust-kalman-update.md): use NIS gating and
  heavy-tailed measurement updates for outlier-prone measurements.
- [Run a tracker](run-a-tracker.md): initialize a labeled multi-Bernoulli
  tracker and process cluttered measurements.
- [Evaluate a simulation](evaluate-a-simulation.md): run a built-in scenario,
  save evaluation output, and summarize filter performance.

## Related Material

- [Getting started](../getting-started.md) covers installation and backend
  selection.
- [API overview](../api-overview.md) maps the main packages.
- [Examples](../examples.md) lists the executable scripts in the
  repository.
