# Diagnostics

Filters, trackers, and samplers should expose diagnostics through explicit
return paths rather than printing or relying on private attributes. The module
`pyrecest.diagnostics` defines lightweight dataclasses for common diagnostic
families:

- `FilterDiagnostics` for innovation, NIS, NEES, log likelihood, and covariance
  trace;
- `ParticleDiagnostics` for effective sample size, resampling decisions, and
  weight entropy;
- `AssociationDiagnostics` for cost matrices, gated measurements, assignments,
  births, and deaths.

A recommended pattern is:

```python
state, diagnostics = filter.update(measurement, return_diagnostics=True)
```

APIs that do not yet support diagnostics should avoid ad-hoc return tuples and
instead add a documented diagnostics object when the feature is introduced.
