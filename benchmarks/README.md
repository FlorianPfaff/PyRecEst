# Benchmarks

Benchmarks should emit machine-readable JSON with both numerical outputs and
timing information. CI can archive the JSON to build a small performance history
without depending on a separate dashboard service.

Run the basic benchmark locally with:

```bash
python benchmarks/basic_regressions.py --output benchmark-results.json
```
