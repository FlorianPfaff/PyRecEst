# Performance And Benchmarking

Benchmarks should report both numerical outputs and timing information. The
script `benchmarks/basic_regressions.py` emits JSON that can be archived by CI and
compared across releases.

Backend-specific targets should be explicit:

| Backend | Performance target                                                        |
|---------|---------------------------------------------------------------------------|
| NumPy   | Reliable default behavior and SciPy-heavy workflows.                      |
| PyTorch | Tensor/autodiff workflows and GPU-capable native paths where implemented. |
| JAX     | Pure functional and vectorized workflows where JIT is practical.          |

Avoid optimizing a backend-specific path until its dtype, device, and autodiff
semantics are documented in the capability matrix.
