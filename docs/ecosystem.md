# Positioning In The Python Ecosystem

PyRecEst focuses on recursive Bayesian estimation, directional and manifold
statistics, multi-target tracking, extended object tracking, and backend-portable
numerical workflows.

It is not intended to replace general-purpose tensor libraries, probabilistic
programming systems, or plotting packages. Instead, it provides estimation-domain
objects and algorithms that can sit on top of NumPy, PyTorch, or JAX where the
backend contract supports the required operations.

A good rule of thumb:

- use NumPy for broadest compatibility and SciPy-heavy workflows;
- use PyTorch when tensor workflows or autograd integration are central;
- use JAX when pure functional, vectorized, or JIT-friendly workflows are the
  target and the selected API is marked as supported.
