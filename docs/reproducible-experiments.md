# Reproducible Experiments

For papers, reports, and benchmark claims, store reproducibility artifacts under
`reproducibility/`.

A strong artifact contains:

- the PyRecEst version or commit hash;
- a lock file or exported environment;
- scenario configuration and random seeds;
- expected metrics with tolerances;
- a single command that regenerates results;
- notes about backend-specific behavior.

Start from `reproducibility/templates/paper-artifact/` and keep generated files
small enough to inspect in a pull request.
