# API grid experiment: S=10, density=0.5, Delta_max=100

This experiment studies revealing violations after alternating person-by-person
optimization on one fixed EffCom-style control MDP. The physical MDP is held
fixed across the parameter grid so that observed changes are caused by
`beta`, `gamma`, and the erasure probability `epsilon`.

## Fixed setup

- states: `S = 10`
- actions: `A = 2`
- EffCom transition density: `0.5`
- MDP and policy-initialization seed: `12`
- initial state distribution: uniform
- maximum represented age: `Delta_max = 100`
- Tx initialization: never transmit
- Rx initialization: fully observed MDP policy copied across ages

The Cartesian grid contains 96 runs:

- `gamma = [0.80, 0.90, 0.95]`
- `epsilon = [0.05, 0.10, 0.20, 0.30]`
- `beta = [0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.00, 4.20]`

The theorem diagnostic is

```text
m = beta - gamma * epsilon * (1 - epsilon) / (1 - gamma).
```

Training is performed on both sides of `m = 0`; the margin never restricts
which grid points are optimized.

## Files

- `api_grid.ipynb`: executable experiment, numerical checks, and Matplotlib
  plots, plus a rotatable Plotly.js 3D parameter-grid plot with the theoretical
  `m = 0` surface.
- `results.json`: resumable machine-readable results and API training histories.

The notebook uses the exact Bellman evaluations and restricted Rx improvement
implemented in `remote_api.py`. The final diagnostic compares the configured
API stopping tolerance with `|m|` and marks runs with revealing violations.
Exact lower-bound one-way values are retained for grid points with at least one
final revealing violation; other points store `null` for that field.
