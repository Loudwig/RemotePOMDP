# API grid with always-transmit / random-Rx initialization

This experiment repeats the 96-point `S=10`, density `0.5` API grid from
`../api_grid_s10_d05_train100_check90_tail/` on the same sampled EffCom-style
control MDP. The only experimental change is the initial policy pair.

## Fixed configuration

- states: `10`
- actions: `2`
- density: `0.5`
- physical MDP seed: `12`
- initial distribution: uniform
- `delta_train = 100`
- `delta_check = 90`
- boundary model: fixed retransmit-until-success tail
- boundary transmission: forced at `delta_train`

The Cartesian grid is:

- `gamma`: `0.80, 0.90, 0.95`
- erasure probability `epsilon`: `0.05, 0.10, 0.20, 0.30`
- communication cost `beta`: `0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.00, 4.20`

There are 96 runs. Training is performed for every point independently of the
theorem margin.

## Initial policy pair

- Tx initially transmits in every table state:
  `pi_tx(s, delta, u) = 1`.
- Rx is a random deterministic table generated once with seed `1012`. The same
  table is used at every grid point.

The Rx table is sampled once and then fixed; this is not a stochastic policy.
API still begins by replacing the initial Tx table with its exact best response
to this initial Rx table.

## Violation categories

- core: reachable ages through `delta_check=90`; these determine the main
  revealing diagnostic;
- buffer: ages `91` through `98`;
- boundary-adjacent: age `99`;
- boundary transmissions: age `100`.

## Files

- `api_grid.ipynb`: executable experiment, checks, and plots;
- `results.json`: incrementally saved compact run records.

## Completed result

All 96 runs completed, converged, and satisfy the implemented restricted-NE
tolerance. Seven of the 60 negative-margin points have final reachable core
violations; none of the 36 positive-margin points does.

| beta | gamma | epsilon | margin | core violations |
|---:|---:|---:|---:|---:|
| 0.10 | 0.80 | 0.20 | -0.5400 | 1 |
| 0.05 | 0.80 | 0.30 | -0.7900 | 19 |
| 0.05 | 0.90 | 0.30 | -1.8400 | 13 |
| 0.10 | 0.90 | 0.30 | -1.7900 | 31 |
| 0.20 | 0.95 | 0.05 | -0.7025 | 1 |
| 0.05 | 0.95 | 0.20 | -2.9900 | 4 |
| 0.05 | 0.95 | 0.30 | -3.9400 | 209 |

The largest case has violations from age zero and total discounted violation
occupancy `0.28284`; it is not a boundary artifact. Some small-count cases are
practically negligible: the first row is a single age-88 reachable state with
numerically zero occupancy, while the fifth and sixth rows have very small
occupancy.

Lower-bound one-way values were computed for all seven violating runs. Relative
to the baseline initialization, the final objective is higher at 53 grid points
and lower at 43, confirming that restricted API is initialization-dependent.
The Rx regret is still relative to the implemented restricted improvement
routine, not exhaustive global policy enumeration.
