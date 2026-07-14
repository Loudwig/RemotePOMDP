# API grid with retransmission tail

This experiment repeats the previous `S=10`, density `0.5` API parameter grid
on the same sampled EffCom-style control MDP (`seed=12`), replacing the
pessimistic overflow value with the fixed retransmit-until-success tail.

## Fixed configuration

- states: `10`
- actions: `2`
- density: `0.5`
- seed: `12`
- initial distribution: uniform
- `delta_train = 100`
- `delta_check = 90`
- `boundary_model = "tail"`
- `boundary_tx_mode = "force_transmit"`

The Cartesian grid is unchanged:

- `gamma`: `0.80, 0.90, 0.95`
- erasure probability `epsilon`: `0.05, 0.10, 0.20, 0.30`
- communication cost `beta`: `0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.00, 4.20`

There are 96 runs. Training is performed for every grid point, independently
of the theorem margin.

## Violation categories

- core: `delta <= 90`; these alone determine `is_revealing`;
- buffer: `90 < delta < 99`;
- boundary-adjacent: `delta = 99`;
- boundary transmissions: `delta = 100`.

Every final violation record includes discounted occupancy under the two-way
tail model. Results also report discounted table occupancy, tail occupancy, and
discounted flow entering the tail.

## Files

- `api_grid.ipynb`: executable experiment and plots;
- `results.json`: incrementally saved run records.

The legacy overflow experiment remains in `../api_grid_s10_d05_delta100/` and
can be reproduced in code with `boundary_model="legacy_overflow"`.

## Completed result

All 96 runs completed and passed the implemented restricted-NE tolerance.
There are no final core or buffer violations, including in the 36
positive-margin runs. Five runs retain boundary-adjacent violations at age 99,
with at most four violating states in a run. No lower-bound one-way value was
triggered because the final core violation count is zero everywhere.
