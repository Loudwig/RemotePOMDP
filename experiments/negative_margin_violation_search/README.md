# Negative-margin revealing-violation search

This experiment searches for final, reachable, non-revealing transmissions in
the region where the theorem margin is strongly negative. It varies the
EffCom-style transition density, the number of receiver actions, and the API
initialization while holding the physical MDP seed and channel parameters
fixed.

## Configuration

- states: `10`
- actions: `2, 4`
- densities: `0.5, 0.9`
- physical MDP seed: `12`
- discount: `gamma = 0.9`
- erasure probability: `epsilon = 0.1`
- theorem threshold: `gamma * epsilon * (1-epsilon) / (1-gamma) = 0.81`
- margins: `-0.80, -0.70, -0.60`
- communication costs: `beta = 0.01, 0.11, 0.21`
- initial state distribution: uniform
- `delta_train = 100`
- `delta_check = 90`
- boundary model: fixed retransmit-until-success tail
- boundary transmission: forced at `delta_train`

The main result counts only reachable core violations with age at most
`delta_check`. Buffer, boundary-adjacent, and boundary states are retained as
separate diagnostics.

## Initializations

1. `never_fully_observed`: Tx initially never transmits (apart from the forced
   boundary); Rx uses the fully observed physical-MDP policy evaluated at the
   last received state `u`.
2. `always_random_rx`: Tx initially transmits everywhere; Rx is a reproducible
   random deterministic policy.
3. `state_change_random_rx`: Tx initially transmits when `s != u`; Rx is a
   reproducible random deterministic policy generated with a different seed.

Because API computes an exact Tx best response before its first Rx update, the
initial Tx table mainly affects stable tie-breaking. The different Rx
initializations are included to create genuinely different first Tx best
responses and optimization paths.

## Files

- `negative_margin_violation_search.ipynb`: executable grid, checks, and plots;
- `results.json`: incrementally saved compact run records.

## Completed result

All 36 runs completed, converged, and satisfy the implemented restricted-NE
tolerance. Five runs have final reachable core violations:

- density `0.5`, two actions, margin `-0.80`: both random-Rx
  initializations (`1292` and `1194` violations);
- density `0.5`, two actions, margin `-0.70`: the state-change/random-Rx
  initialization (`336` violations);
- density `0.5`, four actions, margin `-0.80`: both random-Rx
  initializations (`373` and `751` violations).

No final core violation occurs for density `0.9`, margin `-0.60`, or the
never-transmit/fully-observed baseline initialization. The violating runs
include ages zero and one with substantial discounted occupancy, so these are
not boundary-only artifacts. Lower-bound one-way values were computed for all
five violating runs.

Every one of the 12 structural/channel groups ended at more than one final Tx
and Rx policy across the three initializations. This demonstrates strong API
initialization dependence. The reported Rx regret remains relative to the
implemented restricted fixed-belief improvement routine; it is not a global Rx
best-response certificate.
