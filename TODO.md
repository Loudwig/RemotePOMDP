# TODO

- Support the zero-erasure endpoint `epsilon == 0`. The Rx belief recursion can
  then have a zero denominator: a no-reception information state may be
  unreachable under the current Rx policy but become reachable under a greedy
  candidate. Define and test how the fixed-belief improvement handles this
  change of reachability without inventing an invalid belief. Until this is
  resolved, configuration validation deliberately rejects `epsilon == 0`.

- Run a truncation study for the retransmit-until-success tail model across
  several `(delta_train, delta_check)` pairs. Compare core, buffer, and
  boundary-adjacent violations and their discounted occupancies. The old
  pessimistic boundary remains available only through
  `boundary_model="legacy_overflow"` for reproducibility.
