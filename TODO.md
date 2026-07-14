# TODO

- Support the zero-erasure endpoint `epsilon == 0`. The Rx belief recursion can
  then have a zero denominator: a no-reception information state may be
  unreachable under the current Rx policy but become reachable under a greedy
  candidate. Define and test how the fixed-belief improvement handles this
  change of reachability without inventing an invalid belief. Until this is
  resolved, configuration validation deliberately rejects `epsilon == 0`.

- Resolve the pessimistic overflow boundary layer. For now, non-revealing
  transmissions at age `delta_max - 1` are recorded separately as deferred
  boundary-layer diagnostics and are excluded from the main revealing-violation
  count. Experiments show that these entries track the truncation boundary as
  `delta_max` increases and can be visited with substantial undiscounted
  probability even though their discounted contribution becomes negligible.
