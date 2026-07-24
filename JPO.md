# Infinite-horizon JPO POMDP

We use Julia's `NativeSARSOP` to do the optimisation of infinite-horizon POMDP optimization and exports the resulting alpha-vector policy.

## Architecture

The workflow has three deliberately separate stages:

1. `jpo_model.py` wraps a `FiniteMDP` with the JPO action, observation, reward,
   and Bayesian belief model.
2. `jpo_sarsop.py` serializes that finite model and `julia/solve_jpo.jl` solves
   the unrestricted POMDP with NativeSARSOP. Successful messages always yield
   a Dirac posterior during this stage.
3. `jpo_policy.py` analyzes the extracted policy. If it finds a transmission
   for which reception and non-reception induce the same next receiver action,
   it constructs the lower bound controller only as post-processing
   and evaluates its value.

`jpo.py` orchestrates the stages, while `run_jpo.py` is the command-line entry
point. Nothing changes `remote_api.py` or the existing MDP implementation.

## JPO model

For `n` physical states and `A` receiver actions, there are `A * 2^n` JPO
actions. Prescriptions use the same deterministic ordering as EffCom:

```text
prescription_index = binary(g(0), ..., g(n-1)), with g(0) the most significant bit
jpo_action = prescription_index * n_receiver_actions + receiver_action
```

The physical transition for `(a_rx, g)` is exactly `FiniteMDP.P[a_rx]`. If
`b_bar = b @ P[a_rx]`, the implemented observation probabilities are

```text
Pr(M=s)   = (1-epsilon) * g(s) * b_bar(s)
Pr(M=chi) = sum_s b_bar(s) * (1-g(s)+epsilon*g(s)).
```

A successful observation has posterior `delta_s`. The null posterior is

```text
b_next(s) = b_bar(s) * (1-g(s)+epsilon*g(s)) / Pr(M=chi).
```

Impossible branches return `None`; they are never assigned an arbitrary
normalized belief.

### Reward timing

The implemented timing is

```text
S_t --a_rx--> S_(t+1) --g(S_(t+1)), channel--> M_(t+1) --> B_(t+1).
```

The stage reward at current physical state `z` is therefore

```text
r(z, (a_rx,g)) = sum_s P[a_rx,z,s]
                 * (R[a_rx,z,s] - gamma * beta * g(s)).
```

The factor `gamma` is explicit: the initial Dirac belief represents the state
already known at time zero, and the first charged transmission occurs after
the first physical transition, at time one. If the intended objective instead
charges that transmission in the same discounted stage as the transition
reward, this is the one convention to change: replace `gamma * beta` by
`beta`, together with its focused reward and evaluation-bound tests.

## Multiple initial Dirac beliefs

NativeSARSOP accepts one root belief, whereas the requested objective averages
all Dirac roots under one policy. The exported solver model therefore adds:

- one dummy state `d`;
- one forced initialization action;
- `T(s | d, init) = 1/n`, followed by observation `s`;
- zero initialization reward.

All other uses of the initialization action, and all JPO actions at `d`, have
a dominating negative reward. Thus

```text
V(delta_d) = gamma * (1/n) * sum_s V(delta_s).
```

The reported objective divides the synthetic root value by `gamma` and, more
directly, computes

```text
L0 = mean_s L(delta_s),  U0 = mean_s U(delta_s).
```

This is one alpha-vector policy shared by all roots; the code never solves one
independent POMDP per physical state.

## Bounds and evaluation without Monte Carlo

NativeSARSOP supplies the blind-policy lower bound, sawtooth upper bound,
SARSOP belief sampling, alpha backups, pruning, and the usual root-gap stopping
criterion. Each approximate blind-policy alpha is shifted by its one-sided
Bellman residual so that it is a certified subsolution for that feasible
policy. By default, the corner upper bound is the fully observable MDP value:
revealing the state to the controller can only improve its value, and this
initialization scales linearly rather than quadratically with the exponential
JPO action space. NativeSARSOP's FIB initialization remains available with
`--solver-initial-upper-bound fib`; it starts from
`max(reward)/(1-gamma)`, so every iterate remains an upper bound even if it
reaches its time limit.

Policy evaluation does not require Monte Carlo. The evaluator propagates the
joint mass of the physical state and the controller's internal belief for a
finite certified horizon, then brackets the remaining discounted tail with
global one-stage reward bounds. This also handles the restricted controller,
whose internal belief can intentionally differ from the physical posterior.
The requested tail interval is enforced; an insufficient horizon/node cap
raises an error instead of silently returning an uncertified number.

Monte Carlo simulation is optional and is recorded only as validation. The
existing finite-state API policy can likewise be evaluated without Monte Carlo
by its Bellman policy-evaluation routine.

## Installation and execution

Install Julia once, then instantiate the pinned environment:

```bash
brew install julia
julia --startup-file=no --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Run a small example:

```bash
python3 run_jpo.py \
  --n-states 4 --n-actions 2 --density 0.75 \
  --gamma 0.9 --beta 0.1 --epsilon 0.1 \
  --solver-max-time 300 --output jpo_run
```

The JPO action space is exponential in the number of physical states, so large
`n_states` values can become expensive independently of the solver.

Run all tests, including the NativeSARSOP integration test when Julia is
available:

```bash
python3 -m pytest -q
```

## Saved artifacts

Each run directory retains:

- `cli_arguments.json`: the complete command-line experiment parameters when
  using `run_jpo.py`;
- `model.npz`: physical `P`, `R`, JPO parameters, initial Dirac beliefs and
  weights;
- `native_input/`: exact arrays and metadata passed to Julia;
- `native_output/`: alpha vectors, action map, corner upper bounds, per-Dirac
  bounds, sampled belief points, per-belief metadata, training history and
  stopping reason;
- `policy.npz`: reloadable alpha-vector policy;
- `training.json`: model hash, action encoding, package versions,
  solver/bound parameters,
  initialization residuals, bounds and complete convergence history;
- `result.json`: reachable-policy analysis, violation records, deterministic
  policy-value interval, optional restricted lower bound and optional
  simulation diagnostics;
- `solver_stdout.txt` and `solver_stderr.txt`: Julia logs.
