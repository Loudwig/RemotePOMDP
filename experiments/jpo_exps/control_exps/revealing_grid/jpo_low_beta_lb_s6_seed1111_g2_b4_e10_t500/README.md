# Low-beta JPO grid with restricted lower bounds

This experiment uses random EffCom-style MDP seed 1111, `S=6`, two receiver
actions, density `0.5`, and reward decay `10`.

The 80 points are the Cartesian product of:

- `gamma = [0.9, 0.99]`;
- `beta = [0, 0.05, 0.1, 0.15]`;
- ten evenly spaced `epsilon` values from `0.01` through `0.1`.

NativeSARSOP uses precision `0.01` and a nominal 500-second cap. Sampled
belief matrices are not exported. Every extracted policy is analyzed for all
reachable JPO violations and discounted transmission occupancy.

For every policy with at least one violation, the experiment constructs the
API-style restricted controller after training. A violation already implies a
positive-probability successful transmission. Its selected transformed action
is unchanged. On a non-revealing successful message—where the next receiver
action equals the null-observation receiver action—the controller retains the
null posterior instead of the Dirac posterior. Deterministic joint
state-belief propagation evaluates this feasible controller to interval width
at most `1e-3`; the lower endpoint is the reported restricted lower bound.
The same deterministic `1e-3` evaluation is applied to every normal extracted
policy, regardless of whether it violates.

Run or resume with:

```bash
python3 experiments/jpo_exps/control_exps/revealing_grid/jpo_low_beta_lb_s6_seed1111_g2_b4_e10_t500/run_experiment.py
```

Use `--workers 1` to reduce peak memory or `--limit N` for a pilot.
