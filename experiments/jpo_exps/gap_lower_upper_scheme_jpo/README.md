# Gap lower/upper — scheme 1

This experiment measures the loss induced by restricting the JPO policy:

\[
\Delta W(\epsilon)
= V_{\mathrm{unrestricted}}(\epsilon)
- V_{\mathrm{restricted}}(\epsilon).
\]

## Experiment parameters

| Parameter | Value |
|---|---:|
| Experiment name | `gap_lower_upper_scheme_jpo` |
| Number of MDP states | `6` |
| Number of receiver actions | `2` |
| MDP density | `0.5` |
| Reward decay | `10.0` |
| Discount factor `gamma` | `0.9` |
| Weight `beta` | `0.05` |
| Epsilon grid | `0.01, 0.02, ..., 0.10` |
| Number of epsilon values | `10` |
| Target number of accepted MDP seeds | `30` |
| MDP seed-generator seed | `20260720` |
| Maximum number of candidate MDP seeds | `200` |
| Solver-gap rejection threshold | `0.5` |
| Default number of workers | `4` |

The experiment is resumable. A candidate MDP seed is accepted only if all ten
epsilon runs succeed and its largest achieved SARSOP gap is at most `0.5`.
Rejected seeds remain recorded in `results.json` and are replaced by subsequent
deterministic candidates.

## NativeSARSOP parameters

| Parameter | Value |
|---|---:|
| Search epsilon | `0.01` |
| Target precision | `0.01` |
| Maximum time per run | `600 s` |
| Maximum steps | `1,000,000` |
| Kappa | `0.5` |
| Delta | `0.0001` |
| Prune threshold | `0.1` |
| Initial-bound residual | `1e-8` |
| Initial-bound maximum time | `30 s` |
| Initial upper bound | `fully_observable` |
| Export beliefs | `false` |

Policy analysis uses a discounted-tail tolerance of `1e-8` and at most
`2,000,000` belief nodes. Fixed-policy evaluation uses an interval-width
tolerance of `1e-3` and at most `10,000,000` belief nodes.

When the unrestricted policy has no reachable restriction violation, the
restricted controller is identical and `delta W` is recorded as exactly zero.
Otherwise, both fixed policies are evaluated deterministically. The midpoint
difference is reported as `delta W`; rigorous lower and upper differences are
retained in `policy_gap_by_run.csv`.

## Run or extend the experiment

From the repository root:

```bash
python3 experiments/gap_lower_upper_scheme_jpo/run_experiment.py
```

Useful options:

- `--workers 1` reduces peak memory usage.
- `--limit N` runs at most `N` new epsilon points.
- `--target-valid-seeds N` extends the experiment to `N` accepted seeds. For
  example, use `--target-valid-seeds 40` to add ten accepted seeds.

The existing completed points are reused automatically.

## Analysis and figures

Execute `analysis.ipynb` to analyze every accepted seed currently stored
in `results.json`. The notebook is not capped at 30 seeds: future accepted
seeds are included automatically when it is rerun.

It writes:

- `delta_w_statistics.csv`, containing the mean, standard deviation, median,
  extrema, positive count/fraction, and conditional positive-gap statistics;
- `policy_gap_by_run.csv`, containing `delta W`, its rigorous lower and upper
  bounds, and both policy values for every run;
- `solver_gap_by_run.csv`, `solver_gap_summary.csv`, and
  `solver_gap_summary.json`, reporting the achieved solver precision;
- `plots/delta_w_mean_trajectories.{png,pdf}`;
- `plots/positive_delta_w_share.{png,pdf}`.

The first paper figure is title-free, uses `epsilon` and `delta W` as its axes,
and has the English legend entries `Individual trajectories` and `Mean`. The
second title-free figure reports the percentage of seeds with `delta W > 0`.
No bootstrap confidence interval is used in these two figures.
