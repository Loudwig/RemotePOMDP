# Running experiment grids on the LTCI cluster

`experiment_runner.py` turns a JSON experiment specification into a Slurm job
array. Each array task runs one parameter point and writes one atomic JSON
shard. A dependent merge job creates a single `results.json` after all array
tasks finish.

This layout is intentional: multiple compute nodes must not concurrently edit
the same JSON file on shared NFS storage. Independent shards make interrupted
runs resumable and prevent lost updates.

## Quick start

Copy and edit `experiment_specs/example_grid.json`, giving every experiment a
new `name`. On the cluster, from the repository root:

```bash
module load python/3.11

# Validate, count the points, and inspect the generated sbatch command.
python experiment_runner.py submit experiment_specs/example_grid.json --dry-run

# Submit the array and its dependent merge job.
python experiment_runner.py submit experiment_specs/example_grid.json
```

For a first end-to-end check, use the dedicated four-point smoke test. It is
small enough to finish quickly and uses two concurrent Slurm array tasks:

```bash
# Prepare and validate everything without submitting a job.
python experiment_runner.py submit experiment_specs/smoke_test.json --dry-run

# Submit only after the dry-run output looks correct.
python experiment_runner.py submit experiment_specs/smoke_test.json
```

The command prints the array and merge job IDs. Monitor them with:

```bash
squeue -u "$USER"
```

By default, outputs go to:

```text
experiment_runs/<experiment-name>/
├── README.md              # generated experiment description and summary
├── experiment.json        # normalized specification snapshot
├── manifest.json
├── submission.json
├── run_array.sbatch
├── merge_results.sbatch
├── logs/
├── runs/                 # one JSON shard per point
└── results.json          # single merged file
```

Check progress or rebuild a partial/final merged file at any time:

```bash
python experiment_runner.py status experiment_runs/<experiment-name>/manifest.json
python experiment_runner.py merge experiment_runs/<experiment-name>/manifest.json
```

To retry missing or failed points, run the same submit command again. Already
successful shards are skipped and only unfinished indices are submitted.

## Specification format

`base` contains values shared by all runs. `grid` contains lists whose
Cartesian product defines the experiment. The runner supports independent:

- `mdp_seed`: physical MDP generation;
- `init_seed`: random policy initialization;
- `tx_init`: `never`, `random`, `always`, or `state_change`;
- `rx_init`: `fully_observed` or `random`;
- solver parameters such as `gamma`, `beta`, `epsilon`, `delta_train`, and
  tolerances.

For a non-Cartesian design, replace `grid` with an explicit `points` list:

```json
{
  "name": "explicit_points",
  "base": {
    "n_states": 10,
    "n_actions": 2,
    "density": 0.5,
    "delta_train": 100,
    "delta_check": 90
  },
  "points": [
    {"gamma": 0.8, "beta": 0.1, "epsilon": 0.05, "mdp_seed": 12},
    {"gamma": 0.95, "beta": 1.2, "epsilon": 0.3, "mdp_seed": 13}
  ]
}
```

Use `"result_detail": "compact"` for plotting-oriented records with metrics,
histories, violations, and policy hashes. Use `"full"` to additionally retain
all diagnostics and complete policy tables; this can make `results.json` much
larger.

Every successful record includes one `performance` object:

- `performance.upper_bound` is always present;
- when there are no core revealing violations, the record has
  `performance.kind = "upper_bound_only"` and the performance lower bound is
  `null`;
- when core violations exist, the solver automatically computes
  `performance.lower_bound`, and `performance.gap` is the difference between
  the bounds.

The merge job also creates `README.md`, recording the complete specification,
Slurm resources, completion counts, performance-bound summary, output layout,
and the exact `rsync` command for retrieving the whole experiment directory.

All run parameters are retained once as top-level fields. This includes
`gamma`, `beta`, `epsilon`, `mdp_seed`, `init_seed`, `tx_init`, and `rx_init`,
making them directly available as plotting/grouping columns after retrieval.
Use `pandas.json_normalize(payload["runs"])` to flatten the single nested
`performance` object into columns such as `performance.upper_bound`.

Final violations are retained individually in `core_violations`,
`buffer_violations`, and `boundary_adjacent_violations`. Each item includes
`state`, `age`, `last_received`, the relevant Tx/Rx actions,
`distance_to_boundary`, and `discounted_occupancy`. Age histograms should be
derived from these records rather than stored as a duplicate list.

The Slurm defaults target the `CPU` partition because the current tabular
solver does not use CUDA. `max_concurrent` limits how many points the scheduler
may run simultaneously. Slurm remains free to place multiple points on the
same compute node when resources permit.

## Retrieve results

Use the dedicated `ids-store` data node rather than the login node for file
transfers. From your local machine (on the Télécom Paris network or VPN):

```bash
rsync -avz \
  <tp-username>@ids-store.enst.fr:/absolute/path/to/RemotePOMDP/experiment_runs/<experiment-name>/results.json \
  ./results.json
```

If you want live partial results, run `merge` while the array is still active,
then retrieve the same `results.json`; its `summary` and `pending_indices`
fields show what remains.
