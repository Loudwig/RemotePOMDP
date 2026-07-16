# Running experiment grids on the LTCI cluster

`experiment_runner.py` turns a JSON experiment specification into a Slurm job
array. Each array task runs one or more independent parameter points and every
point writes one atomic JSON shard. A dependent merge job creates a single
`results.json` after all array tasks finish.

This layout is intentional: multiple compute nodes must not concurrently edit
the same JSON file on shared NFS storage. Independent shards make interrupted
runs resumable and prevent lost updates.

## Quick start

Copy and edit `experiment_specs/example_grid.json`, giving every experiment a
new `name`. On the cluster, from the repository root:

```bash
# Validate, count the points, and inspect the generated sbatch command.
/usr/bin/python3 experiment_runner.py submit experiment_specs/example_grid.json --dry-run

# Submit the array and its dependent merge job.
/usr/bin/python3 experiment_runner.py submit experiment_specs/example_grid.json
```

For a first end-to-end check, use the dedicated four-point smoke test. It is
small enough to finish quickly and uses two concurrent Slurm array tasks:

```bash
# Prepare and validate everything without submitting a job.
/usr/bin/python3 experiment_runner.py submit experiment_specs/smoke_test.json --dry-run

# Submit only after the dry-run output looks correct.
/usr/bin/python3 experiment_runner.py submit experiment_specs/smoke_test.json
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
├── submissions/          # immutable task-index mappings for array chunks
├── logs/
├── runs/                 # one JSON shard per point
└── results.json          # single merged file
```

Check progress or rebuild a partial/final merged file at any time:

```bash
/usr/bin/python3 experiment_runner.py status experiment_runs/<experiment-name>/manifest.json
/usr/bin/python3 experiment_runner.py merge experiment_runs/<experiment-name>/manifest.json
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

When one parameter range depends on another parameter, use `blocks`. The
runner expands each block independently and combines their points into one
manifest. For example, this gives each gamma its own beta range:

```json
{
  "blocks": [
    {
      "base": {"gamma": 0.9},
      "grid": {"beta": [0.0, 1.0, 2.0], "mdp_seed": [10, 11]}
    },
    {
      "base": {"gamma": 0.99},
      "grid": {"beta": [0.0, 11.0, 22.0], "mdp_seed": [10, 11]}
    }
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
solver does not use CUDA. `max_concurrent` limits how many Slurm array tasks
the scheduler may run simultaneously. `points_per_task` controls how many
independent experiment points each task launches concurrently. Reserve at
least the same number of cores with `cpus_per_task`. For example,
`max_concurrent: 4`, `cpus_per_task: 8`, and `points_per_task: 8` use at most
four Slurm jobs but can train 32 experiment points at once. Each point still
writes its own shard, so merging and resuming work exactly as before. Slurm
remains free to place those tasks on the same or different compute nodes.

The runner uses the exact Python executable used for submission. On the LTCI
cluster, invoke it with `/usr/bin/python3`; `/usr/bin/python` is not present on
all compute nodes. The system Python already provides NumPy. A software module
can still be requested with `slurm.python_module`, but the default is `null`.

Because the cluster has `MaxArraySize=1001`, `array_chunk_size` defaults to
1000 Slurm tasks. Larger experiments are transparently split into chained
arrays with local task IDs `0-999`. Each task reads a generated mapping file to
find its group of global manifest indices. Chunks run sequentially so
`max_concurrent` remains a global Slurm-job cap, and one merge job runs after
the final chunk.

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
