# Random-reward CMAB: beta 0, 0.05, 0.10, 0.15

This experiment applies the initial 40-point remote-estimation grid to a
classical random-reward CMAB:

- 6 physical states and 2 classical actions;
- action-independent circular transition kernels;
- density `0.5`;
- one reproducible uniform reward in `[0, 1]` for every state-action pair;
- MDP and reward seed `1111`;
- `gamma = 0.9`;
- `beta = [0, 0.05, 0.10, 0.15]`;
- `epsilon = 0.01, 0.02, ..., 0.10`.

NativeSARSOP uses the fully observable initial upper bound, precision `0.01`,
and a 500-second limit per point. Four points run concurrently. Each point is
checkpointed below `runs/`, and `results.json` is updated after every
completion. The shared analysis notebook is one level above this folder.

Run or resume the experiment with:

```bash
python3 -B run_experiment.py --workers 4
```

Inspect the expanded grid without launching a solver:

```bash
python3 -B run_experiment.py --dry-run
```
