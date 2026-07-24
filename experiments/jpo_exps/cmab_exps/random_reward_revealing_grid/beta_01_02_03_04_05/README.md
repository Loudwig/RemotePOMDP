# Random-reward CMAB: beta 0.01 through 0.05

This grid keeps the established random-reward physical MDP:

- 6 states and 2 classical actions;
- action-independent transitions, density `0.5`, seed `1111`;
- fully observable initial upper bound;
- `gamma = 0.9`;
- `beta = [0.01, 0.02, 0.03, 0.04, 0.05]`;
- `epsilon = 0.01, 0.02, ..., 0.10`.

The ten `beta = 0.01` points from the exploratory row were reused. The shared
analysis notebook is one level above this folder.

Run or resume the experiment with:

```bash
python3 -B run_experiment.py --workers 4
```

Inspect the expanded grid without launching a solver:

```bash
python3 -B run_experiment.py --dry-run
```
