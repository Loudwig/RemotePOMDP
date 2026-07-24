# Remote-estimation CMAB: beta 0, 0.05, 0.10, 0.15

This experiment studies the action-independent CMAB estimation MDP with
`S = A = 6`, density `0.5`, and MDP seed `1111`.

This is the 40-point anchor grid at gamma 0.9:

- `gamma = [0.9]`;
- `beta = [0, 0.05, 0.10, 0.15]`;
- ten evenly spaced `epsilon` values from `0.01` through `0.10`.

The CMAB sufficient-condition boundary shown in the analysis is

```text
beta > gamma * epsilon * (1 - epsilon) / (1 - gamma^2).
```

NativeSARSOP uses its fully observable initial upper bound, precision `0.01`,
and a 500-second cap per point. Four points run concurrently by default. The
runner is resumable and records every point beneath `runs/`. The shared
analysis notebook is one level above this folder.

Run or resume with:

```bash
python3 -B run_experiment.py --workers 4
```

Inspect the grid without launching SARSOP with:

```bash
python3 -B run_experiment.py --dry-run
```
