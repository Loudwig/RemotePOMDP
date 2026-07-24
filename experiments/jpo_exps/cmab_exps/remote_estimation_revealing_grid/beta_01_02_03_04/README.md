# Remote-estimation CMAB: beta 0.01 through 0.04

This is a 40-point refinement of the CMAB remote-estimation zoom:

- `gamma = 0.9`;
- `beta = [0.01, 0.02, 0.03, 0.04]`;
- `epsilon = 0.01, 0.02, ..., 0.10`;
- `S = A = 6`, density `0.5`, and MDP seed `1111`;
- fully observable initial upper bound;
- SARSOP precision `0.01` and 500-second cap per point;
- four workers by default.

The CMAB boundary is

```text
beta > gamma * epsilon * (1 - epsilon) / (1 - gamma^2).
```

Run or resume with:

```bash
python3 -B run_experiment.py --workers 4
```

The runner never overwrites completed points unless `--force` is supplied.
The shared analysis notebook is one level above this folder.
