# Remote-estimation CMAB zoom: seed 1000

Goal is to check that we didn't just get unlucky with our precedding seeds so i am rerunning it with a different seed.

This 40-point experiment repeats the fine-beta remote-estimation grid with a
new physical MDP seed:

- `gamma = 0.9`;
- `beta = [0.01, 0.02, 0.03, 0.04]`;
- `epsilon = 0.01, 0.02, ..., 0.10`;
- `S = A = 6`, density `0.5`, and MDP seed `1000`;
- fully observable initial upper bound;
- SARSOP precision `0.01` and a 500-second cap per point;
- four workers by default.

Run or resume from the repository root with:

```bash
python3 -B experiments/jpo_exps/cmab_exps/remote_estimation_revealing_grid/beta_01_02_03_04_seed_1000/run_experiment.py --workers 4
```

Completed points are reused unless `--force` is supplied. Use the parent-level
`gamma_0p9_zoom_seed_1000.ipynb` notebook to compare this grid with seed 1111.
