# JPO gamma=0.9 full grid, seed 1111, 700-second cap

This resumable experiment uses the EffCom-style MDP with seed 1111, `S=6`,
two receiver actions, density `0.5`, and reward decay `10`.

The 100 points are the Cartesian product of:

- `gamma = [0.9]`;
- 10 evenly spaced `beta` values from `0` through `2`;
- 10 evenly spaced `epsilon` values from `0.03` through `0.30`.

NativeSARSOP uses precision `0.01` and a nominal 700-second cap per grid
point. Each returned policy is analyzed for reachable JPO violations and
discounted transmission occupancy. The runner records both `sarsop_gap` and
`root_gap`; the plotting convergence criterion is the strict
`sarsop_gap < 0.1` requested for the comparison.

The implementation reuses the established large-grid runner and changes only
the experiment definition and output root. It is resumable from each completed
`runs/<point_id>/summary.json` file.

Run or resume with:

```bash
python3 -B experiments/jpo_exps/control_exps/revealing_grid/jpo_grid_s6_seed1111_g09_b10_e10_eps003_030_t700/run_experiment.py --workers 2
```

Use `--limit N` for a pilot or `--force` to recompute completed points.
