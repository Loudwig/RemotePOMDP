# Random-reward CMAB revealing grids

These two completed experiments use the same fully observable six-state,
two-action random-reward CMAB (density `0.5`, seed `1111`, `gamma = 0.9`) and
vary the transmission cost and channel error:

- `beta_00_05_10_15/`: the anchor grid with
  `beta = [0, 0.05, 0.10, 0.15]`;
- `beta_01_02_03_04_05/`: the fine grid with
  `beta = [0.01, 0.02, 0.03, 0.04, 0.05]`.

Both use `epsilon = 0.01, 0.02, ..., 0.10`, the fully observable SARSOP
initial upper bound, precision `0.01`, and a 500-second limit per point.

`gamma_0p9_fullgrid_zoom.ipynb` reads both `results.json` files and produces:

1. revealing/non-revealing plots for each grid with the CMAB bound;
2. dedicated solver-precision plots;
3. discounted transmission occupancy plots;
4. a combined anchor-grid and fine-grid zoom view.

Revealing/non-revealing points are always plotted at full opacity. Solver
precision is shown only in its dedicated diagnostic plot.
