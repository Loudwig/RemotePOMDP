# Remote-estimation CMAB revealing grids

These two completed experiments use the same six-state remote-estimation CMAB
(`S = A = 6`, density `0.5`, seed `1111`, `gamma = 0.9`) and vary the
transmission cost and channel erasure probability:

- `beta_00_05_10_15/`: the anchor grid with
  `beta = [0, 0.05, 0.10, 0.15]`;
- `beta_01_02_03_04/`: the fine-beta zoom with
  `beta = [0.01, 0.02, 0.03, 0.04]`.
- `beta_01_02_03_04_seed_1000/`: the same fine-beta zoom for MDP seed `1000`.

Both use `epsilon = 0.01, 0.02, ..., 0.10`, the fully observable SARSOP
initial upper bound, precision `0.01`, and a 500-second limit per point.

`gamma_0p9_fullgrid_zoom.ipynb` reads both `results.json` files and generates
the combined boundary/zoom figure, diagnostics, and the beta-zero
free-communication comparison. Revealing/non-revealing markers are plotted at
full opacity; solver precision is reported separately.

`gamma_0p9_zoom_seed_1000.ipynb` compares the seed-1000 zoom with the
seed-1111 zoom and writes figures under `plots/seed_1000/`.
