"""Run the beta=0,0.05,0.10,0.15 random-reward CMAB grid."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.jpo_exps.cmab_exps.random_reward_revealing_grid import (
    grid_runner,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT = {
    "name": "cmab_random_reward_beta_00_05_10_15",
    "grid_region": "anchor",
    "mdp_type": "cmab-random-reward",
    "n_states": 6,
    "n_actions": 2,
    "density": 0.5,
    "mdp_seed": 1111,
    "gammas": [0.9],
    "betas": [0.0, 0.05, 0.1, 0.15],
    "epsilons": np.linspace(0.01, 0.1, 10).tolist(),
    "expected_points": 40,
    "solver": dict(grid_runner.DEFAULT_SOLVER),
    "default_workers": 4,
}


if __name__ == "__main__":
    grid_runner.run_grid(ROOT, EXPERIMENT)
