"""Run the six-state random-reward CMAB fine-beta grid."""

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
    "name": "cmab_random_reward_beta_01_02_03_04_05",
    "grid_region": "fine_beta",
    "mdp_type": "cmab-random-reward",
    "n_states": 6,
    "n_actions": 2,
    "density": 0.5,
    "mdp_seed": 1111,
    "gammas": [0.9],
    "betas": [0.01, 0.02, 0.03, 0.04, 0.05],
    "epsilons": np.linspace(0.01, 0.1, 10).tolist(),
    "expected_points": 50,
    "solver": dict(grid_runner.DEFAULT_SOLVER),
    "default_workers": 4,
}


if __name__ == "__main__":
    grid_runner.run_grid(ROOT, EXPERIMENT)
