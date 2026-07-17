"""Run the seed-1111 gamma=0.9 JPO grid on beta x epsilon."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.jpo_grid_s6_seed933_g2_b10_e10 import (
    run_experiment as base_runner,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT = deepcopy(base_runner.EXPERIMENT)
EXPERIMENT.update(
    {
        "name": "jpo_grid_s6_seed1111_g09_b10_e10_eps003_030_t700",
        "mdp_seed": 1111,
        "gammas": [0.9],
        "betas": np.linspace(0.0, 2.0, 10).tolist(),
        "epsilons": np.linspace(0.03, 0.30, 10).tolist(),
        "expected_points": 100,
        "default_workers": 2,
    }
)
EXPERIMENT["solver"]["max_time"] = 700.0
EXPERIMENT["solver"]["max_belief_nodes"] = 2_000_000

_BASE_RUN_POINT = base_runner.run_point


def _configure_base_runner() -> None:
    """Point the shared implementation at this experiment and output root."""
    base_runner.ROOT = ROOT
    base_runner.EXPERIMENT = EXPERIMENT


def run_point(point, force: bool = False):
    """Configure globals inside each spawned worker before running one point."""
    _configure_base_runner()
    return _BASE_RUN_POINT(point, force)


def main() -> None:
    _configure_base_runner()
    base_runner.run_point = run_point
    base_runner.main()


if __name__ == "__main__":
    main()
