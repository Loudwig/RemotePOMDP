"""Run the fine-beta gamma=0.9 remote-estimation JPO grid."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.jpo_exps.cmab_exps.remote_estimation_revealing_grid import (
    grid_runner as base_runner,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT = deepcopy(base_runner.EXPERIMENT)
EXPERIMENT.update(
    {
        "name": "remote_estimation_beta_01_02_03_04",
        "grid_region": "fine_beta_zoom",
        "betas": [0.01, 0.02, 0.03, 0.04],
        "expected_points": 40,
        "default_workers": 4,
    }
)

_BASE_RUN_POINT = base_runner.run_point


def _configure_base_runner() -> None:
    """Point the shared implementation at this experiment and output root."""

    base_runner.ROOT = ROOT
    base_runner.EXPERIMENT = EXPERIMENT


def run_point(point, force: bool = False):
    """Configure shared globals inside each spawned worker."""

    _configure_base_runner()
    return _BASE_RUN_POINT(point, force)


def main() -> None:
    _configure_base_runner()
    base_runner.run_point = run_point
    base_runner.main()


if __name__ == "__main__":
    main()
