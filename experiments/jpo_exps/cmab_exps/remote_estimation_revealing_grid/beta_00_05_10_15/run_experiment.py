"""Run the beta=0,0.05,0.10,0.15 remote-estimation CMAB grid."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.jpo_exps.cmab_exps.remote_estimation_revealing_grid import (
    grid_runner,
)


if __name__ == "__main__":
    grid_runner.main()
