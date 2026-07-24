"""Run the seed-1111 gamma=0.9 control grid with a FIB upper bound."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.jpo_exps.control_exps.revealing_grid import grid_runner


if __name__ == "__main__":
    grid_runner.main()
