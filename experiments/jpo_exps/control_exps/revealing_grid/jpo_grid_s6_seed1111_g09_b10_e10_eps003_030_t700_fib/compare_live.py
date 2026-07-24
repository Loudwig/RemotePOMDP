"""Compare live FIB histories with the completed fully-observable grid."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "jpo_grid_s6_seed1111_g09_b10_e10_eps003_030_t700"
GAMMA = 0.9


def _history(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def _at_time(rows: list[dict[str, float]], elapsed: float) -> dict[str, float]:
    eligible = [row for row in rows if row["elapsed_seconds"] <= elapsed]
    return eligible[-1] if eligible else rows[0]


def main() -> None:
    header = (
        "point       t_fib    L_fib    U_fib  gap_fib  "
        "gap_fo@t  improvement"
    )
    print(header)
    histories = sorted((ROOT / "runs").glob("*/jpo/native_output/history.tsv"))
    if not histories:
        print("No live FIB history yet (FIB initialization may still be running).")
        return
    for path in histories:
        point_id = path.parents[2].name
        fib = _history(path)
        old = _history(BASELINE / "runs" / point_id / "jpo/native_output/history.tsv")
        if not fib or not old:
            continue
        current = fib[-1]
        reference = _at_time(old, current["elapsed_seconds"])
        fib_lower = current["root_lower"] / GAMMA
        fib_upper = current["root_upper"] / GAMMA
        fib_gap = current["root_gap"] / GAMMA
        old_gap = reference["root_gap"] / GAMMA
        print(
            f"{point_id:11s} {current['elapsed_seconds']:7.1f} "
            f"{fib_lower:8.4f} {fib_upper:8.4f} {fib_gap:8.4f} "
            f"{old_gap:9.4f} {old_gap-fib_gap:11.4f}"
        )


if __name__ == "__main__":
    main()
