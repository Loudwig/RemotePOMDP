"""Tests for the reusable local and Slurm experiment runner."""

import json
from pathlib import Path

import pytest

from experiment_runner import (
    ExperimentSpecError,
    compress_indices,
    expand_points,
    load_manifest,
    load_spec,
    merge_results,
    plan_experiment,
    run_manifest_index,
    write_slurm_scripts,
)


def write_spec(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def tiny_spec(tmp_path: Path) -> dict:
    return {
        "name": "tiny_grid",
        "description": "Fast runner integration test",
        "output_dir": str(tmp_path / "output"),
        "result_detail": "compact",
        "base": {
            "n_states": 2,
            "n_actions": 1,
            "density": 0.5,
            "reward_decay": 1.0,
            "mdp_seed": 3,
            "init_seed": 7,
            "gamma": 0.5,
            "epsilon": 0.2,
            "delta_train": 2,
            "delta_check": 1,
            "vi_tol": 1e-8,
            "api_tol": 1e-8,
            "ne_tol": 1e-7,
            "max_vi_iterations": 5000,
            "max_rx_iterations": 20,
            "max_api_iterations": 20
        },
        "grid": {
            "beta": [0.0, 0.2],
            "tx_init": ["never", "always"]
        },
        "slurm": {
            "partition": "CPU",
            "time": "00:30:00",
            "cpus_per_task": 1,
            "mem": "2G",
            "max_concurrent": 2,
            "python_module": "python/3.11"
        }
    }


def test_grid_expansion_separates_mdp_and_initialization_seeds(tmp_path: Path) -> None:
    path = write_spec(
        tmp_path / "spec.json",
        {
            "name": "seed_grid",
            "base": {
                "n_states": 2,
                "n_actions": 1,
                "density": 0.5,
                "delta_train": 2,
                "delta_check": 1
            },
            "grid": {"mdp_seed": [1, 2], "init_seed": [10, 20, 30]}
        },
    )
    points = expand_points(load_spec(path))
    assert len(points) == 6
    assert len({point["run_id"] for point in points}) == 6
    assert {(p["parameters"]["mdp_seed"], p["parameters"]["init_seed"]) for p in points} == {
        (1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)
    }


def test_explicit_duplicate_points_are_rejected(tmp_path: Path) -> None:
    path = write_spec(
        tmp_path / "duplicates.json",
        {
            "name": "duplicates",
            "base": {
                "n_states": 2,
                "n_actions": 1,
                "density": 0.5,
                "delta_train": 2,
                "delta_check": 1
            },
            "points": [{"beta": 0.1}, {"beta": 0.1}]
        },
    )
    with pytest.raises(ExperimentSpecError, match="duplicate experiment point"):
        expand_points(load_spec(path))


def test_run_shards_merge_and_resume(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path / "tiny.json", tiny_spec(tmp_path))
    manifest_path = plan_experiment(spec_path)
    manifest = load_manifest(manifest_path)
    assert manifest["expected_runs"] == 4

    first = run_manifest_index(manifest_path, 0)
    assert first["status"] == "ok"
    assert first["mdp_seed"] == 3
    assert first["init_seed"] == 7
    assert first["initial_tx_hash"]
    assert first["gamma"] == 0.5
    assert first["epsilon"] == 0.2
    assert first["tx_init"] in {"never", "always"}
    assert first["rx_init"] == "fully_observed"
    assert "parameters" not in first
    assert "objective" not in first
    assert "performance_upper_bound" not in first
    assert first["performance"]["upper_bound"] is not None
    if first["core_violation_count"]:
        assert first["performance"]["kind"] == "upper_and_lower_bounds"
        assert first["performance"]["lower_bound"] is not None
    else:
        assert first["performance"]["kind"] == "upper_bound_only"
        assert first["performance"]["lower_bound"] is None

    # A successful shard is reused instead of being recomputed.
    assert run_manifest_index(manifest_path, 0)["finished_at"] == first["finished_at"]

    results_path = merge_results(manifest_path)
    partial = json.loads(results_path.read_text(encoding="utf-8"))
    assert partial["summary"] == {
        "expected": 4,
        "completed": 1,
        "ok": 1,
        "error": 0,
        "corrupt": 0,
        "missing": 3,
        "pending_indices": [1, 2, 3],
    }

    for index in range(1, 4):
        assert run_manifest_index(manifest_path, index)["status"] == "ok"
    results_path = merge_results(manifest_path)
    complete = json.loads(results_path.read_text(encoding="utf-8"))
    assert complete["summary"]["ok"] == 4
    assert complete["summary"]["missing"] == 0
    assert [record["index"] for record in complete["runs"]] == [0, 1, 2, 3]
    assert complete["performance_summary"]["upper_bound_only_runs"] >= 1
    assert complete["performance_summary"]["interval_runs"] >= 1
    for record in complete["runs"]:
        if record["core_violation_count"] > 0:
            assert record["performance"]["lower_bound"] is not None
            assert record["performance"]["gap"] == pytest.approx(
                record["performance"]["upper_bound"]
                - record["performance"]["lower_bound"]
            )
            assert all(
                {
                    "state",
                    "age",
                    "last_received",
                    "tx_action",
                    "success_action",
                    "no_reception_action",
                    "distance_to_boundary",
                    "discounted_occupancy",
                }
                <= violation.keys()
                for violation in record["core_violations"]
            )
        else:
            assert record["performance"]["lower_bound"] is None
            assert record["performance"]["gap"] is None

    readme = results_path.with_name("README.md").read_text(encoding="utf-8")
    assert "## Performance values" in readme
    assert "Full normalized specification" in readme
    assert "rsync -avz" in readme
    assert results_path.with_name("experiment.json").exists()


def test_slurm_scripts_use_cpu_array_tasks_and_one_merge_job(tmp_path: Path) -> None:
    spec_path = write_spec(tmp_path / "tiny.json", tiny_spec(tmp_path))
    manifest_path = plan_experiment(spec_path)
    array_script, merge_script = write_slurm_scripts(manifest_path)
    array_text = array_script.read_text(encoding="utf-8")
    merge_text = merge_script.read_text(encoding="utf-8")
    assert "#SBATCH --partition=CPU" in array_text
    assert "experiment_runner.py run-one" in array_text
    assert "OPENBLAS_NUM_THREADS" in array_text
    assert "experiment_runner.py merge" in merge_text
    assert compress_indices([0, 1, 2, 4, 6, 7]) == "0-2,4,6-7"
