"""Tests for the reusable local and Slurm experiment runner."""

import json
import sys
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
    submit_experiment,
    submission_for_display,
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


def test_blocked_grids_support_conditional_parameter_ranges(tmp_path: Path) -> None:
    path = write_spec(
        tmp_path / "blocks.json",
        {
            "name": "conditional_ranges",
            "base": {
                "n_states": 2,
                "n_actions": 1,
                "density": 0.5,
                "delta_train": 2,
                "delta_check": 1,
            },
            "blocks": [
                {
                    "base": {"gamma": 0.9},
                    "grid": {"beta": [0.0, 2.0], "mdp_seed": [10, 11]},
                },
                {
                    "base": {"gamma": 0.99},
                    "grid": {"beta": [0.0, 22.0], "mdp_seed": [10, 11]},
                },
            ],
        },
    )

    points = expand_points(load_spec(path))

    assert len(points) == 8
    assert {
        (point["parameters"]["gamma"], point["parameters"]["beta"])
        for point in points
    } == {(0.9, 0.0), (0.9, 2.0), (0.99, 0.0), (0.99, 22.0)}


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
    expected_python = str(Path(sys.executable).absolute())
    assert "#SBATCH --partition=CPU" in array_text
    assert expected_python in array_text
    assert expected_python in merge_text
    assert "experiment_runner.py run-one" in array_text
    assert "OPENBLAS_NUM_THREADS" in array_text
    assert "experiment_runner.py merge" in merge_text
    assert compress_indices([0, 1, 2, 4, 6, 7]) == "0-2,4,6-7"


def test_submission_display_compacts_pending_indices() -> None:
    display = submission_for_display(
        {"manifest": "/tmp/manifest.json", "pending": list(range(5000))}
    )
    assert display == {
        "manifest": "/tmp/manifest.json",
        "pending_count": 5000,
        "pending_selector": "0-4999",
    }


def test_submission_chunks_indices_below_slurm_array_limit(tmp_path: Path) -> None:
    payload = tiny_spec(tmp_path)
    payload["slurm"]["array_chunk_size"] = 2
    spec_path = write_spec(tmp_path / "chunked.json", payload)

    submission = submit_experiment(spec_path, dry_run=True)

    assert submission["array_chunk_count"] == 2
    first, second = submission["array_chunks"]
    assert first["pending_selector"] == "0-1"
    assert second["pending_selector"] == "2-3"
    assert "--array=0-1%2" in first["command"]
    assert "--array=0-1%2" in second["command"]
    assert "--dependency=afterany:<ARRAY_JOB_ID_0>" in second["command"]
    assert Path(first["index_file"]).read_text(encoding="utf-8") == "0\n1\n"
    assert Path(second["index_file"]).read_text(encoding="utf-8") == "2\n3\n"

    array_script = Path(submission["manifest"]).with_name("run_array.sbatch")
    script_text = array_script.read_text(encoding="utf-8")
    assert 'sed -n "${index_line}p"' in script_text
    assert '--index "$experiment_index"' in script_text


def test_array_tasks_run_multiple_points_in_parallel(tmp_path: Path) -> None:
    payload = tiny_spec(tmp_path)
    payload["slurm"].update(
        {
            "cpus_per_task": 2,
            "points_per_task": 2,
            "array_chunk_size": 1000,
            "max_concurrent": 2,
        }
    )
    spec_path = write_spec(tmp_path / "batched.json", payload)

    submission = submit_experiment(spec_path, dry_run=True)

    assert submission["max_parallel_points"] == 4
    assert submission["array_chunk_count"] == 1
    chunk = submission["array_chunks"][0]
    assert chunk["array_task_count"] == 2
    assert chunk["pending_count"] == 4
    assert "--array=0-1%2" in chunk["command"]
    assert Path(chunk["index_file"]).read_text(encoding="utf-8") == "0,1\n2,3\n"

    array_script = Path(submission["manifest"]).with_name("run_array.sbatch")
    script_text = array_script.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=2" in script_text
    assert "export OPENBLAS_NUM_THREADS=1" in script_text
    assert "worker_pids" in script_text
    assert 'IFS=\',\' read -r -a experiment_indices' in script_text


def test_points_per_task_requires_one_cpu_per_point(tmp_path: Path) -> None:
    payload = tiny_spec(tmp_path)
    payload["slurm"].update({"cpus_per_task": 2, "points_per_task": 3})
    spec_path = write_spec(tmp_path / "too_many_workers.json", payload)

    with pytest.raises(ExperimentSpecError, match="points_per_task cannot exceed"):
        load_spec(spec_path)


def test_gamma_beta_epsilon_specs_have_expected_design() -> None:
    root = Path(__file__).resolve().parents[1]
    pilot = expand_points(
        load_spec(root / "experiment_specs" / "gamma_beta_epsilon_pilot.json")
    )
    full = expand_points(
        load_spec(root / "experiment_specs" / "gamma_beta_epsilon_full.json")
    )

    assert len(pilot) == 10
    assert len(full) == 3000
    assert {point["parameters"]["gamma"] for point in full} == {
        0.8,
        0.9,
        0.95,
    }
    assert {point["parameters"]["mdp_seed"] for point in full} == set(range(10))
    assert all(point["parameters"]["epsilon"] > 0 for point in full)
    assert all(point["parameters"]["init_seed"] == 1234 for point in full)
    assert all(point["parameters"]["tx_init"] == "always" for point in full)
    assert all(point["parameters"]["rx_init"] == "random" for point in full)
    assert all(point["parameters"]["delta_train"] == 70 for point in full)
    assert all(point["parameters"]["delta_check"] == 60 for point in full)
    full_slurm = load_spec(
        root / "experiment_specs" / "gamma_beta_epsilon_full.json"
    )["slurm"]
    assert full_slurm["cpus_per_task"] == 8
    assert full_slurm["points_per_task"] == 8
    assert full_slurm["max_concurrent"] == 4


def test_s8_gamma_specific_beta_spec_has_expected_design() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(
        root / "experiment_specs" / "gamma_beta_epsilon_s8_seed10_19.json"
    )
    points = expand_points(spec)

    assert len(points) == 2000
    assert {point["parameters"]["n_states"] for point in points} == {8}
    assert {point["parameters"]["n_actions"] for point in points} == {2}
    assert {point["parameters"]["density"] for point in points} == {0.625}
    assert {point["parameters"]["mdp_seed"] for point in points} == set(
        range(10, 20)
    )
    assert {point["parameters"]["init_seed"] for point in points} == {1111}
    assert {point["parameters"]["epsilon"] for point in points} == {
        0.03,
        0.06,
        0.09,
        0.12,
        0.15,
        0.18,
        0.21,
        0.24,
        0.27,
        0.3,
    }
    betas_by_gamma = {
        gamma: {
            point["parameters"]["beta"]
            for point in points
            if point["parameters"]["gamma"] == gamma
        }
        for gamma in (0.9, 0.99)
    }
    assert len(betas_by_gamma[0.9]) == 10
    assert min(betas_by_gamma[0.9]) == 0.0
    assert max(betas_by_gamma[0.9]) == 2.0
    assert len(betas_by_gamma[0.99]) == 10
    assert min(betas_by_gamma[0.99]) == 0.0
    assert max(betas_by_gamma[0.99]) == 22.0
    assert all(point["parameters"]["tx_init"] == "always" for point in points)
    assert all(point["parameters"]["rx_init"] == "random" for point in points)
    assert spec["slurm"]["cpus_per_task"] == 8
    assert spec["slurm"]["points_per_task"] == 8
    assert spec["slurm"]["max_concurrent"] == 4


def test_s6_mdp1111_three_tx_initializations_spec_has_expected_design() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = load_spec(
        root / "experiment_specs" / "gamma_beta_epsilon_s6_mdp1111_tx3.json"
    )
    points = expand_points(spec)

    assert len(points) == 240
    assert {point["parameters"]["n_states"] for point in points} == {6}
    assert {point["parameters"]["n_actions"] for point in points} == {2}
    assert {point["parameters"]["density"] for point in points} == {0.5}
    assert {point["parameters"]["reward_decay"] for point in points} == {10.0}
    assert {point["parameters"]["mdp_seed"] for point in points} == {1111}
    assert {point["parameters"]["init_seed"] for point in points} == {1111}
    assert {point["parameters"]["gamma"] for point in points} == {0.9, 0.99}
    assert {point["parameters"]["beta"] for point in points} == {
        0.0,
        0.05,
        0.1,
        0.15,
    }
    assert {point["parameters"]["epsilon"] for point in points} == {
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
    }
    assert {point["parameters"]["tx_init"] for point in points} == {
        "always",
        "never",
        "random",
    }
    assert {point["parameters"]["rx_init"] for point in points} == {"random"}
    assert all(point["parameters"]["delta_train"] == 70 for point in points)
    assert all(point["parameters"]["delta_check"] == 60 for point in points)
    assert spec["slurm"]["cpus_per_task"] == 8
    assert spec["slurm"]["points_per_task"] == 8
    assert spec["slurm"]["max_concurrent"] == 4
