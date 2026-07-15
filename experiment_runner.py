"""Reusable local and Slurm runner for RemotePOMDP experiment grids.

Each grid point writes its own atomic JSON shard.  A separate merge step builds
the single ``results.json`` file, avoiding concurrent writes from Slurm tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mdp import create_effcom_control_family, initial_distribution, select_density
from remote_api import SolverConfig, initialize_policies, run_api


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

DEFAULT_PARAMETERS: dict[str, Any] = {
    "n_states": 10,
    "n_actions": 2,
    "density": 0.5,
    "reward_decay": 10.0,
    "mdp_seed": 1234,
    "init_seed": 1234,
    "initial_state": None,
    "tx_init": "never",
    "rx_init": "fully_observed",
    "gamma": 0.9,
    "beta": 0.1,
    "epsilon": 0.1,
    "delta_train": 20,
    "delta_check": 10,
    "boundary_model": "tail",
    "boundary_tx_mode": "force_transmit",
    "vi_tol": 1e-10,
    "rx_accept_tol": 1e-9,
    "api_tol": 1e-9,
    "ne_tol": 1e-8,
    "margin_tol": 1e-10,
    "tie_tol": 1e-12,
    "max_vi_iterations": 100_000,
    "max_rx_iterations": 100,
    "max_api_iterations": 100,
    "compute_lower_bound": False,
}

DEFAULT_SLURM: dict[str, Any] = {
    "partition": "CPU",
    "time": "04:00:00",
    "cpus_per_task": 1,
    "points_per_task": 1,
    "mem": "4G",
    "max_concurrent": 16,
    "array_chunk_size": 1000,
    "python_module": None,
    "extra_sbatch_args": [],
}

TOP_LEVEL_KEYS = {
    "name",
    "description",
    "base",
    "grid",
    "points",
    "result_detail",
    "output_dir",
    "slurm",
}


class ExperimentSpecError(ValueError):
    """Raised when an experiment specification is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=json_default,
    )


def digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_text(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_integer(parameters: dict[str, Any], name: str, minimum: int | None = None) -> int:
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentSpecError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ExperimentSpecError(f"{name} must be at least {minimum}, got {value}")
    return value


def solver_config(parameters: dict[str, Any]) -> SolverConfig:
    return SolverConfig(
        gamma=parameters["gamma"],
        beta=parameters["beta"],
        epsilon=parameters["epsilon"],
        delta_train=parameters["delta_train"],
        delta_check=parameters["delta_check"],
        boundary_model=parameters["boundary_model"],
        boundary_tx_mode=parameters["boundary_tx_mode"],
        vi_tol=parameters["vi_tol"],
        rx_accept_tol=parameters["rx_accept_tol"],
        api_tol=parameters["api_tol"],
        ne_tol=parameters["ne_tol"],
        margin_tol=parameters["margin_tol"],
        tie_tol=parameters["tie_tol"],
        max_vi_iterations=parameters["max_vi_iterations"],
        max_rx_iterations=parameters["max_rx_iterations"],
        max_api_iterations=parameters["max_api_iterations"],
    )


def validate_parameters(parameters: dict[str, Any]) -> None:
    unknown = set(parameters) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ExperimentSpecError(f"unknown run parameter(s): {sorted(unknown)}")

    _require_integer(parameters, "n_states", 2)
    _require_integer(parameters, "n_actions", 1)
    _require_integer(parameters, "mdp_seed")
    _require_integer(parameters, "init_seed")
    _require_integer(parameters, "delta_train", 1)
    _require_integer(parameters, "delta_check", 0)
    _require_integer(parameters, "max_vi_iterations", 1)
    _require_integer(parameters, "max_rx_iterations", 1)
    _require_integer(parameters, "max_api_iterations", 1)

    if parameters["tx_init"] not in {"never", "random", "always", "state_change"}:
        raise ExperimentSpecError(
            "tx_init must be one of: never, random, always, state_change"
        )
    if parameters["rx_init"] not in {"fully_observed", "random"}:
        raise ExperimentSpecError("rx_init must be fully_observed or random")
    if not isinstance(parameters["compute_lower_bound"], bool):
        raise ExperimentSpecError("compute_lower_bound must be true or false")

    config = solver_config(parameters)
    family = create_effcom_control_family(
        n_states=parameters["n_states"],
        n_actions=parameters["n_actions"],
        reward_decay=parameters["reward_decay"],
        seed=parameters["mdp_seed"],
    )
    select_density(family, parameters["density"])
    initial_distribution(parameters["n_states"], parameters["initial_state"])
    # Accessing this property also validates the configured boundary combination.
    _ = config.effective_boundary_tx_mode


def normalize_slurm(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ExperimentSpecError("slurm must be a JSON object")
    unknown = set(raw) - set(DEFAULT_SLURM)
    if unknown:
        raise ExperimentSpecError(f"unknown Slurm option(s): {sorted(unknown)}")
    slurm = {**DEFAULT_SLURM, **raw}
    for name in (
        "cpus_per_task",
        "points_per_task",
        "max_concurrent",
        "array_chunk_size",
    ):
        if (
            isinstance(slurm[name], bool)
            or not isinstance(slurm[name], int)
            or slurm[name] < 1
        ):
            raise ExperimentSpecError(f"slurm.{name} must be a positive integer")
    if slurm["points_per_task"] > slurm["cpus_per_task"]:
        raise ExperimentSpecError(
            "slurm.points_per_task cannot exceed slurm.cpus_per_task; "
            "reserve at least one CPU core for every concurrent point"
        )
    for name in ("partition", "time", "mem"):
        if not isinstance(slurm[name], str) or not slurm[name].strip():
            raise ExperimentSpecError(f"slurm.{name} must be a non-empty string")
    if slurm["python_module"] is not None and not isinstance(slurm["python_module"], str):
        raise ExperimentSpecError("slurm.python_module must be a string or null")
    extra = slurm["extra_sbatch_args"]
    if not isinstance(extra, list) or not all(
        isinstance(item, str) and item.startswith("--") and not any(c.isspace() for c in item)
        for item in extra
    ):
        raise ExperimentSpecError(
            "slurm.extra_sbatch_args must be a list of whitespace-free --options"
        )
    return slurm


def load_spec(path: Path) -> dict[str, Any]:
    spec = load_json(path)
    if not isinstance(spec, dict):
        raise ExperimentSpecError("the experiment specification must be a JSON object")
    unknown = set(spec) - TOP_LEVEL_KEYS
    if unknown:
        raise ExperimentSpecError(f"unknown top-level field(s): {sorted(unknown)}")
    name = spec.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ExperimentSpecError(
            "name must contain only letters, numbers, '.', '_' and '-', and start alphanumeric"
        )
    detail = spec.get("result_detail", "compact")
    if detail not in {"compact", "full"}:
        raise ExperimentSpecError("result_detail must be compact or full")
    raw_base = spec.get("base", {})
    if not isinstance(raw_base, dict):
        raise ExperimentSpecError("base must be a JSON object")
    unknown_base = set(raw_base) - set(DEFAULT_PARAMETERS)
    if unknown_base:
        raise ExperimentSpecError(
            f"unknown base parameter(s): {sorted(unknown_base)}"
        )
    spec["base"] = {**DEFAULT_PARAMETERS, **raw_base}
    spec["result_detail"] = detail
    spec["slurm"] = normalize_slurm(spec.get("slurm"))
    return spec


def expand_points(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw_base = spec.get("base", {})
    if not isinstance(raw_base, dict):
        raise ExperimentSpecError("base must be a JSON object")
    unknown = set(raw_base) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ExperimentSpecError(f"unknown base parameter(s): {sorted(unknown)}")
    base = {**DEFAULT_PARAMETERS, **raw_base}

    has_grid = "grid" in spec
    has_points = "points" in spec
    if has_grid and has_points:
        raise ExperimentSpecError("use either grid or points, not both")

    overrides: list[dict[str, Any]]
    if has_points:
        raw_points = spec["points"]
        if not isinstance(raw_points, list) or not raw_points:
            raise ExperimentSpecError("points must be a non-empty list")
        overrides = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, dict):
                raise ExperimentSpecError(f"points[{index}] must be a JSON object")
            unknown = set(point) - set(DEFAULT_PARAMETERS)
            if unknown:
                raise ExperimentSpecError(
                    f"unknown parameter(s) in points[{index}]: {sorted(unknown)}"
                )
            overrides.append(point)
    else:
        raw_grid = spec.get("grid", {})
        if not isinstance(raw_grid, dict):
            raise ExperimentSpecError("grid must be a JSON object")
        unknown = set(raw_grid) - set(DEFAULT_PARAMETERS)
        if unknown:
            raise ExperimentSpecError(f"unknown grid parameter(s): {sorted(unknown)}")
        names = list(raw_grid)
        values: list[list[Any]] = []
        for name in names:
            candidates = raw_grid[name]
            if not isinstance(candidates, list) or not candidates:
                raise ExperimentSpecError(f"grid.{name} must be a non-empty list")
            values.append(candidates)
        overrides = [dict(zip(names, combination)) for combination in itertools.product(*values)]
        if not overrides:
            overrides = [{}]

    points: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, override in enumerate(overrides):
        parameters = {**base, **override}
        validate_parameters(parameters)
        run_id = digest(parameters)
        if run_id in seen:
            raise ExperimentSpecError(
                f"duplicate experiment point at index {index} (run_id={run_id})"
            )
        seen.add(run_id)
        points.append({"index": index, "run_id": run_id, "parameters": parameters})
    return points


def resolve_output_dir(spec: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        output = override
    elif "output_dir" in spec:
        raw = spec["output_dir"]
        if not isinstance(raw, str) or not raw:
            raise ExperimentSpecError("output_dir must be a non-empty path string")
        output = Path(raw)
    else:
        output = Path("experiment_runs") / spec["name"]
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    return output.resolve()


def plan_experiment(spec_path: Path, output_dir: Path | None = None) -> Path:
    spec_path = spec_path.resolve()
    spec = load_spec(spec_path)
    points = expand_points(spec)
    target = resolve_output_dir(spec, output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "runs").mkdir(exist_ok=True)
    (target / "logs").mkdir(exist_ok=True)
    manifest_path = target / "manifest.json"
    spec_fingerprint = digest(spec, length=64)

    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing.get("spec_digest") != spec_fingerprint:
            raise ExperimentSpecError(
                f"{manifest_path} belongs to a different specification; "
                "choose a new experiment name or output_dir"
            )
        experiment_snapshot = target / "experiment.json"
        if not experiment_snapshot.exists():
            atomic_write_json(experiment_snapshot, spec)
        return manifest_path

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": spec["name"],
        "description": spec.get("description", ""),
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "spec_path": str(spec_path),
        "spec_digest": spec_fingerprint,
        "output_dir": str(target),
        "result_detail": spec["result_detail"],
        "slurm": spec["slurm"],
        "expected_runs": len(points),
        "points": points,
        "spec": spec,
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(target / "experiment.json", spec)
    return manifest_path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path.resolve())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentSpecError(
            f"unsupported manifest schema: {manifest.get('schema_version')!r}"
        )
    if len(manifest.get("points", [])) != manifest.get("expected_runs"):
        raise ExperimentSpecError("manifest point count does not match expected_runs")
    return manifest


def shard_path(manifest: dict[str, Any], point: dict[str, Any]) -> Path:
    return (
        Path(manifest["output_dir"])
        / "runs"
        / f"{point['index']:06d}_{point['run_id']}.json"
    )


def array_hash(array: np.ndarray) -> str:
    digest_builder = hashlib.sha256()
    digest_builder.update(str(array.shape).encode("ascii"))
    digest_builder.update(np.ascontiguousarray(array).view(np.uint8))
    return digest_builder.hexdigest()[:16]


def build_initial_policies(
    mdp: Any, config: SolverConfig, parameters: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    tx_init = parameters["tx_init"]
    rx_init = parameters["rx_init"]
    # Using random here for custom Tx modes preserves the RNG order used by the
    # existing always-random and state-change-random notebooks.
    generator_tx_mode = "never" if tx_init == "never" else "random"
    base_tx, pi_rx = initialize_policies(
        mdp,
        config,
        seed=parameters["init_seed"],
        tx_mode=generator_tx_mode,
        rx_mode=rx_init,
    )
    shape = (mdp.n_states, config.delta_train + 1, mdp.n_states)
    if tx_init in {"never", "random"}:
        pi_tx = base_tx
    elif tx_init == "always":
        pi_tx = np.ones(shape, dtype=np.int64)
    elif tx_init == "state_change":
        state = np.arange(mdp.n_states)[:, None, None]
        last_received = np.arange(mdp.n_states)[None, None, :]
        pi_tx = np.broadcast_to(state != last_received, shape).astype(np.int64).copy()
        if config.effective_boundary_tx_mode == "force_transmit":
            pi_tx[:, config.delta_train, :] = 1
    else:  # pragma: no cover - validate_parameters rejects this first.
        raise ExperimentSpecError(f"unknown tx_init: {tx_init}")
    return pi_tx, pi_rx


def stop_reason(result: Any, config: SolverConfig) -> str:
    final_step = result.history[-1] if result.history else None
    if final_step and not final_step["tx_changed"] and not final_step["rx_changed"]:
        return "policies_unchanged"
    if final_step and abs(float(final_step["improvement"])) <= config.api_tol:
        return "api_tolerance"
    if result.api_iterations >= config.max_api_iterations:
        return "iteration_cap"
    return "other"


def execute_point(point: dict[str, Any], result_detail: str) -> dict[str, Any]:
    parameters = point["parameters"]
    family = create_effcom_control_family(
        n_states=parameters["n_states"],
        n_actions=parameters["n_actions"],
        reward_decay=parameters["reward_decay"],
        seed=parameters["mdp_seed"],
    )
    mdp = select_density(family, parameters["density"])
    config = solver_config(parameters)
    mu0 = initial_distribution(parameters["n_states"], parameters["initial_state"])
    initial_tx, initial_rx = build_initial_policies(mdp, config, parameters)
    result = run_api(
        mdp,
        config,
        mu0=mu0,
        seed=parameters["init_seed"],
        initial_pi_tx=initial_tx,
        initial_pi_rx=initial_rx,
        compute_lower_bound=parameters["compute_lower_bound"],
    )
    revealing = result.revealing
    final_step = result.history[-1] if result.history else None
    has_core_violations = not revealing.is_revealing
    performance_upper_bound = float(result.objective)
    performance_lower_bound = (
        None
        if not has_core_violations
        else (
            None
            if result.lower_bound_objective is None
            else float(result.lower_bound_objective)
        )
    )
    if has_core_violations and performance_lower_bound is None:
        raise RuntimeError(
            "a run with core revealing violations must compute a lower bound"
        )
    performance_gap = (
        None
        if performance_lower_bound is None
        else performance_upper_bound - performance_lower_bound
    )

    objective_history = [
        {
            "api_iteration": 0,
            "objective": float(result.diagnostics["initial_objective"]),
            "improvement": None,
        }
    ] + [
        {
            "api_iteration": int(item["api_iteration"]),
            "objective": float(item["objective"]),
            "improvement": float(item["improvement"]),
        }
        for item in result.history
    ]
    violation_history = [
        {
            "api_iteration": int(item["api_iteration"]),
            "core_violation_count": int(item["core_violation_count"]),
            "buffer_violation_count": int(item["buffer_violation_count"]),
            "boundary_adjacent_violation_count": int(
                item["boundary_adjacent_violation_count"]
            ),
            "boundary_transmission_count": int(item["boundary_transmission_count"]),
        }
        for item in result.violation_history
    ]

    record: dict[str, Any] = {
        "run_id": point["run_id"],
        "index": point["index"],
        "status": "ok",
        **parameters,
        "margin": float(config.theorem_margin),
        "margin_region": config.margin_region,
        "revealing": bool(revealing.is_revealing),
        "performance": {
            "kind": (
                "upper_bound_only"
                if not has_core_violations
                else "upper_and_lower_bounds"
            ),
            "upper_bound": performance_upper_bound,
            "lower_bound": performance_lower_bound,
            "gap": performance_gap,
        },
        "tx_regret": float(result.tx_regret),
        "rx_restricted_regret": float(result.rx_restricted_regret),
        "approximate_restricted_ne": bool(result.approximate_restricted_ne),
        "api_converged": bool(result.converged),
        "api_iterations": int(result.api_iterations),
        "api_stop_reason": stop_reason(result, config),
        "final_abs_improvement": (
            None if final_step is None else abs(float(final_step["improvement"]))
        ),
        "initial_tx_fraction": float(initial_tx[:, : config.delta_train, :].mean()),
        "initial_tx_hash": array_hash(initial_tx),
        "initial_rx_hash": array_hash(initial_rx),
        "final_tx_fraction": float(result.pi_tx[:, : config.delta_train, :].mean()),
        "final_tx_hash": array_hash(result.pi_tx),
        "final_rx_hash": array_hash(result.pi_rx),
        "core_violation_count": int(len(revealing.core_violations)),
        "buffer_violation_count": int(len(revealing.buffer_violations)),
        "boundary_adjacent_violation_count": int(
            len(revealing.boundary_adjacent_violations)
        ),
        "boundary_transmission_count": int(len(revealing.boundary_transmissions)),
        "core_violations": revealing.core_violations,
        "buffer_violations": revealing.buffer_violations,
        "boundary_adjacent_violations": revealing.boundary_adjacent_violations,
        "reachable_statistics": revealing.statistics,
        "objective_history": objective_history,
        "violation_history": violation_history,
    }
    if result_detail == "full":
        record["diagnostics"] = result.diagnostics
    return record


def slurm_context() -> dict[str, Any]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
    )
    return {name.lower(): os.environ.get(name) for name in names if os.environ.get(name)}


def run_manifest_index(
    manifest_path: Path, index: int | None = None, force: bool = False
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    if index is None:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_index is None:
            raise ExperimentSpecError(
                "run-one needs --index outside a Slurm array task"
            )
        index = int(raw_index)
    if index < 0 or index >= len(manifest["points"]):
        raise ExperimentSpecError(
            f"index {index} is outside [0, {len(manifest['points']) - 1}]"
        )
    point = manifest["points"][index]
    target = shard_path(manifest, point)
    if target.exists() and not force:
        existing = load_json(target)
        if existing.get("status") == "ok" and existing.get("run_id") == point["run_id"]:
            return existing

    started_at = utc_now()
    started = time.perf_counter()
    try:
        record = execute_point(point, manifest["result_detail"])
    except Exception as exc:
        record = {
            "run_id": point["run_id"],
            "index": point["index"],
            "status": "error",
            **point["parameters"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    record.update(
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "hostname": socket.gethostname(),
            "slurm": slurm_context(),
        }
    )
    atomic_write_json(target, record)
    return record


def collect_records(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[int]]:
    records: list[dict[str, Any]] = []
    missing: list[int] = []
    for point in manifest["points"]:
        path = shard_path(manifest, point)
        if not path.exists():
            missing.append(point["index"])
            continue
        try:
            record = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            record = {
                "run_id": point["run_id"],
                "index": point["index"],
                "status": "corrupt",
                **point["parameters"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if record.get("run_id") != point["run_id"]:
            record = {
                "run_id": point["run_id"],
                "index": point["index"],
                "status": "corrupt",
                **point["parameters"],
                "error": f"shard run_id does not match {point['run_id']}",
            }
        records.append(record)
    records.sort(key=lambda item: item["index"])
    return records, missing


def result_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    records, missing = collect_records(manifest)
    ok = sum(record.get("status") == "ok" for record in records)
    error = sum(record.get("status") == "error" for record in records)
    corrupt = sum(record.get("status") == "corrupt" for record in records)
    return {
        "expected": manifest["expected_runs"],
        "completed": len(records),
        "ok": ok,
        "error": error,
        "corrupt": corrupt,
        "missing": len(missing),
        "pending_indices": missing,
    }


def markdown_value(value: Any) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)
    return rendered.replace("|", "\\|").replace("\n", " ")


def performance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "ok"]
    upper_only = [
        record
        for record in successful
        if record.get("performance", {}).get("kind") == "upper_bound_only"
    ]
    intervals = [
        record
        for record in successful
        if record.get("performance", {}).get("kind") == "upper_and_lower_bounds"
    ]
    upper_values = [float(record["performance"]["upper_bound"]) for record in successful]
    lower_values = [
        float(record["performance"]["lower_bound"])
        for record in intervals
        if record["performance"].get("lower_bound") is not None
    ]
    gaps = [
        float(record["performance"]["gap"])
        for record in intervals
        if record["performance"].get("gap") is not None
    ]
    return {
        "upper_bound_only_runs": len(upper_only),
        "interval_runs": len(intervals),
        "upper_bound_min": min(upper_values, default=None),
        "upper_bound_max": max(upper_values, default=None),
        "lower_bound_min": min(lower_values, default=None),
        "lower_bound_max": max(lower_values, default=None),
        "maximum_bound_gap": max(gaps, default=None),
    }


def render_experiment_readme(
    manifest: dict[str, Any], results_payload: dict[str, Any]
) -> str:
    spec = manifest["spec"]
    summary = results_payload["summary"]
    bounds = results_payload["performance_summary"]
    output_dir = Path(manifest["output_dir"])
    base = spec.get("base", {})
    grid = spec.get("grid")
    points = spec.get("points")

    lines = [
        f"# {manifest['name']}",
        "",
        manifest["description"] or "RemotePOMDP experiment.",
        "",
        f"Generated: `{results_payload['generated_at']}`",
        "",
        "## Completion",
        "",
        "| Expected | Successful | Errors | Corrupt | Missing |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {summary['expected']} | {summary['ok']} | {summary['error']} | "
            f"{summary['corrupt']} | {summary['missing']} |"
        ),
        "",
        "## Performance values",
        "",
        (
            "Every successful run stores `performance.upper_bound`. Runs with no "
            "reachable core revealing violations report only that upper bound. "
            "Runs with one or more core violations additionally store "
            "`performance.lower_bound` and `performance.gap`."
        ),
        "",
        "| Upper only (no core violations) | Upper + lower | Upper range | Lower range | Maximum gap |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {bounds['upper_bound_only_runs']} | {bounds['interval_runs']} | "
            f"{markdown_value([bounds['upper_bound_min'], bounds['upper_bound_max']])} | "
            f"{markdown_value([bounds['lower_bound_min'], bounds['lower_bound_max']])} | "
            f"{markdown_value(bounds['maximum_bound_gap'])} |"
        ),
        "",
        "The core region is `age <= delta_check`; only core violations determine "
        "whether the lower performance bound is required. Buffer and "
        "boundary-adjacent violations remain separate diagnostics.",
        "",
        (
            "The final `core_violations`, `buffer_violations`, and "
            "`boundary_adjacent_violations` lists retain each individual "
            "violation's `state`, `age`, `last_received`, transmission and "
            "receiver actions, `distance_to_boundary`, and "
            "`discounted_occupancy`. An age histogram can therefore be derived "
            "without storing a duplicate age-only list."
        ),
        "",
        (
            "Each run keeps `gamma`, `beta`, `epsilon`, `mdp_seed`, `init_seed`, "
            "`tx_init`, and `rx_init` as top-level columns, so local plots can "
            "group by channel settings and distinguish physical-MDP randomness "
            "from policy-initialization randomness."
        ),
        "",
        "## Shared parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for name, value in base.items():
        lines.append(f"| `{name}` | {markdown_value(value)} |")

    lines.extend(["", "## Experiment design", ""])
    if grid is not None:
        lines.extend(["Cartesian grid:", "", "| Parameter | Values |", "|---|---|"])
        for name, values in grid.items():
            lines.append(f"| `{name}` | {markdown_value(values)} |")
    elif points is not None:
        lines.append(f"Explicit point list with {len(points)} entries.")
    else:
        lines.append("One run using the shared parameters.")

    lines.extend(
        [
            "",
            "## Slurm resources",
            "",
            "| Setting | Value |",
            "|---|---|",
        ]
    )
    for name, value in manifest["slurm"].items():
        lines.append(f"| `{name}` | {markdown_value(value)} |")

    lines.extend(
        [
            "",
            (
                "Maximum concurrent experiment points: "
                f"`{manifest['slurm']['max_concurrent'] * manifest['slurm']['points_per_task']}` "
                f"({manifest['slurm']['max_concurrent']} Slurm array tasks × "
                f"{manifest['slurm']['points_per_task']} points per task)."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `experiment.json`: normalized experiment specification.",
            "- `manifest.json`: expanded points with stable run IDs.",
            "- `results.json`: merged data used for local analysis and plots.",
            "- `runs/`: one atomic result shard per point.",
            "- `logs/`: Slurm stdout and stderr.",
            "",
            "A minimal local loader:",
            "",
            "```python",
            "import json",
            "from pathlib import Path",
            "import pandas as pd",
            "",
            "payload = json.loads(Path(\"results.json\").read_text())",
            "runs = [run for run in payload[\"runs\"] if run[\"status\"] == \"ok\"]",
            "df = pd.json_normalize(runs)",
            "# Columns include gamma, beta, epsilon, mdp_seed, init_seed,",
            "# tx_init, rx_init, performance.upper_bound, and performance.lower_bound.",
            "```",
            "",
            "## Full normalized specification",
            "",
            "```json",
            json.dumps(spec, indent=2, ensure_ascii=False, default=json_default),
            "```",
            "",
            "## Retrieve from the cluster",
            "",
            "From a local machine on the Télécom Paris network or VPN:",
            "",
            "```bash",
            "rsync -avz <tp-username>@ids-store.enst.fr:" + str(output_dir) + "/ ./" + manifest["name"] + "/",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def merge_results(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    records, missing = collect_records(manifest)
    summary = result_summary(manifest)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": manifest["name"],
        "description": manifest["description"],
        "generated_at": utc_now(),
        "manifest": str(manifest_path),
        "spec_digest": manifest["spec_digest"],
        "spec": manifest["spec"],
        "summary": summary,
        "performance_summary": performance_summary(records),
        "pending_indices": missing,
        "runs": records,
    }
    target = Path(manifest["output_dir"]) / "results.json"
    atomic_write_json(target, payload)
    atomic_write_text(
        Path(manifest["output_dir"]) / "README.md",
        render_experiment_readme(manifest, payload),
    )
    return target


def pending_indices(manifest: dict[str, Any]) -> list[int]:
    records, missing = collect_records(manifest)
    retry = [record["index"] for record in records if record.get("status") != "ok"]
    return sorted(set(missing + retry))


def compress_indices(indices: Iterable[int]) -> str:
    values = sorted(set(indices))
    if not values:
        return ""
    parts: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def _module_setup(module_name: str | None) -> str:
    if not module_name:
        return ""
    return f"module purge\nmodule load {shlex.quote(module_name)}\n"


def write_slurm_scripts(manifest_path: Path) -> tuple[Path, Path]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    output_dir = Path(manifest["output_dir"])
    logs_dir = output_dir / "logs"
    slurm = manifest["slurm"]
    safe_job_name = re.sub(r"[^A-Za-z0-9_-]", "_", manifest["name"])[:80]
    runner = PROJECT_ROOT / "experiment_runner.py"
    # Capture the interpreter that planned/submitted the experiment instead of
    # relying on PATH inside a Slurm login shell.  In particular, this keeps a
    # caller's virtual environment (and its installed NumPy) available on the
    # compute node.
    python_executable = Path(sys.executable).absolute()
    module_setup = _module_setup(slurm["python_module"])
    threads_per_point = max(1, slurm["cpus_per_task"] // slurm["points_per_task"])
    common_environment = (
        f"export OMP_NUM_THREADS={threads_per_point}\n"
        f"export OPENBLAS_NUM_THREADS={threads_per_point}\n"
        f"export MKL_NUM_THREADS={threads_per_point}\n"
        f"export NUMEXPR_NUM_THREADS={threads_per_point}\n"
    )

    array_script = output_dir / "run_array.sbatch"
    array_content = f"""#!/bin/bash -l
#SBATCH --job-name={safe_job_name}
#SBATCH --partition={slurm['partition']}
#SBATCH --time={slurm['time']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['mem']}
#SBATCH --output={logs_dir}/%A_%a.out
#SBATCH --error={logs_dir}/%A_%a.err

set -euo pipefail
index_file="${{1:?missing array index file}}"
index_line=$((SLURM_ARRAY_TASK_ID + 1))
experiment_index_group="$(sed -n "${{index_line}}p" "$index_file")"
if [[ -z "$experiment_index_group" ]]; then
    echo "No manifest index mapped for Slurm task $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
{module_setup}{common_environment}cd {shlex.quote(str(PROJECT_ROOT))}
IFS=',' read -r -a experiment_indices <<< "$experiment_index_group"
worker_pids=()
for experiment_index in "${{experiment_indices[@]}}"; do
    {shlex.quote(str(python_executable))} {shlex.quote(str(runner))} run-one {shlex.quote(str(manifest_path))} --index "$experiment_index" &
    worker_pids+=("$!")
done

worker_status=0
for worker_pid in "${{worker_pids[@]}}"; do
    if ! wait "$worker_pid"; then
        worker_status=1
    fi
done
exit "$worker_status"
"""
    atomic_write_text(array_script, array_content, mode=0o755)

    merge_script = output_dir / "merge_results.sbatch"
    merge_content = f"""#!/bin/bash -l
#SBATCH --job-name={safe_job_name}_merge
#SBATCH --partition={slurm['partition']}
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output={logs_dir}/%j_merge.out
#SBATCH --error={logs_dir}/%j_merge.err

set -euo pipefail
{module_setup}cd {shlex.quote(str(PROJECT_ROOT))}
{shlex.quote(str(python_executable))} {shlex.quote(str(runner))} merge {shlex.quote(str(manifest_path))}
"""
    atomic_write_text(merge_script, merge_content, mode=0o755)
    return array_script, merge_script


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def prepare_array_chunks(
    manifest: dict[str, Any], pending: list[int]
) -> list[dict[str, Any]]:
    """Write stable local-task-to-manifest-index mappings for Slurm chunks."""
    chunk_size = manifest["slurm"]["array_chunk_size"]
    points_per_task = manifest["slurm"]["points_per_task"]
    index_groups = [
        pending[start : start + points_per_task]
        for start in range(0, len(pending), points_per_task)
    ]
    submission_dir = (
        Path(manifest["output_dir"])
        / "submissions"
        / f"{time.time_ns()}_{digest(pending)}"
    )
    chunks: list[dict[str, Any]] = []
    for chunk_number, start in enumerate(range(0, len(index_groups), chunk_size)):
        chunk_groups = index_groups[start : start + chunk_size]
        indices = [index for group in chunk_groups for index in group]
        index_file = submission_dir / f"indices_{chunk_number:04d}.txt"
        atomic_write_text(
            index_file,
            "".join(
                ",".join(str(index) for index in group) + "\n"
                for group in chunk_groups
            ),
        )
        chunks.append(
            {
                "chunk": chunk_number,
                "indices": indices,
                "index_groups": chunk_groups,
                "index_file": str(index_file),
            }
        )
    return chunks


def build_array_command(
    array_script: Path,
    extra_sbatch_args: list[str],
    chunk: dict[str, Any],
    max_concurrent: int,
    dependency: str | None = None,
) -> list[str]:
    command = ["sbatch", "--parsable", *extra_sbatch_args]
    if dependency is not None:
        command.append(f"--dependency=afterany:{dependency}")
    command.extend(
        [
            f"--array=0-{len(chunk['index_groups']) - 1}%{max_concurrent}",
            str(array_script),
            chunk["index_file"],
        ]
    )
    return command


def chunk_summary(chunk: dict[str, Any], command: list[str]) -> dict[str, Any]:
    return {
        "chunk": chunk["chunk"],
        "array_task_count": len(chunk["index_groups"]),
        "pending_count": len(chunk["indices"]),
        "pending_selector": compress_indices(chunk["indices"]),
        "index_file": chunk["index_file"],
        "command": shlex.join(command),
    }


def submit_experiment(
    spec_path: Path, output_dir: Path | None = None, dry_run: bool = False
) -> dict[str, Any]:
    manifest_path = plan_experiment(spec_path, output_dir)
    manifest = load_manifest(manifest_path)
    array_script, merge_script = write_slurm_scripts(manifest_path)
    pending = pending_indices(manifest)
    if not pending:
        results_path = merge_results(manifest_path)
        return {
            "manifest": str(manifest_path),
            "results": str(results_path),
            "message": "all points were already successful; results were merged",
        }

    extra = manifest["slurm"]["extra_sbatch_args"]
    max_concurrent = manifest["slurm"]["max_concurrent"]
    max_parallel_points = max_concurrent * manifest["slurm"]["points_per_task"]
    chunks = prepare_array_chunks(manifest, pending)
    if dry_run:
        chunk_details = []
        for position, chunk in enumerate(chunks):
            dependency = None if position == 0 else f"<ARRAY_JOB_ID_{position - 1}>"
            command = build_array_command(
                array_script, extra, chunk, max_concurrent, dependency
            )
            chunk_details.append(chunk_summary(chunk, command))
        payload = {
            "manifest": str(manifest_path),
            "pending": pending,
            "max_parallel_points": max_parallel_points,
            "array_chunk_count": len(chunks),
            "array_chunks": chunk_details,
            "merge_command_template": shlex.join(
                [
                    "sbatch",
                    "--parsable",
                    *extra,
                    f"--dependency=afterany:<ARRAY_JOB_ID_{len(chunks) - 1}>",
                    str(merge_script),
                ]
            ),
        }
        if len(chunk_details) == 1:
            payload["array_command"] = chunk_details[0]["command"]
        return payload

    array_jobs = []
    previous_job_id: str | None = None
    for chunk in chunks:
        array_command = build_array_command(
            array_script, extra, chunk, max_concurrent, previous_job_id
        )
        array_output = run_command(array_command)
        array_job_id = array_output.split(";", 1)[0]
        job = chunk_summary(chunk, array_command)
        job["job_id"] = array_job_id
        array_jobs.append(job)
        previous_job_id = array_job_id

    assert previous_job_id is not None
    merge_command = [
        "sbatch",
        "--parsable",
        *extra,
        f"--dependency=afterany:{previous_job_id}",
        str(merge_script),
    ]
    merge_output = run_command(merge_command)
    merge_job_id = merge_output.split(";", 1)[0]
    submission = {
        "submitted_at": utc_now(),
        "manifest": str(manifest_path),
        "pending_indices": pending,
        "max_parallel_points": max_parallel_points,
        "array_chunk_count": len(chunks),
        "array_jobs": array_jobs,
        "merge_job_id": merge_job_id,
        "merge_command": shlex.join(merge_command),
    }
    if len(array_jobs) == 1:
        submission["array_job_id"] = array_jobs[0]["job_id"]
        submission["array_command"] = array_jobs[0]["command"]
    atomic_write_json(Path(manifest["output_dir"]) / "submission.json", submission)
    return submission


def submission_for_display(submission: dict[str, Any]) -> dict[str, Any]:
    """Compact large pending-index lists for human-readable CLI output."""
    display = dict(submission)
    for key in ("pending", "pending_indices"):
        values = display.pop(key, None)
        if values is not None:
            display["pending_count"] = len(values)
            display["pending_selector"] = compress_indices(values)
            break
    return display


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="validate a spec and create a manifest")
    plan.add_argument("spec", type=Path)
    plan.add_argument("--output-dir", type=Path)

    run_one = subparsers.add_parser("run-one", help="run one manifest point")
    run_one.add_argument("manifest", type=Path)
    run_one.add_argument("--index", type=int)
    run_one.add_argument("--force", action="store_true")

    merge = subparsers.add_parser("merge", help="merge point shards into results.json")
    merge.add_argument("manifest", type=Path)

    status = subparsers.add_parser("status", help="show completion counts")
    status.add_argument("manifest", type=Path)

    submit = subparsers.add_parser("submit", help="plan and submit a Slurm job array")
    submit.add_argument("spec", type=Path)
    submit.add_argument("--output-dir", type=Path)
    submit.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            path = plan_experiment(args.spec, args.output_dir)
            manifest = load_manifest(path)
            print(
                json.dumps(
                    {
                        "manifest": str(path),
                        "expected_runs": manifest["expected_runs"],
                        "output_dir": manifest["output_dir"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run-one":
            record = run_manifest_index(args.manifest, args.index, args.force)
            print(
                json.dumps(
                    {
                        "index": record["index"],
                        "run_id": record["run_id"],
                        "status": record["status"],
                        "elapsed_seconds": record.get("elapsed_seconds"),
                    },
                    indent=2,
                )
            )
            return 0 if record["status"] == "ok" else 1
        if args.command == "merge":
            target = merge_results(args.manifest)
            payload = load_json(target)
            print(json.dumps({"results": str(target), **payload["summary"]}, indent=2))
            return 0
        if args.command == "status":
            manifest = load_manifest(args.manifest)
            print(json.dumps(result_summary(manifest), indent=2))
            return 0
        if args.command == "submit":
            submission = submit_experiment(args.spec, args.output_dir, args.dry_run)
            print(json.dumps(submission_for_display(submission), indent=2))
            return 0
    except (ExperimentSpecError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
