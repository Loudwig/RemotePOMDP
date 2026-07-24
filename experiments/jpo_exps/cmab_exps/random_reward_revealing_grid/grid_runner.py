"""Shared resumable runner for the random-reward CMAB revealing grids."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from jpo_model import JPOConfig, JPOModel
from jpo_policy import PolicyAnalysisConfig, analyze_jpo_policy
from jpo_sarsop import NativeSARSOPConfig, load_sarsop_policy, run_native_sarsop
from mdp import create_cmab_random_reward_family, select_density


DEFAULT_SOLVER = {
    "search_epsilon": 0.01,
    "precision": 0.01,
    "max_time": 500.0,
    "max_steps": 1_000_000,
    "kappa": 0.5,
    "delta": 0.0001,
    "prune_threshold": 0.10,
    "initial_bound_residual": 1e-8,
    "initial_bound_max_time": 30.0,
    "initial_upper_bound": "fully_observable",
    "analysis_tail_tolerance": 1e-8,
    "max_belief_nodes": 2_000_000,
    "export_beliefs": False,
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonify(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _point_id(gamma_index: int, beta_index: int, epsilon_index: int) -> str:
    return f"g{gamma_index}_b{beta_index:02d}_e{epsilon_index:02d}"


def _cmab_margin(gamma: float, beta: float, epsilon: float) -> float:
    return beta - gamma * epsilon * (1.0 - epsilon) / (1.0 - gamma**2)


def _with_current_margin(point: dict[str, Any]) -> dict[str, Any]:
    updated = dict(point)
    margin = _cmab_margin(
        float(updated["gamma"]),
        float(updated["beta"]),
        float(updated["epsilon"]),
    )
    updated["margin"] = margin
    updated["margin_region"] = (
        "m>0" if margin > 0.0 else ("m<0" if margin < 0.0 else "m=0")
    )
    return updated


def build_points(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    index = 0
    for gamma_index, gamma in enumerate(experiment["gammas"]):
        for beta_index, beta in enumerate(experiment["betas"]):
            for epsilon_index, epsilon in enumerate(experiment["epsilons"]):
                margin = _cmab_margin(gamma, beta, epsilon)
                points.append(
                    {
                        "index": index,
                        "point_id": _point_id(
                            gamma_index, beta_index, epsilon_index
                        ),
                        "gamma_index": gamma_index,
                        "beta_index": beta_index,
                        "epsilon_index": epsilon_index,
                        "gamma": gamma,
                        "beta": beta,
                        "epsilon": epsilon,
                        "margin": margin,
                        "margin_region": (
                            "m>0"
                            if margin > 0.0
                            else ("m<0" if margin < 0.0 else "m=0")
                        ),
                    }
                )
                index += 1
    if len(points) != experiment["expected_points"]:
        raise RuntimeError("expanded grid does not match expected_points")
    return points


def _discounted_transmission_occupancy(
    model: JPOModel,
    analysis: Any,
) -> float:
    total = 0.0
    gamma = model.config.gamma
    for record in analysis.belief_records:
        belief = np.asarray(record["belief"], dtype=float)
        prescription = np.asarray(record["prescription"], dtype=float)
        receiver_action = int(record["receiver_action"])
        tx_probability = float(
            (belief @ model.mdp.P[receiver_action]) @ prescription
        )
        total += gamma * float(record["discounted_occupancy"]) * tx_probability
    return total


def run_point(
    root: Path,
    experiment: dict[str, Any],
    point: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    point_dir = root / "runs" / point["point_id"]
    summary_path = point_dir / "summary.json"
    if summary_path.is_file() and not force:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("status") == "ok":
            return saved

    point_dir.mkdir(parents=True, exist_ok=True)
    _write_json(point_dir / "configuration.json", {**experiment, **point})
    try:
        family = create_cmab_random_reward_family(
            n_states=experiment["n_states"],
            n_actions=experiment["n_actions"],
            seed=experiment["mdp_seed"],
        )
        mdp = select_density(family, experiment["density"])
        model = JPOModel(
            mdp,
            JPOConfig(
                gamma=point["gamma"],
                beta=point["beta"],
                epsilon=point["epsilon"],
            ),
        )
        settings = experiment["solver"]
        jpo_dir = point_dir / "jpo"
        policy_path = jpo_dir / "policy.npz"
        training_path = jpo_dir / "training.json"
        reused_solver_checkpoint = (
            not force and policy_path.is_file() and training_path.is_file()
        )
        if reused_solver_checkpoint:
            policy = load_sarsop_policy(model, policy_path)
            diagnostics = json.loads(training_path.read_text(encoding="utf-8"))
        else:
            training = run_native_sarsop(
                model=model,
                output_directory=jpo_dir,
                config=NativeSARSOPConfig(
                    search_epsilon=settings["search_epsilon"],
                    precision=settings["precision"],
                    max_time=settings["max_time"],
                    max_steps=settings["max_steps"],
                    kappa=settings["kappa"],
                    delta=settings["delta"],
                    prune_threshold=settings["prune_threshold"],
                    initial_bound_residual=settings["initial_bound_residual"],
                    initial_bound_max_time=settings["initial_bound_max_time"],
                    initial_upper_bound=settings["initial_upper_bound"],
                    export_beliefs=settings["export_beliefs"],
                ),
            )
            policy = training.policy
            diagnostics = training.diagnostics

        analysis = analyze_jpo_policy(
            model,
            policy,
            PolicyAnalysisConfig(
                discounted_tail_tolerance=settings["analysis_tail_tolerance"],
                max_belief_nodes=settings["max_belief_nodes"],
            ),
        )
        violations = analysis.violations
        depths = [int(item["first_depth"]) for item in violations]
        bounds = diagnostics["bounds"]
        training_summary = diagnostics["training"]
        summary = {
            "status": "ok",
            **point,
            "mdp_type": experiment["mdp_type"],
            "mdp_seed": experiment["mdp_seed"],
            "initial_upper_bound": training_summary["initial_upper_bound"],
            "initialization_seconds": training_summary["initialization_seconds"],
            "initial_upper_residual": training_summary["initial_upper_residual"],
            "sarsop_lower_bound": bounds["lower_objective"],
            "sarsop_upper_bound": bounds["upper_objective"],
            "sarsop_gap": bounds["gap"],
            "root_gap": bounds["root_gap"],
            "solver_stop_reason": training_summary["stop_reason"],
            "solver_iterations": training_summary["iterations"],
            "solver_elapsed_seconds": training_summary["elapsed_seconds"],
            "sampled_belief_count": training_summary["belief_count"],
            "alpha_vector_count": training_summary["alpha_count"],
            "policy_evaluation_skipped": True,
            "reused_solver_checkpoint": reused_solver_checkpoint,
            "violation_count": len(violations),
            "violation_discounted_event_occupancy": sum(
                float(item["discounted_event_occupancy"]) for item in violations
            ),
            "violation_first_depth_min": None if not depths else min(depths),
            "violation_first_depth_max": None if not depths else max(depths),
            "discounted_transmission_occupancy": (
                _discounted_transmission_occupancy(model, analysis)
            ),
            "analysis_horizon": analysis.horizon,
            "analysis_total_belief_nodes": analysis.total_belief_nodes,
            "analysis_discounted_tail_bound": analysis.discounted_tail_bound,
        }
        _write_json(summary_path, summary)
        if not settings["export_beliefs"]:
            _write_json(
                jpo_dir / "native_output" / "discarded_belief_dump.json",
                {
                    "sampled_belief_count": summary["sampled_belief_count"],
                    "reason": "Belief dump disabled for the CMAB grid.",
                },
            )
        return summary
    except Exception as error:
        summary = {
            "status": "error",
            **point,
            "mdp_type": experiment["mdp_type"],
            "mdp_seed": experiment["mdp_seed"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, summary)
        return summary


def _write_merged(
    root: Path,
    experiment: dict[str, Any],
    points_by_id: dict[str, dict[str, Any]],
) -> None:
    _write_json(
        root / "results.json",
        {
            "experiment": experiment,
            "points": sorted(
                (_with_current_margin(point) for point in points_by_id.values()),
                key=lambda point: point["index"],
            ),
        },
    )


def _priority_key(point: dict[str, Any]) -> tuple[float, float]:
    return (-float(point["beta"]), float(point["epsilon"]))


def run_grid(
    root: Path,
    experiment: dict[str, Any],
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=experiment["default_workers"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "experiment.json", experiment)
    all_points = build_points(experiment)
    existing: dict[str, dict[str, Any]] = {}
    results_path = root / "results.json"
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        existing = {
            point["point_id"]: point for point in payload.get("points", [])
        }
    for point in all_points:
        summary_path = root / "runs" / point["point_id"] / "summary.json"
        if summary_path.is_file():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved.get("status") == "ok":
                existing[point["point_id"]] = saved

    pending = [
        point
        for point in all_points
        if args.force or existing.get(point["point_id"], {}).get("status") != "ok"
    ]
    pending.sort(key=_priority_key)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"completed={sum(item.get('status') == 'ok' for item in existing.values())} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    print(
        "next=" + ",".join(point["point_id"] for point in pending[:8]),
        flush=True,
    )
    if args.dry_run:
        _write_merged(root, experiment, existing)
        return
    if not pending:
        _write_merged(root, experiment, existing)
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
    ) as executor:
        future_to_point = {
            executor.submit(run_point, root, experiment, point, args.force): point
            for point in pending
        }
        finished = 0
        for future in as_completed(future_to_point):
            point = future_to_point[future]
            try:
                summary = future.result()
            except Exception as error:
                summary = {
                    "status": "error",
                    **point,
                    "mdp_type": experiment["mdp_type"],
                    "mdp_seed": experiment["mdp_seed"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            existing[point["point_id"]] = summary
            finished += 1
            _write_merged(root, experiment, existing)
            if summary["status"] == "ok":
                print(
                    f"[{finished}/{len(pending)}] {point['point_id']} "
                    f"b={point['beta']:.3f} e={point['epsilon']:.3f} "
                    f"violations={summary['violation_count']} "
                    f"tx={summary['discounted_transmission_occupancy']:.4g} "
                    f"gap={summary['sarsop_gap']:.4g} "
                    f"init={summary['initialization_seconds']:.2f}s "
                    f"stop={summary['solver_stop_reason']}",
                    flush=True,
                )
            else:
                print(
                    f"[{finished}/{len(pending)}] {point['point_id']} ERROR "
                    f"{summary['error_type']}: {summary['error']}",
                    flush=True,
                )
