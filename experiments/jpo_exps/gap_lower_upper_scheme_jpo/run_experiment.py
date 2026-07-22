"""Run a resumable multi-MDP-seed JPO epsilon sweep.

The experiment keeps sampling deterministic candidate MDP seeds until it has
the requested number of accepted seeds.  A seed is accepted only when every
epsilon run succeeds and its largest achieved SARSOP gap is at most the
configured quality threshold.
"""

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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from jpo_model import JPOConfig, JPOModel
from jpo_policy import (
    PolicyAnalysisConfig,
    PolicyEvaluationConfig,
    RestrictedJPOPolicy,
    analyze_jpo_policy,
    evaluate_jpo_policy,
)
from jpo_sarsop import NativeSARSOPConfig, load_sarsop_policy, run_native_sarsop
from mdp import create_effcom_control_family, select_density


ROOT = Path(__file__).resolve().parent

EXPERIMENT = {
    "name": "gap_lower_upper_scheme_jpo",
    "n_states": 6,
    "n_actions": 2,
    "density": 0.5,
    "reward_decay": 10.0,
    "gamma": 0.9,
    "beta": 0.05,
    "epsilons": np.linspace(0.01, 0.1, 10).tolist(),
    "target_valid_mdp_seeds": 30,
    "mdp_seed_sampling_seed": 20260720,
    "max_candidate_mdp_seeds": 200,
    "solver_gap_acceptance_threshold": 0.5,
    "solver": {
        "search_epsilon": 0.01,
        "precision": 0.01,
        "max_time": 600.0,
        "max_steps": 1_000_000,
        "kappa": 0.5,
        "delta": 0.0001,
        "prune_threshold": 0.10,
        "initial_bound_residual": 1e-8,
        "initial_bound_max_time": 30.0,
        "initial_upper_bound": "fully_observable",
        "export_beliefs": False,
    },
    "analysis": {
        "discounted_tail_tolerance": 1e-8,
        "max_belief_nodes": 2_000_000,
    },
    "policy_evaluation": {
        "tail_interval_tolerance": 1e-3,
        "max_belief_nodes": 10_000_000,
    },
    "default_workers": 4,
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


def candidate_mdp_seeds() -> list[int]:
    rng = np.random.default_rng(EXPERIMENT["mdp_seed_sampling_seed"])
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < EXPERIMENT["max_candidate_mdp_seeds"]:
        seed = int(rng.integers(0, 2**31 - 1))
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
    return seeds


CANDIDATE_MDP_SEEDS = candidate_mdp_seeds()


def _point_id(mdp_seed: int, epsilon_index: int) -> str:
    return f"seed_{mdp_seed}_e{epsilon_index:02d}"


def build_seed_points(candidate_index: int) -> list[dict[str, Any]]:
    mdp_seed = CANDIDATE_MDP_SEEDS[candidate_index]
    points = []
    for epsilon_index, epsilon in enumerate(EXPERIMENT["epsilons"]):
        margin = (
            EXPERIMENT["beta"]
            - EXPERIMENT["gamma"] * epsilon * (1.0 - epsilon)
            / (1.0 - EXPERIMENT["gamma"])
        )
        points.append(
            {
                "index": candidate_index * len(EXPERIMENT["epsilons"])
                + epsilon_index,
                "point_id": _point_id(mdp_seed, epsilon_index),
                "candidate_index": candidate_index,
                "mdp_seed": mdp_seed,
                "epsilon_index": epsilon_index,
                "gamma": EXPERIMENT["gamma"],
                "beta": EXPERIMENT["beta"],
                "epsilon": epsilon,
                "margin": margin,
                "margin_region": (
                    "m>0" if margin > 0.0 else ("m<0" if margin < 0.0 else "m=0")
                ),
            }
        )
    return points


def _discounted_transmission_occupancy(model: JPOModel, analysis: Any) -> float:
    total = 0.0
    gamma = model.config.gamma
    for record in analysis.belief_records:
        belief = np.asarray(record["belief"], dtype=float)
        prescription = np.asarray(record["prescription"], dtype=float)
        receiver_action = int(record["receiver_action"])
        transmission_probability = float(
            (belief @ model.mdp.P[receiver_action]) @ prescription
        )
        total += (
            gamma
            * float(record["discounted_occupancy"])
            * transmission_probability
        )
    return total


def _load_or_train(
    model: JPOModel,
    jpo_dir: Path,
) -> tuple[Any, dict[str, Any], bool]:
    policy_path = jpo_dir / "policy.npz"
    training_path = jpo_dir / "training.json"
    reused = policy_path.is_file() and training_path.is_file()
    if reused:
        policy = load_sarsop_policy(model, policy_path)
        diagnostics = json.loads(training_path.read_text(encoding="utf-8"))
        return policy, diagnostics, True

    settings = EXPERIMENT["solver"]
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
    return training.policy, training.diagnostics, False


def _evaluate_fixed_policy(model: JPOModel, policy: Any) -> dict[str, Any]:
    settings = EXPERIMENT["policy_evaluation"]
    evaluation = evaluate_jpo_policy(
        model,
        policy,
        PolicyEvaluationConfig(
            tail_interval_tolerance=settings["tail_interval_tolerance"],
            max_belief_nodes=settings["max_belief_nodes"],
        ),
    )
    return {
        "status": "ok",
        "method": evaluation.method,
        "lower_objective": evaluation.lower_objective,
        "upper_objective": evaluation.upper_objective,
        "objective_estimate": evaluation.objective_estimate,
        "lower_values": evaluation.lower_values,
        "upper_values": evaluation.upper_values,
        "horizon": evaluation.horizon,
        "tail_interval_width": evaluation.tail_interval_width,
        "total_belief_nodes": evaluation.total_belief_nodes,
    }


def _restricted_policy_evaluation(
    model: JPOModel,
    policy: Any,
    violation_count: int,
    normal: dict[str, Any],
    sarsop_upper: float,
) -> dict[str, Any]:
    if violation_count == 0:
        return {
            **normal,
            "status": "same_as_normal_no_violations",
            "unrestricted_upper_minus_restricted_lower": (
                sarsop_upper - normal["lower_objective"]
            ),
        }
    evaluated = _evaluate_fixed_policy(model, RestrictedJPOPolicy(model, policy))
    if evaluated["lower_objective"] > sarsop_upper + 1e-7:
        raise RuntimeError(
            "restricted feasible-policy lower bound exceeds the unrestricted "
            "SARSOP upper bound"
        )
    evaluated["unrestricted_upper_minus_restricted_lower"] = (
        sarsop_upper - evaluated["lower_objective"]
    )
    return evaluated


def run_point(point: dict[str, Any]) -> dict[str, Any]:
    point_dir = ROOT / "runs" / point["point_id"]
    summary_path = point_dir / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("status") == "ok":
            return saved

    point_dir.mkdir(parents=True, exist_ok=True)
    _write_json(point_dir / "configuration.json", {**EXPERIMENT, **point})
    try:
        family = create_effcom_control_family(
            n_states=EXPERIMENT["n_states"],
            n_actions=EXPERIMENT["n_actions"],
            reward_decay=EXPERIMENT["reward_decay"],
            seed=point["mdp_seed"],
        )
        mdp = select_density(family, EXPERIMENT["density"])
        model = JPOModel(
            mdp,
            JPOConfig(
                gamma=point["gamma"],
                beta=point["beta"],
                epsilon=point["epsilon"],
            ),
        )
        policy, diagnostics, reused = _load_or_train(model, point_dir / "jpo")

        analysis_settings = EXPERIMENT["analysis"]
        analysis = analyze_jpo_policy(
            model,
            policy,
            PolicyAnalysisConfig(
                discounted_tail_tolerance=analysis_settings[
                    "discounted_tail_tolerance"
                ],
                max_belief_nodes=analysis_settings["max_belief_nodes"],
            ),
        )
        violations = analysis.violations
        transmission_occupancy = _discounted_transmission_occupancy(
            model, analysis
        )
        bounds = diagnostics["bounds"]
        training = diagnostics["training"]
        _write_json(
            point_dir / "violations.json",
            {
                "violation_count": len(violations),
                "violations": violations,
                "analysis_horizon": analysis.horizon,
                "analysis_total_belief_nodes": analysis.total_belief_nodes,
                "analysis_discounted_tail_bound": analysis.discounted_tail_bound,
                "discounted_transmission_occupancy": transmission_occupancy,
            },
        )

        normal = _evaluate_fixed_policy(model, policy)
        if bounds["lower_objective"] > normal["upper_objective"] + 1e-7:
            raise RuntimeError(
                "normal policy evaluation lies below its alpha-vector lower bound"
            )
        if normal["lower_objective"] > bounds["upper_objective"] + 1e-7:
            raise RuntimeError(
                "normal policy evaluation exceeds the unrestricted SARSOP upper bound"
            )
        _write_json(point_dir / "normal_policy_value.json", normal)

        restricted = _restricted_policy_evaluation(
            model,
            policy,
            len(violations),
            normal,
            bounds["upper_objective"],
        )
        _write_json(point_dir / "restricted_policy_value.json", restricted)

        no_policy_change = len(violations) == 0
        loss_value = (
            0.0
            if no_policy_change
            else normal["objective_estimate"] - restricted["objective_estimate"]
        )
        loss_lower = (
            0.0
            if no_policy_change
            else normal["lower_objective"] - restricted["upper_objective"]
        )
        loss_upper = (
            0.0
            if no_policy_change
            else normal["upper_objective"] - restricted["lower_objective"]
        )
        depths = [int(item["first_depth"]) for item in violations]
        summary = {
            "status": "ok",
            **point,
            "sarsop_lower_bound": bounds["lower_objective"],
            "sarsop_upper_bound": bounds["upper_objective"],
            "sarsop_gap": bounds["gap"],
            "sarsop_gap_within_acceptance_threshold": (
                bounds["gap"]
                <= EXPERIMENT["solver_gap_acceptance_threshold"]
            ),
            "root_gap": bounds["root_gap"],
            "solver_stop_reason": training["stop_reason"],
            "solver_iterations": training["iterations"],
            "solver_elapsed_seconds": training["elapsed_seconds"],
            "sampled_belief_count": training["belief_count"],
            "alpha_vector_count": training["alpha_count"],
            "reused_solver_checkpoint": reused,
            "violation_count": len(violations),
            "violation_discounted_event_occupancy": sum(
                float(item["discounted_event_occupancy"]) for item in violations
            ),
            "violation_first_depth_min": None if not depths else min(depths),
            "violation_first_depth_max": None if not depths else max(depths),
            "discounted_transmission_occupancy": transmission_occupancy,
            "analysis_horizon": analysis.horizon,
            "analysis_total_belief_nodes": analysis.total_belief_nodes,
            "analysis_discounted_tail_bound": analysis.discounted_tail_bound,
            "normal_policy_lower_bound": normal["lower_objective"],
            "normal_policy_upper_bound": normal["upper_objective"],
            "normal_policy_value": normal["objective_estimate"],
            "normal_policy_interval_width": normal["tail_interval_width"],
            "normal_policy_horizon": normal["horizon"],
            "normal_policy_total_belief_nodes": normal["total_belief_nodes"],
            "restricted_policy_status": restricted["status"],
            "restricted_policy_lower_bound": restricted["lower_objective"],
            "restricted_policy_upper_bound": restricted["upper_objective"],
            "restricted_policy_value": restricted["objective_estimate"],
            "restricted_policy_interval_width": restricted[
                "tail_interval_width"
            ],
            "restricted_policy_horizon": restricted["horizon"],
            "restricted_policy_total_belief_nodes": restricted[
                "total_belief_nodes"
            ],
            "normal_minus_restricted_value": loss_value,
            "normal_minus_restricted_lower": loss_lower,
            "normal_minus_restricted_upper": loss_upper,
        }
        _write_json(summary_path, summary)
        return summary
    except Exception as error:
        summary = {
            "status": "error",
            **point,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, summary)
        return summary


def _load_existing() -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    for path in (ROOT / "runs").glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        point_id = summary.get("point_id")
        if isinstance(point_id, str):
            existing[point_id] = summary
    return existing


def _seed_status(
    candidate_index: int,
    points_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    points = build_seed_points(candidate_index)
    records = [points_by_id.get(point["point_id"]) for point in points]
    completed = [record for record in records if record is not None]
    errors = [record for record in completed if record.get("status") != "ok"]
    excessive = [
        record
        for record in completed
        if record.get("status") == "ok"
        and float(record["sarsop_gap"])
        > EXPERIMENT["solver_gap_acceptance_threshold"]
    ]
    if errors or excessive:
        status = "rejected"
        if errors:
            reason = "run_error"
        else:
            reason = "solver_gap_above_threshold"
    elif len(completed) == len(points):
        status = "accepted"
        reason = None
    else:
        status = "incomplete"
        reason = None

    ok_records = [record for record in completed if record.get("status") == "ok"]
    gaps = [float(record["sarsop_gap"]) for record in ok_records]
    return {
        "candidate_index": candidate_index,
        "mdp_seed": CANDIDATE_MDP_SEEDS[candidate_index],
        "status": status,
        "rejection_reason": reason,
        "completed_points": len(completed),
        "successful_points": len(ok_records),
        "error_points": len(errors),
        "excessive_gap_points": len(excessive),
        "max_sarsop_gap": None if not gaps else max(gaps),
        "mean_sarsop_gap": None if not gaps else float(np.mean(gaps)),
    }


def _seed_statuses(
    points_by_id: dict[str, dict[str, Any]],
    frontier_count: int,
) -> list[dict[str, Any]]:
    return [
        _seed_status(candidate_index, points_by_id)
        for candidate_index in range(frontier_count)
    ]


def _write_merged(
    points_by_id: dict[str, dict[str, Any]],
    frontier_count: int,
) -> None:
    statuses = _seed_statuses(points_by_id, frontier_count)
    accepted = [item["mdp_seed"] for item in statuses if item["status"] == "accepted"]
    rejected = [item["mdp_seed"] for item in statuses if item["status"] == "rejected"]
    _write_json(
        ROOT / "results.json",
        {
            "experiment": EXPERIMENT,
            "candidate_mdp_seeds": CANDIDATE_MDP_SEEDS,
            "accepted_mdp_seeds": accepted,
            "rejected_mdp_seeds": rejected,
            "seed_statuses": statuses,
            "points": sorted(
                points_by_id.values(), key=lambda point: point["index"]
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=EXPERIMENT["default_workers"])
    parser.add_argument(
        "--target-valid-seeds",
        type=int,
        default=EXPERIMENT["target_valid_mdp_seeds"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run at most this many new epsilon points (useful for a pilot)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.target_valid_seeds < 1:
        raise ValueError("target-valid-seeds must be positive")
    if args.target_valid_seeds > EXPERIMENT["max_candidate_mdp_seeds"]:
        raise ValueError("target-valid-seeds exceeds max_candidate_mdp_seeds")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(ROOT / "experiment.json", EXPERIMENT)
    existing = _load_existing()
    executed = 0
    frontier_count = args.target_valid_seeds

    while True:
        statuses = _seed_statuses(existing, frontier_count)
        accepted_count = sum(item["status"] == "accepted" for item in statuses)
        rejected_count = sum(item["status"] == "rejected" for item in statuses)
        required_frontier = args.target_valid_seeds + rejected_count
        if required_frontier > EXPERIMENT["max_candidate_mdp_seeds"]:
            raise RuntimeError(
                "exhausted candidate MDP seeds before reaching the target"
            )
        if required_frontier > frontier_count:
            frontier_count = required_frontier
            statuses = _seed_statuses(existing, frontier_count)

        if accepted_count >= args.target_valid_seeds:
            _write_merged(existing, frontier_count)
            print(
                f"target reached: accepted={accepted_count} "
                f"rejected={rejected_count} candidates={frontier_count}",
                flush=True,
            )
            return

        pending: list[dict[str, Any]] = []
        for status in statuses:
            if status["status"] != "incomplete":
                continue
            for point in build_seed_points(status["candidate_index"]):
                if point["point_id"] not in existing:
                    pending.append(point)
        pending.sort(key=lambda point: (point["epsilon_index"], point["candidate_index"]))

        if args.limit is not None:
            remaining = args.limit - executed
            if remaining <= 0:
                _write_merged(existing, frontier_count)
                print("point limit reached; experiment remains resumable", flush=True)
                return
            pending = pending[:remaining]
        if not pending:
            _write_merged(existing, frontier_count)
            raise RuntimeError("no pending points but the valid-seed target is unmet")

        print(
            f"accepted={accepted_count}/{args.target_valid_seeds} "
            f"rejected={rejected_count} pending_batch={len(pending)} "
            f"workers={args.workers}",
            flush=True,
        )
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            future_to_point = {
                executor.submit(run_point, point): point for point in pending
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
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                existing[point["point_id"]] = summary
                finished += 1
                executed += 1
                _write_merged(existing, frontier_count)
                if summary["status"] == "ok":
                    print(
                        f"[{finished}/{len(pending)}] {point['point_id']} "
                        f"eps={point['epsilon']:.2f} "
                        f"loss={summary['normal_minus_restricted_value']:.6g} "
                        f"solver_gap={summary['sarsop_gap']:.6g} "
                        f"violations={summary['violation_count']} "
                        f"stop={summary['solver_stop_reason']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{finished}/{len(pending)}] {point['point_id']} ERROR "
                        f"{summary['error_type']}: {summary['error']}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
