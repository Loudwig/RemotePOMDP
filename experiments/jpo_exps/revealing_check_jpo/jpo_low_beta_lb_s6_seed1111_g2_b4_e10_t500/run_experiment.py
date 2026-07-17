"""Run a resumable low-beta JPO grid with normal and restricted values."""

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
from jpo_sarsop import (
    NativeSARSOPConfig,
    load_sarsop_policy,
    run_native_sarsop,
)
from mdp import create_effcom_control_family, select_density


ROOT = Path(__file__).resolve().parent

EXPERIMENT = {
    "name": "jpo_low_beta_lb_s6_seed1111_g2_b4_e10_t500",
    "n_states": 6,
    "n_actions": 2,
    "density": 0.5,
    "reward_decay": 10.0,
    "mdp_seed_sampling_seed": 20260717,
    "mdp_seed": 1111,
    "gammas": [0.9, 0.99],
    "betas": [0.0, 0.05, 0.1, 0.15],
    "epsilons": np.linspace(0.01, 0.1, 10).tolist(),
    "expected_points": 80,
    "solver": {
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


def _point_id(gamma_index: int, beta_index: int, epsilon_index: int) -> str:
    return f"g{gamma_index}_b{beta_index:02d}_e{epsilon_index:02d}"


def build_points() -> list[dict[str, Any]]:
    points = []
    index = 0
    for gamma_index, gamma in enumerate(EXPERIMENT["gammas"]):
        for beta_index, beta in enumerate(EXPERIMENT["betas"]):
            for epsilon_index, epsilon in enumerate(EXPERIMENT["epsilons"]):
                margin = beta - gamma * epsilon * (1.0 - epsilon) / (1.0 - gamma)
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
    if len(points) != EXPERIMENT["expected_points"]:
        raise RuntimeError("expanded grid does not match expected_points")
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
    force: bool,
) -> tuple[Any, dict[str, Any], bool]:
    policy_path = jpo_dir / "policy.npz"
    training_path = jpo_dir / "training.json"
    reused = not force and policy_path.is_file() and training_path.is_file()
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


def _evaluate_fixed_policy(
    model: JPOModel,
    policy: Any,
) -> dict[str, Any]:
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
    sarsop_upper: float,
) -> dict[str, Any]:
    if violation_count == 0:
        return {"status": "not_applicable_no_violations"}
    evaluated = _evaluate_fixed_policy(
        model, RestrictedJPOPolicy(model, policy)
    )
    if evaluated["lower_objective"] > sarsop_upper + 1e-7:
        raise RuntimeError(
            "restricted feasible-policy lower bound exceeds the unrestricted "
            "SARSOP upper bound"
        )
    evaluated["unrestricted_upper_minus_restricted_lower"] = (
        sarsop_upper - evaluated["lower_objective"]
    )
    return evaluated


def run_point(point: dict[str, Any], force: bool = False) -> dict[str, Any]:
    point_dir = ROOT / "runs" / point["point_id"]
    summary_path = point_dir / "summary.json"
    if summary_path.is_file() and not force:
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
            seed=EXPERIMENT["mdp_seed"],
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
        policy, diagnostics, reused = _load_or_train(
            model, point_dir / "jpo", force
        )
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
            bounds["upper_objective"],
        )
        _write_json(point_dir / "restricted_lower_bound.json", restricted)

        depths = [int(item["first_depth"]) for item in violations]
        summary = {
            "status": "ok",
            **point,
            "mdp_seed": EXPERIMENT["mdp_seed"],
            "sarsop_lower_bound": bounds["lower_objective"],
            "sarsop_upper_bound": bounds["upper_objective"],
            "sarsop_gap": bounds["gap"],
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
            "restricted_lower_bound_status": restricted["status"],
            "restricted_policy_lower_bound": restricted.get("lower_objective"),
            "restricted_policy_upper_bound": restricted.get("upper_objective"),
            "restricted_policy_interval_width": restricted.get(
                "tail_interval_width"
            ),
            "restricted_policy_horizon": restricted.get("horizon"),
            "restricted_policy_total_belief_nodes": restricted.get(
                "total_belief_nodes"
            ),
            "unrestricted_upper_minus_restricted_lower": restricted.get(
                "unrestricted_upper_minus_restricted_lower"
            ),
            "normal_minus_restricted_value": (
                None
                if restricted["status"] != "ok"
                else normal["objective_estimate"]
                - restricted["objective_estimate"]
            ),
            "normal_minus_restricted_lower": (
                None
                if restricted["status"] != "ok"
                else normal["lower_objective"]
                - restricted["upper_objective"]
            ),
            "normal_minus_restricted_upper": (
                None
                if restricted["status"] != "ok"
                else normal["upper_objective"]
                - restricted["lower_objective"]
            ),
        }
        _write_json(summary_path, summary)
        return summary
    except Exception as error:
        summary = {
            "status": "error",
            **point,
            "mdp_seed": EXPERIMENT["mdp_seed"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(summary_path, summary)
        return summary


def _write_merged(points_by_id: dict[str, dict[str, Any]]) -> None:
    _write_json(
        ROOT / "results.json",
        {
            "experiment": EXPERIMENT,
            "points": sorted(
                points_by_id.values(), key=lambda point: point["index"]
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=EXPERIMENT["default_workers"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(ROOT / "experiment.json", EXPERIMENT)
    all_points = build_points()
    existing: dict[str, dict[str, Any]] = {}
    results_path = ROOT / "results.json"
    if results_path.is_file():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        existing = {
            point["point_id"]: point for point in payload.get("points", [])
        }
    for point in all_points:
        summary_path = ROOT / "runs" / point["point_id"] / "summary.json"
        if summary_path.is_file():
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            if saved.get("status") == "ok":
                existing[point["point_id"]] = saved

    pending = [
        point
        for point in all_points
        if args.force or existing.get(point["point_id"], {}).get("status") != "ok"
    ]
    pending.sort(key=lambda point: (point["beta"], point["gamma"], point["epsilon"]))
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"completed={sum(item.get('status') == 'ok' for item in existing.values())} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if not pending:
        _write_merged(existing)
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        future_to_point = {
            executor.submit(run_point, point, args.force): point for point in pending
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
                    "mdp_seed": EXPERIMENT["mdp_seed"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            existing[point["point_id"]] = summary
            finished += 1
            _write_merged(existing)
            if summary["status"] == "ok":
                lower_bound = summary["restricted_policy_lower_bound"]
                lower_bound_text = "-" if lower_bound is None else f"{lower_bound:.4g}"
                print(
                    f"[{finished}/{len(pending)}] {point['point_id']} "
                    f"g={point['gamma']:.2f} b={point['beta']:.3f} "
                    f"e={point['epsilon']:.3f} violations={summary['violation_count']} "
                    f"tx={summary['discounted_transmission_occupancy']:.4g} "
                    f"value={summary['normal_policy_value']:.4g} "
                    f"restricted_lb={lower_bound_text} gap={summary['sarsop_gap']:.4g} "
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
