"""Three-stage JPO training, policy analysis, and lower-bound workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jpo_model import JPOConfig, JPOModel
from jpo_policy import (
    PolicyAnalysisConfig,
    PolicyAnalysisResult,
    PolicyEvaluationConfig,
    PolicyEvaluationResult,
    PolicySimulationResult,
    RestrictedJPOPolicy,
    analyze_jpo_policy,
    evaluate_jpo_policy,
    simulate_jpo_policy,
)
from jpo_sarsop import (
    NativeSARSOPConfig,
    SARSOPTrainingResult,
    run_native_sarsop,
)
from mdp import FiniteMDP


@dataclass
class JPOResult:
    model: JPOModel
    training: SARSOPTrainingResult
    analysis: PolicyAnalysisResult
    policy_evaluation: PolicyEvaluationResult
    restricted_policy: RestrictedJPOPolicy | None
    restricted_evaluation: PolicyEvaluationResult | None
    policy_simulation: PolicySimulationResult | None
    restricted_simulation: PolicySimulationResult | None
    output_directory: Path
    diagnostics: dict[str, Any]


def run_jpo(
    mdp: FiniteMDP,
    model_config: JPOConfig,
    solver_config: NativeSARSOPConfig | None = None,
    analysis_config: PolicyAnalysisConfig | None = None,
    evaluation_config: PolicyEvaluationConfig | None = None,
    output_directory: str | Path = "jpo_run",
    julia_executable: str = "julia",
    simulation_episodes: int = 0,
    simulation_horizon: int = 500,
    simulation_seed: int = 1234,
) -> JPOResult:
    """Run the required unrestricted-train, analyze, restrict pipeline."""

    if int(simulation_episodes) != simulation_episodes or simulation_episodes < 0:
        raise ValueError("simulation_episodes must be a nonnegative integer")
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = JPOModel(mdp, model_config)

    # Stage 1: the POMDP is trained without any revealing restriction.
    training = run_native_sarsop(
        model,
        solver_config,
        root,
        julia_executable=julia_executable,
    )

    # Stage 2: violations are diagnosed only after policy extraction.
    analysis = analyze_jpo_policy(model, training.policy, analysis_config)
    policy_evaluation = evaluate_jpo_policy(
        model, training.policy, evaluation_config
    )

    if training.lower_objective > policy_evaluation.upper_objective + 1e-7:
        raise RuntimeError(
            "the deterministic evaluation of the extracted policy lies below "
            "its alpha-vector lower bound"
        )
    if policy_evaluation.lower_objective > training.upper_objective + 1e-7:
        raise RuntimeError(
            "the deterministic policy evaluation exceeds the SARSOP upper bound"
        )

    # Stage 3: construct the API-style restricted policy iff violations exist.
    restricted_policy: RestrictedJPOPolicy | None = None
    restricted_evaluation: PolicyEvaluationResult | None = None
    if analysis.violations:
        restricted_policy = RestrictedJPOPolicy(model, training.policy)
        restricted_evaluation = evaluate_jpo_policy(
            model, restricted_policy, evaluation_config
        )
        if restricted_evaluation.lower_objective > training.upper_objective + 1e-7:
            raise RuntimeError(
                "restricted feasible-policy lower bound exceeds the SARSOP upper bound"
            )

    policy_simulation: PolicySimulationResult | None = None
    restricted_simulation: PolicySimulationResult | None = None
    if simulation_episodes:
        policy_simulation = simulate_jpo_policy(
            model,
            training.policy,
            simulation_episodes,
            simulation_horizon,
            simulation_seed,
        )
        if restricted_policy is not None:
            restricted_simulation = simulate_jpo_policy(
                model,
                restricted_policy,
                simulation_episodes,
                simulation_horizon,
                simulation_seed,
            )

    diagnostics: dict[str, Any] = {
        "stages": {
            "1_unrestricted_training": training.diagnostics,
            "2_policy_analysis": analysis.diagnostics(),
            "3_restricted_lower_bound": (
                None
                if restricted_evaluation is None
                else restricted_evaluation.diagnostics()
            ),
        },
        "unrestricted_policy_evaluation": policy_evaluation.diagnostics(),
        "optional_simulation": (
            None if policy_simulation is None else policy_simulation.diagnostics()
        ),
        "optional_restricted_simulation": (
            None
            if restricted_simulation is None
            else restricted_simulation.diagnostics()
        ),
        "reported_bounds": {
            "sarsop_lower_objective": training.lower_objective,
            "sarsop_upper_objective": training.upper_objective,
            "sarsop_gap": training.gap,
            "restricted_policy_certified_lower": (
                None
                if restricted_evaluation is None
                else restricted_evaluation.lower_objective
            ),
            "restricted_policy_value_interval": (
                None
                if restricted_evaluation is None
                else [
                    restricted_evaluation.lower_objective,
                    restricted_evaluation.upper_objective,
                ]
            ),
        },
    }
    _write_json(root / "result.json", diagnostics)
    return JPOResult(
        model=model,
        training=training,
        analysis=analysis,
        policy_evaluation=policy_evaluation,
        restricted_policy=restricted_policy,
        restricted_evaluation=restricted_evaluation,
        policy_simulation=policy_simulation,
        restricted_simulation=restricted_simulation,
        output_directory=root,
        diagnostics=diagnostics,
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
