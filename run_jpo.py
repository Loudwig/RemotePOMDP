"""Train and analyze one infinite-horizon JPO policy with NativeSARSOP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jpo import run_jpo
from jpo_model import JPOConfig
from jpo_policy import PolicyAnalysisConfig, PolicyEvaluationConfig
from jpo_sarsop import NativeSARSOPConfig
from mdp import create_effcom_control_family, select_density


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--n-actions", type=int, default=2)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--reward-decay", type=float, default=10.0)
    parser.add_argument("--mdp-seed", type=int, default=1234)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--solver-precision", type=float, default=0.01)
    parser.add_argument("--solver-search-epsilon", type=float, default=0.01)
    parser.add_argument("--solver-max-time", type=float, default=300.0)
    parser.add_argument("--solver-max-steps", type=int, default=1_000_000)
    parser.add_argument("--solver-kappa", type=float, default=0.5)
    parser.add_argument("--solver-delta", type=float, default=0.0001)
    parser.add_argument("--solver-prune-threshold", type=float, default=0.10)
    parser.add_argument("--solver-initial-bound-residual", type=float, default=1e-8)
    parser.add_argument("--solver-initial-bound-max-time", type=float, default=30.0)
    parser.add_argument(
        "--solver-initial-upper-bound",
        choices=("fully_observable", "fib"),
        default="fully_observable",
    )
    parser.add_argument("--disable-solver-binning", action="store_true")
    parser.add_argument("--analysis-tail-tolerance", type=float, default=1e-8)
    parser.add_argument("--evaluation-tail-tolerance", type=float, default=1e-8)
    parser.add_argument("--simulation-episodes", type=int, default=0)
    parser.add_argument("--simulation-horizon", type=int, default=500)
    parser.add_argument("--simulation-seed", type=int, default=1234)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--output", default="jpo_run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_directory = Path(args.output).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    arguments_path = output_directory / "cli_arguments.json"
    temporary_arguments_path = arguments_path.with_suffix(".json.tmp")
    temporary_arguments_path.write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_arguments_path.replace(arguments_path)
    family = create_effcom_control_family(
        n_states=args.n_states,
        n_actions=args.n_actions,
        reward_decay=args.reward_decay,
        seed=args.mdp_seed,
    )
    mdp = select_density(family, args.density)
    result = run_jpo(
        mdp=mdp,
        model_config=JPOConfig(
            gamma=args.gamma,
            beta=args.beta,
            epsilon=args.epsilon,
        ),
        solver_config=NativeSARSOPConfig(
            search_epsilon=args.solver_search_epsilon,
            precision=args.solver_precision,
            kappa=args.solver_kappa,
            delta=args.solver_delta,
            max_time=args.solver_max_time,
            max_steps=args.solver_max_steps,
            prune_threshold=args.solver_prune_threshold,
            use_binning=not args.disable_solver_binning,
            initial_bound_residual=args.solver_initial_bound_residual,
            initial_bound_max_time=args.solver_initial_bound_max_time,
            initial_upper_bound=args.solver_initial_upper_bound,
        ),
        analysis_config=PolicyAnalysisConfig(
            discounted_tail_tolerance=args.analysis_tail_tolerance
        ),
        evaluation_config=PolicyEvaluationConfig(
            tail_interval_tolerance=args.evaluation_tail_tolerance
        ),
        output_directory=output_directory,
        julia_executable=args.julia,
        simulation_episodes=args.simulation_episodes,
        simulation_horizon=args.simulation_horizon,
        simulation_seed=args.simulation_seed,
    )
    print(
        json.dumps(
            {
                "output_directory": str(result.output_directory),
                "lower_objective": result.training.lower_objective,
                "upper_objective": result.training.upper_objective,
                "gap": result.training.gap,
                "violation_count": len(result.analysis.violations),
                "restricted_lower_objective": (
                    None
                    if result.restricted_evaluation is None
                    else result.restricted_evaluation.lower_objective
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
