"""Run one reproducible API experiment on an EffCom-style control MDP."""

from __future__ import annotations

import argparse
import json

from mdp import create_effcom_control_family, initial_distribution, select_density
from remote_api import SolverConfig, run_api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-states", type=int, default=10)
    parser.add_argument("--n-actions", type=int, default=2)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--reward-decay", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta-max", type=int, default=20)
    parser.add_argument(
        "--initial-state",
        type=int,
        default=None,
        help="Use a point-mass mu0 for debugging; default is uniform.",
    )
    parser.add_argument("--vi-tol", type=float, default=1e-10)
    parser.add_argument("--rx-accept-tol", type=float, default=1e-9)
    parser.add_argument("--api-tol", type=float, default=1e-9)
    parser.add_argument("--ne-tol", type=float, default=1e-8)
    parser.add_argument("--margin-tol", type=float, default=1e-10)
    parser.add_argument("--tie-tol", type=float, default=1e-12)
    parser.add_argument("--max-vi-iterations", type=int, default=100_000)
    parser.add_argument("--max-rx-iterations", type=int, default=100)
    parser.add_argument("--max-api-iterations", type=int, default=100)
    parser.add_argument("--compute-lower-bound", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    family = create_effcom_control_family(
        n_states=args.n_states,
        n_actions=args.n_actions,
        reward_decay=args.reward_decay,
        seed=args.seed,
    )
    mdp = select_density(family, args.density)
    config = SolverConfig(
        gamma=args.gamma,
        beta=args.beta,
        epsilon=args.epsilon,
        delta_max=args.delta_max,
        vi_tol=args.vi_tol,
        rx_accept_tol=args.rx_accept_tol,
        api_tol=args.api_tol,
        ne_tol=args.ne_tol,
        margin_tol=args.margin_tol,
        tie_tol=args.tie_tol,
        max_vi_iterations=args.max_vi_iterations,
        max_rx_iterations=args.max_rx_iterations,
        max_api_iterations=args.max_api_iterations,
    )
    mu0 = initial_distribution(mdp.n_states, args.initial_state)
    result = run_api(
        mdp,
        config,
        mu0=mu0,
        seed=args.seed,
        compute_lower_bound=args.compute_lower_bound,
    )
    print(json.dumps(result.diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
