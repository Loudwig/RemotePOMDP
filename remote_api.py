"""Tabular API for remote control over a costly packet-erasure channel."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from mdp import FiniteMDP, initial_distribution, validate_initial_distribution


class ConvergenceError(RuntimeError):
    """Raised when a Bellman iteration reaches its configured iteration cap."""


@dataclass(frozen=True)
class SolverConfig:
    gamma: float
    beta: float
    epsilon: float
    delta_max: int = 20
    vi_tol: float = 1e-10
    rx_accept_tol: float = 1e-9
    api_tol: float = 1e-9
    ne_tol: float = 1e-8
    margin_tol: float = 1e-10
    tie_tol: float = 1e-12
    max_vi_iterations: int = 100_000
    max_rx_iterations: int = 100
    max_api_iterations: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if self.beta < 0.0 or not np.isfinite(self.beta):
            raise ValueError("beta must be finite and nonnegative")
        if self.epsilon == 0.0:
            raise NotImplementedError(
                "epsilon == 0 is intentionally unsupported; see TODO.md"
            )
        if not 0.0 < self.epsilon <= 1.0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must lie in (0, 1]")
        if int(self.delta_max) != self.delta_max or self.delta_max < 0:
            raise ValueError("delta_max must be a nonnegative integer")
        for name in (
            "vi_tol",
            "rx_accept_tol",
            "api_tol",
            "ne_tol",
            "margin_tol",
            "tie_tol",
        ):
            value = getattr(self, name)
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.vi_tol == 0.0:
            raise ValueError("vi_tol must be strictly positive")
        for name in (
            "max_vi_iterations",
            "max_rx_iterations",
            "max_api_iterations",
        ):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def overflow_value(self) -> float:
        return -self.beta / (1.0 - self.gamma)

    @property
    def theorem_margin(self) -> float:
        return self.beta - (
            self.gamma * self.epsilon * (1.0 - self.epsilon) / (1.0 - self.gamma)
        )

    @property
    def margin_region(self) -> str:
        margin = self.theorem_margin
        if margin > self.margin_tol:
            return "guaranteed-positive"
        if margin < -self.margin_tol:
            return "outside-guarantee"
        return "boundary"


@dataclass
class BellmanResult:
    values: np.ndarray
    iterations: int
    residual: float
    converged: bool


@dataclass
class TxBestResponseResult:
    policy: np.ndarray
    bellman: BellmanResult
    q_values: np.ndarray


@dataclass
class BeliefResult:
    beliefs: np.ndarray
    valid: np.ndarray


@dataclass
class RxCandidateResult:
    policy: np.ndarray
    bellman: BellmanResult
    q_values: np.ndarray


@dataclass
class RxImprovementResult:
    policy: np.ndarray
    objective: float
    attempts: int
    accepted_steps: int
    converged: bool
    history: list[dict[str, object]] = field(default_factory=list)


@dataclass
class RevealingResult:
    is_revealing: bool
    violations: list[dict[str, int]]
    deferred_boundary_layer_violations: list[dict[str, int]]
    boundary_transmissions: list[dict[str, int]]
    boundary_states: list[tuple[int, int, int]]
    reachable_states: list[tuple[int, int, int]]
    statistics: dict[str, int | float]


@dataclass
class APIResult:
    pi_tx: np.ndarray
    pi_rx: np.ndarray
    two_way: BellmanResult
    lower_bound: BellmanResult | None
    objective: float
    lower_bound_objective: float | None
    tx_regret: float
    rx_restricted_regret: float
    approximate_restricted_ne: bool
    revealing: RevealingResult
    api_iterations: int
    converged: bool
    history: list[dict[str, object]]
    violation_history: list[dict[str, object]]
    diagnostics: dict[str, object]


def _validate_tx_policy(
    policy: np.ndarray, n_states: int, delta_max: int
) -> np.ndarray:
    out = np.asarray(policy)
    expected = (n_states, delta_max + 1, n_states)
    if out.shape != expected:
        raise ValueError(f"pi_tx must have shape {expected}, got {out.shape}")
    if not np.issubdtype(out.dtype, np.integer):
        raise ValueError("pi_tx must contain integer actions")
    if np.any((out != 0) & (out != 1)):
        raise ValueError("pi_tx actions must be zero or one")
    return out.astype(np.int64, copy=False)


def _validate_rx_policy(
    policy: np.ndarray, n_states: int, n_actions: int, delta_max: int
) -> np.ndarray:
    out = np.asarray(policy)
    expected = (delta_max + 1, n_states)
    if out.shape != expected:
        raise ValueError(f"pi_rx must have shape {expected}, got {out.shape}")
    if not np.issubdtype(out.dtype, np.integer):
        raise ValueError("pi_rx must contain integer actions")
    if np.any(out < 0) or np.any(out >= n_actions):
        raise ValueError(f"pi_rx actions must lie in [0, {n_actions})")
    return out.astype(np.int64, copy=False)


def _stable_argmax(values: np.ndarray, previous: int, tie_tol: float) -> int:
    best = float(np.max(values))
    if float(values[previous]) >= best - tie_tol:
        return int(previous)
    candidates = np.flatnonzero(values >= best - tie_tol)
    return int(candidates[0])


def _branch_value(
    mdp: FiniteMDP,
    expected_rewards: np.ndarray,
    gamma: float,
    action: int,
    state: int,
    continuation: np.ndarray,
) -> float:
    return float(
        expected_rewards[action, state]
        + gamma * np.dot(mdp.P[action, state], continuation)
    )


def _iterate_operator(
    operator: Callable[[np.ndarray], np.ndarray],
    shape: tuple[int, ...],
    config: SolverConfig,
) -> BellmanResult:
    values = np.zeros(shape, dtype=float)
    residual = np.inf
    for iteration in range(1, config.max_vi_iterations + 1):
        updated = operator(values)
        step_residual = float(np.max(np.abs(updated - values)))
        values = updated
        if step_residual <= config.vi_tol:
            residual = float(np.max(np.abs(operator(values) - values)))
            if residual <= config.vi_tol:
                return BellmanResult(values, iteration, residual, True)
    residual = float(np.max(np.abs(operator(values) - values)))
    return BellmanResult(values, config.max_vi_iterations, residual, False)


def _require_converged(result: BellmanResult, name: str) -> None:
    if not result.converged:
        raise ConvergenceError(
            f"{name} did not converge after {result.iterations} iterations; "
            f"Bellman residual={result.residual:.3e}, vi_tol must be relaxed or "
            "the iteration cap increased"
        )


def objective_from_values(values: np.ndarray, mu0: np.ndarray) -> float:
    states = np.arange(mu0.size)
    return float(np.dot(mu0, values[states, 0, states]))


def evaluate_policy(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_tx: np.ndarray,
    pi_rx: np.ndarray,
    mu0: np.ndarray | None = None,
    mode: Literal["two_way", "lower_bound"] = "two_way",
) -> BellmanResult:
    """Evaluate a deterministic joint policy by infinite-horizon Bellman iteration."""

    if mode not in ("two_way", "lower_bound"):
        raise ValueError("mode must be 'two_way' or 'lower_bound'")
    n_states = mdp.n_states
    delta_max = config.delta_max
    tx = _validate_tx_policy(pi_tx, n_states, delta_max)
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_max)
    if mu0 is not None:
        validate_initial_distribution(mu0, n_states)
    expected_rewards = mdp.expected_rewards
    overflow = config.overflow_value

    def operator(values: np.ndarray) -> np.ndarray:
        updated = np.empty_like(values)
        for state in range(n_states):
            for age in range(delta_max + 1):
                for last_received in range(n_states):
                    communication = int(tx[state, age, last_received])
                    if age < delta_max:
                        no_reception_action = int(rx[age + 1, last_received])
                        no_reception = _branch_value(
                            mdp,
                            expected_rewards,
                            config.gamma,
                            no_reception_action,
                            state,
                            values[:, age + 1, last_received],
                        )
                        if communication == 0:
                            updated[state, age, last_received] = no_reception
                            continue
                        success_action = int(rx[0, state])
                        if mode == "lower_bound" and success_action == no_reception_action:
                            updated[state, age, last_received] = -config.beta + no_reception
                            continue
                        success = _branch_value(
                            mdp,
                            expected_rewards,
                            config.gamma,
                            success_action,
                            state,
                            values[:, 0, state],
                        )
                        updated[state, age, last_received] = (
                            -config.beta
                            + (1.0 - config.epsilon) * success
                            + config.epsilon * no_reception
                        )
                    else:
                        boundary_action = int(rx[delta_max, last_received])
                        boundary = float(
                            expected_rewards[boundary_action, state]
                            + config.gamma * overflow
                        )
                        if communication == 0:
                            updated[state, age, last_received] = boundary
                            continue
                        success_action = int(rx[0, state])
                        if mode == "lower_bound" and success_action == boundary_action:
                            updated[state, age, last_received] = -config.beta + boundary
                            continue
                        success = _branch_value(
                            mdp,
                            expected_rewards,
                            config.gamma,
                            success_action,
                            state,
                            values[:, 0, state],
                        )
                        updated[state, age, last_received] = (
                            -config.beta
                            + (1.0 - config.epsilon) * success
                            + config.epsilon * boundary
                        )
        return updated

    result = _iterate_operator(
        operator, (n_states, delta_max + 1, n_states), config
    )
    _require_converged(result, f"{mode} policy evaluation")
    return result


def tx_best_response(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_rx: np.ndarray,
    previous_pi_tx: np.ndarray | None = None,
) -> TxBestResponseResult:
    """Compute the exact tabular Tx best response by value iteration."""

    n_states = mdp.n_states
    delta_max = config.delta_max
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_max)
    if previous_pi_tx is None:
        previous = np.zeros((n_states, delta_max + 1, n_states), dtype=np.int64)
    else:
        previous = _validate_tx_policy(previous_pi_tx, n_states, delta_max)
    expected_rewards = mdp.expected_rewards
    overflow = config.overflow_value

    def q_values(values: np.ndarray) -> np.ndarray:
        q = np.empty((n_states, delta_max + 1, n_states, 2), dtype=float)
        for state in range(n_states):
            success_action = int(rx[0, state])
            success = _branch_value(
                mdp,
                expected_rewards,
                config.gamma,
                success_action,
                state,
                values[:, 0, state],
            )
            for age in range(delta_max + 1):
                for last_received in range(n_states):
                    if age < delta_max:
                        no_reception_action = int(rx[age + 1, last_received])
                        no_reception = _branch_value(
                            mdp,
                            expected_rewards,
                            config.gamma,
                            no_reception_action,
                            state,
                            values[:, age + 1, last_received],
                        )
                    else:
                        boundary_action = int(rx[delta_max, last_received])
                        no_reception = float(
                            expected_rewards[boundary_action, state]
                            + config.gamma * overflow
                        )
                    q[state, age, last_received, 0] = no_reception
                    q[state, age, last_received, 1] = (
                        -config.beta
                        + (1.0 - config.epsilon) * success
                        + config.epsilon * no_reception
                    )
        return q

    result = _iterate_operator(
        lambda values: np.max(q_values(values), axis=3),
        (n_states, delta_max + 1, n_states),
        config,
    )
    _require_converged(result, "Tx value iteration")
    final_q = q_values(result.values)
    policy = np.empty_like(previous)
    for state in range(n_states):
        for age in range(delta_max + 1):
            for last_received in range(n_states):
                policy[state, age, last_received] = _stable_argmax(
                    final_q[state, age, last_received],
                    int(previous[state, age, last_received]),
                    config.tie_tol,
                )
    return TxBestResponseResult(policy, result, final_q)


def compute_rx_beliefs(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_tx: np.ndarray,
    pi_rx: np.ndarray,
) -> BeliefResult:
    """Recompute Rx beliefs induced by the current deterministic policies."""

    n_states = mdp.n_states
    delta_max = config.delta_max
    tx = _validate_tx_policy(pi_tx, n_states, delta_max)
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_max)
    beliefs = np.zeros((delta_max + 1, n_states, n_states), dtype=float)
    valid = np.zeros((delta_max + 1, n_states), dtype=bool)
    for last_received in range(n_states):
        beliefs[0, last_received, last_received] = 1.0
        valid[0, last_received] = True
        for age in range(delta_max):
            if not valid[age, last_received]:
                break
            action = int(rx[age, last_received])
            prediction = beliefs[age, last_received] @ mdp.P[action]
            likelihood = 1.0 - (1.0 - config.epsilon) * tx[:, age, last_received]
            unnormalized = prediction * likelihood
            denominator = float(unnormalized.sum())
            if denominator <= 0.0:
                valid[age + 1, last_received] = False
                continue
            beliefs[age + 1, last_received] = unnormalized / denominator
            valid[age + 1, last_received] = True
    return BeliefResult(beliefs, valid)


def rx_greedy_candidate(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_tx: np.ndarray,
    current_pi_rx: np.ndarray,
    belief_result: BeliefResult,
) -> RxCandidateResult:
    """Solve the aggregated Rx Bellman equation with beliefs held fixed."""

    n_states = mdp.n_states
    delta_max = config.delta_max
    tx = _validate_tx_policy(pi_tx, n_states, delta_max)
    current = _validate_rx_policy(
        current_pi_rx, n_states, mdp.n_actions, delta_max
    )
    if belief_result.beliefs.shape != (delta_max + 1, n_states, n_states):
        raise ValueError("belief array has the wrong shape")
    if belief_result.valid.shape != (delta_max + 1, n_states):
        raise ValueError("belief validity mask has the wrong shape")
    expected_rewards = mdp.expected_rewards
    overflow = config.overflow_value

    def q_values(values: np.ndarray) -> np.ndarray:
        q = np.full(
            (delta_max + 1, n_states, mdp.n_actions), -np.inf, dtype=float
        )
        for age in range(delta_max + 1):
            for last_received in range(n_states):
                if not belief_result.valid[age, last_received]:
                    old_action = int(current[age, last_received])
                    q[age, last_received, old_action] = values[age, last_received]
                    continue
                belief = belief_result.beliefs[age, last_received]
                if age < delta_max:
                    no_reception_value = float(values[age + 1, last_received])
                    communication = tx[:, age, last_received]
                    continuation = np.where(
                        communication == 1,
                        -config.beta
                        + (1.0 - config.epsilon) * values[0, :]
                        + config.epsilon * no_reception_value,
                        no_reception_value,
                    )
                else:
                    communication = tx[:, delta_max, last_received]
                    continuation = np.where(
                        communication == 1,
                        -config.beta
                        + (1.0 - config.epsilon) * values[0, :]
                        + config.epsilon * overflow,
                        overflow,
                    )
                for action in range(mdp.n_actions):
                    state_values = expected_rewards[action] + config.gamma * (
                        mdp.P[action] @ continuation
                    )
                    q[age, last_received, action] = float(
                        np.dot(belief, state_values)
                    )
        return q

    result = _iterate_operator(
        lambda values: np.max(q_values(values), axis=2),
        (delta_max + 1, n_states),
        config,
    )
    _require_converged(result, "fixed-belief Rx value iteration")
    final_q = q_values(result.values)
    candidate = np.array(current, copy=True)
    for age in range(delta_max + 1):
        for last_received in range(n_states):
            if belief_result.valid[age, last_received]:
                candidate[age, last_received] = _stable_argmax(
                    final_q[age, last_received],
                    int(current[age, last_received]),
                    config.tie_tol,
                )
    return RxCandidateResult(candidate, result, final_q)


def rx_restricted_best_response(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_tx: np.ndarray,
    initial_pi_rx: np.ndarray,
    mu0: np.ndarray | None = None,
) -> RxImprovementResult:
    """Run self-consistent fixed-belief Rx improvements with exact acceptance."""

    if mu0 is None:
        distribution = initial_distribution(mdp.n_states)
    else:
        distribution = validate_initial_distribution(mu0, mdp.n_states)
    tx = _validate_tx_policy(pi_tx, mdp.n_states, config.delta_max)
    current = np.array(
        _validate_rx_policy(
            initial_pi_rx, mdp.n_states, mdp.n_actions, config.delta_max
        ),
        copy=True,
    )
    current_evaluation = evaluate_policy(mdp, config, tx, current, distribution)
    current_objective = objective_from_values(current_evaluation.values, distribution)
    history: list[dict[str, object]] = []
    accepted_steps = 0

    for attempt in range(1, config.max_rx_iterations + 1):
        beliefs = compute_rx_beliefs(mdp, config, tx, current)
        candidate_result = rx_greedy_candidate(
            mdp, config, tx, current, beliefs
        )
        candidate = candidate_result.policy
        if np.array_equal(candidate, current):
            history.append(
                {
                    "attempt": attempt,
                    "accepted": False,
                    "reason": "policy_unchanged",
                    "objective": current_objective,
                    "beliefs_valid": int(np.sum(beliefs.valid)),
                }
            )
            return RxImprovementResult(
                current,
                current_objective,
                attempt,
                accepted_steps,
                True,
                history,
            )

        # Required by the specification even though exact joint evaluation below
        # uses the full augmented state rather than these aggregated beliefs.
        candidate_beliefs = compute_rx_beliefs(mdp, config, tx, candidate)
        candidate_evaluation = evaluate_policy(
            mdp, config, tx, candidate, distribution
        )
        candidate_objective = objective_from_values(
            candidate_evaluation.values, distribution
        )
        accepted = (
            candidate_objective
            >= current_objective + config.rx_accept_tol
        )
        history.append(
            {
                "attempt": attempt,
                "accepted": accepted,
                "old_objective": current_objective,
                "candidate_objective": candidate_objective,
                "improvement": candidate_objective - current_objective,
                "beliefs_valid": int(np.sum(candidate_beliefs.valid)),
                "rx_vi_iterations": candidate_result.bellman.iterations,
                "rx_vi_residual": candidate_result.bellman.residual,
            }
        )
        if not accepted:
            return RxImprovementResult(
                current,
                current_objective,
                attempt,
                accepted_steps,
                True,
                history,
            )
        current = np.array(candidate, copy=True)
        current_objective = candidate_objective
        accepted_steps += 1

    return RxImprovementResult(
        current,
        current_objective,
        config.max_rx_iterations,
        accepted_steps,
        False,
        history,
    )


def fully_observed_mdp_policy(
    mdp: FiniteMDP, config: SolverConfig
) -> tuple[np.ndarray, BellmanResult]:
    """Solve the fully observed physical MDP for deterministic Rx initialization."""

    expected_rewards = mdp.expected_rewards

    def q_values(values: np.ndarray) -> np.ndarray:
        q = np.empty((mdp.n_states, mdp.n_actions), dtype=float)
        for action in range(mdp.n_actions):
            q[:, action] = expected_rewards[action] + config.gamma * (
                mdp.P[action] @ values
            )
        return q

    result = _iterate_operator(
        lambda values: np.max(q_values(values), axis=1),
        (mdp.n_states,),
        config,
    )
    _require_converged(result, "fully observed MDP value iteration")
    final_q = q_values(result.values)
    policy = np.argmax(final_q, axis=1).astype(np.int64)
    return policy, result


def initialize_policies(
    mdp: FiniteMDP,
    config: SolverConfig,
    seed: int = 1234,
    tx_mode: Literal["never", "random"] = "never",
    rx_mode: Literal["fully_observed", "random"] = "fully_observed",
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic initial policies."""

    rng = np.random.default_rng(seed)
    tx_shape = (mdp.n_states, config.delta_max + 1, mdp.n_states)
    rx_shape = (config.delta_max + 1, mdp.n_states)
    if tx_mode == "never":
        pi_tx = np.zeros(tx_shape, dtype=np.int64)
    elif tx_mode == "random":
        pi_tx = rng.integers(0, 2, size=tx_shape, dtype=np.int64)
    else:
        raise ValueError("tx_mode must be 'never' or 'random'")

    if rx_mode == "fully_observed":
        state_policy, _ = fully_observed_mdp_policy(mdp, config)
        pi_rx = np.broadcast_to(state_policy, rx_shape).copy()
    elif rx_mode == "random":
        pi_rx = rng.integers(0, mdp.n_actions, size=rx_shape, dtype=np.int64)
    else:
        raise ValueError("rx_mode must be 'fully_observed' or 'random'")
    return pi_tx, pi_rx


def check_revealing(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_tx: np.ndarray,
    pi_rx: np.ndarray,
    mu0: np.ndarray | None = None,
    all_states: bool = False,
) -> RevealingResult:
    """Check revealing only on augmented states reachable in the two-way model."""

    n_states = mdp.n_states
    delta_max = config.delta_max
    tx = _validate_tx_policy(pi_tx, n_states, delta_max)
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_max)
    distribution = (
        initial_distribution(n_states)
        if mu0 is None
        else validate_initial_distribution(mu0, n_states)
    )

    if all_states:
        reachable = {
            (state, age, last_received)
            for state in range(n_states)
            for age in range(delta_max + 1)
            for last_received in range(n_states)
        }
    else:
        queue = deque(
            (state, 0, state)
            for state in range(n_states)
            if distribution[state] > 0.0
        )
        reachable: set[tuple[int, int, int]] = set(queue)

        def enqueue_physical_successors(
            state: int, action: int, next_age: int, next_last_received: int
        ) -> None:
            for next_state, probability in enumerate(mdp.P[action, state]):
                if probability > 0.0:
                    target = (next_state, next_age, next_last_received)
                    if target not in reachable:
                        reachable.add(target)
                        queue.append(target)

        while queue:
            state, age, last_received = queue.popleft()
            communication = int(tx[state, age, last_received])
            if age == delta_max:
                if communication == 1 and (1.0 - config.epsilon) > 0.0:
                    success_action = int(rx[0, state])
                    enqueue_physical_successors(state, success_action, 0, state)
                # No reception at the boundary terminates at overflow.
                continue
            if communication == 0:
                action = int(rx[age + 1, last_received])
                enqueue_physical_successors(
                    state, action, age + 1, last_received
                )
                continue
            if (1.0 - config.epsilon) > 0.0:
                success_action = int(rx[0, state])
                enqueue_physical_successors(state, success_action, 0, state)
            if config.epsilon > 0.0:
                failure_action = int(rx[age + 1, last_received])
                enqueue_physical_successors(
                    state, failure_action, age + 1, last_received
                )

    violations: list[dict[str, int]] = []
    deferred_boundary_layer_violations: list[dict[str, int]] = []
    boundary_transmissions: list[dict[str, int]] = []
    boundary_states: list[tuple[int, int, int]] = []
    for state, age, last_received in sorted(reachable):
        communication = int(tx[state, age, last_received])
        if age == delta_max:
            boundary_states.append((state, age, last_received))
            if communication == 1:
                boundary_transmissions.append(
                    {
                        "state": state,
                        "age": age,
                        "last_received": last_received,
                        "tx_action": communication,
                        "success_action": int(rx[0, state]),
                        "boundary_action": int(rx[delta_max, last_received]),
                    }
                )
            continue
        if communication == 1:
            success_action = int(rx[0, state])
            no_reception_action = int(rx[age + 1, last_received])
            if success_action == no_reception_action:
                record = {
                    "state": state,
                    "age": age,
                    "last_received": last_received,
                    "tx_action": communication,
                    "success_action": success_action,
                    "no_reception_action": no_reception_action,
                }
                if age == delta_max - 1:
                    deferred_boundary_layer_violations.append(record)
                else:
                    violations.append(record)

    total_states = n_states * n_states * (delta_max + 1)
    statistics: dict[str, int | float] = {
        "reachable_count": len(reachable),
        "reachable_interior_count": len(reachable) - len(boundary_states),
        "reachable_boundary_count": len(boundary_states),
        "total_tabular_states": total_states,
        "reachable_fraction": len(reachable) / total_states,
        "initial_support_size": int(np.count_nonzero(distribution > 0.0)),
    }
    return RevealingResult(
        is_revealing=not violations,
        violations=violations,
        deferred_boundary_layer_violations=deferred_boundary_layer_violations,
        boundary_transmissions=boundary_transmissions,
        boundary_states=boundary_states,
        reachable_states=sorted(reachable),
        statistics=statistics,
    )


def run_api(
    mdp: FiniteMDP,
    config: SolverConfig,
    mu0: np.ndarray | None = None,
    seed: int = 1234,
    initial_pi_tx: np.ndarray | None = None,
    initial_pi_rx: np.ndarray | None = None,
    compute_lower_bound: bool = False,
) -> APIResult:
    """Run alternating person-by-person optimization and final diagnostics."""

    distribution = (
        initial_distribution(mdp.n_states)
        if mu0 is None
        else validate_initial_distribution(mu0, mdp.n_states)
    )
    if initial_pi_tx is None or initial_pi_rx is None:
        default_tx, default_rx = initialize_policies(mdp, config, seed=seed)
        if initial_pi_tx is None:
            initial_pi_tx = default_tx
        if initial_pi_rx is None:
            initial_pi_rx = default_rx
    tx = np.array(
        _validate_tx_policy(initial_pi_tx, mdp.n_states, config.delta_max),
        copy=True,
    )
    rx = np.array(
        _validate_rx_policy(
            initial_pi_rx, mdp.n_states, mdp.n_actions, config.delta_max
        ),
        copy=True,
    )
    logged_initial_tx = np.array(tx, copy=True)
    logged_initial_rx = np.array(rx, copy=True)
    current_evaluation = evaluate_policy(mdp, config, tx, rx, distribution)
    current_objective = objective_from_values(current_evaluation.values, distribution)
    initial_objective = current_objective
    history: list[dict[str, object]] = []
    initial_revealing = check_revealing(mdp, config, tx, rx, distribution)
    violation_history: list[dict[str, object]] = [
        {
            "api_iteration": 0,
            "revealing_violation_count": len(initial_revealing.violations),
            "deferred_boundary_layer_violation_count": len(
                initial_revealing.deferred_boundary_layer_violations
            ),
            "boundary_transmission_count": len(
                initial_revealing.boundary_transmissions
            ),
            "reachable_count": initial_revealing.statistics["reachable_count"],
        }
    ]
    converged = False

    for api_iteration in range(1, config.max_api_iterations + 1):
        old_tx = np.array(tx, copy=True)
        old_rx = np.array(rx, copy=True)
        tx_result = tx_best_response(mdp, config, rx, previous_pi_tx=tx)
        tx = tx_result.policy
        rx_result = rx_restricted_best_response(
            mdp, config, tx, rx, distribution
        )
        rx = rx_result.policy
        new_evaluation = evaluate_policy(mdp, config, tx, rx, distribution)
        new_objective = objective_from_values(new_evaluation.values, distribution)
        iteration_revealing = check_revealing(
            mdp, config, tx, rx, distribution
        )
        tx_changed = not np.array_equal(tx, old_tx)
        rx_changed = not np.array_equal(rx, old_rx)
        improvement = new_objective - current_objective
        history.append(
            {
                "api_iteration": api_iteration,
                "objective": new_objective,
                "improvement": improvement,
                "tx_changed": tx_changed,
                "rx_changed": rx_changed,
                "tx_vi_iterations": tx_result.bellman.iterations,
                "tx_vi_residual": tx_result.bellman.residual,
                "rx_attempts": rx_result.attempts,
                "rx_accepted_steps": rx_result.accepted_steps,
                "rx_converged": rx_result.converged,
                "revealing_violation_count": len(
                    iteration_revealing.violations
                ),
                "deferred_boundary_layer_violation_count": len(
                    iteration_revealing.deferred_boundary_layer_violations
                ),
                "boundary_transmission_count": len(
                    iteration_revealing.boundary_transmissions
                ),
            }
        )
        violation_history.append(
            {
                "api_iteration": api_iteration,
                "revealing_violation_count": len(
                    iteration_revealing.violations
                ),
                "deferred_boundary_layer_violation_count": len(
                    iteration_revealing.deferred_boundary_layer_violations
                ),
                "boundary_transmission_count": len(
                    iteration_revealing.boundary_transmissions
                ),
                "reachable_count": iteration_revealing.statistics[
                    "reachable_count"
                ],
            }
        )
        current_evaluation = new_evaluation
        current_objective = new_objective
        if (not tx_changed and not rx_changed) or abs(improvement) <= config.api_tol:
            converged = True
            break

    api_iterations = len(history)
    revealing = check_revealing(mdp, config, tx, rx, distribution)
    lower_bound_result: BellmanResult | None = None
    lower_bound_objective: float | None = None
    if compute_lower_bound or not revealing.is_revealing:
        lower_bound_result = evaluate_policy(
            mdp, config, tx, rx, distribution, mode="lower_bound"
        )
        lower_bound_objective = objective_from_values(
            lower_bound_result.values, distribution
        )

    final_tx_br = tx_best_response(mdp, config, rx, previous_pi_tx=tx)
    final_tx_br_evaluation = evaluate_policy(
        mdp, config, final_tx_br.policy, rx, distribution
    )
    final_tx_br_objective = objective_from_values(
        final_tx_br_evaluation.values, distribution
    )
    tx_regret = final_tx_br_objective - current_objective

    final_rx_br = rx_restricted_best_response(
        mdp, config, tx, rx, distribution
    )
    rx_regret = final_rx_br.objective - current_objective
    approximate_ne = max(tx_regret, rx_regret) <= config.ne_tol

    diagnostics: dict[str, object] = {
        "n_states": mdp.n_states,
        "n_actions": mdp.n_actions,
        "density": mdp.density,
        "gamma": config.gamma,
        "beta": config.beta,
        "epsilon": config.epsilon,
        "delta_max": config.delta_max,
        "v_overflow": config.overflow_value,
        "mu0": distribution.tolist(),
        "seed": seed,
        "initial_pi_tx": logged_initial_tx.tolist(),
        "initial_pi_rx": logged_initial_rx.tolist(),
        "final_pi_tx": tx.tolist(),
        "final_pi_rx": rx.tolist(),
        "initial_objective": initial_objective,
        "final_j_mu0": current_objective,
        "final_two_way_value": current_objective,
        "final_lower_bound_one_way_value": lower_bound_objective,
        "two_way_bellman_residual": current_evaluation.residual,
        "two_way_bellman_iterations": current_evaluation.iterations,
        "tx_regret": tx_regret,
        "rx_restricted_regret": rx_regret,
        "rx_regret_scope": "implemented restricted fixed-belief improvement",
        "approximate_restricted_ne": approximate_ne,
        "ne_tol": config.ne_tol,
        "revealing": revealing.is_revealing,
        "revealing_violation_count": len(revealing.violations),
        "revealing_violations": revealing.violations,
        "deferred_boundary_layer_violation_count": len(
            revealing.deferred_boundary_layer_violations
        ),
        "deferred_boundary_layer_violations": (
            revealing.deferred_boundary_layer_violations
        ),
        "boundary_transmission_count": len(revealing.boundary_transmissions),
        "boundary_transmissions": revealing.boundary_transmissions,
        "reachable_statistics": revealing.statistics,
        "theorem_margin": config.theorem_margin,
        "margin_region": config.margin_region,
        "tolerances": {
            "vi_tol": config.vi_tol,
            "rx_accept_tol": config.rx_accept_tol,
            "api_tol": config.api_tol,
            "ne_tol": config.ne_tol,
            "margin_tol": config.margin_tol,
            "tie_tol": config.tie_tol,
        },
        "api_iterations": api_iterations,
        "api_converged": converged,
        "api_history": history,
        "revealing_training_history": violation_history,
        "final_rx_regret_attempts": final_rx_br.attempts,
        "final_rx_regret_converged": final_rx_br.converged,
    }
    return APIResult(
        pi_tx=tx,
        pi_rx=rx,
        two_way=current_evaluation,
        lower_bound=lower_bound_result,
        objective=current_objective,
        lower_bound_objective=lower_bound_objective,
        tx_regret=tx_regret,
        rx_restricted_regret=rx_regret,
        approximate_restricted_ne=approximate_ne,
        revealing=revealing,
        api_iterations=api_iterations,
        converged=converged,
        history=history,
        violation_history=violation_history,
        diagnostics=diagnostics,
    )
