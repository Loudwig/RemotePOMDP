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
    delta_train: int = 20
    delta_check: int = 10
    boundary_model: Literal["tail", "legacy_overflow"] = "tail"
    boundary_tx_mode: Literal["free", "force_transmit"] = "force_transmit"
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
        if int(self.delta_train) != self.delta_train or self.delta_train < 1:
            raise ValueError("delta_train must be a positive integer")
        if int(self.delta_check) != self.delta_check or self.delta_check < 0:
            raise ValueError("delta_check must be a nonnegative integer")
        if self.delta_check >= self.delta_train:
            raise ValueError("delta_check must be strictly smaller than delta_train")
        if self.boundary_model not in ("tail", "legacy_overflow"):
            raise ValueError(
                "boundary_model must be 'tail' or 'legacy_overflow'"
            )
        if self.boundary_tx_mode not in ("free", "force_transmit"):
            raise ValueError(
                "boundary_tx_mode must be 'free' or 'force_transmit'"
            )
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
    def delta_max(self) -> int:
        """Deprecated read-only alias for code that reports the table size."""

        return self.delta_train

    @property
    def effective_boundary_tx_mode(self) -> Literal["free", "force_transmit"]:
        """Legacy overflow reproduces the old unconstrained boundary policy."""

        if self.boundary_model == "legacy_overflow":
            return "free"
        return self.boundary_tx_mode

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
    tail_values: np.ndarray | None = None


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
    core_violations: list[dict[str, int | float]]
    buffer_violations: list[dict[str, int | float]]
    boundary_adjacent_violations: list[dict[str, int | float]]
    boundary_transmissions: list[dict[str, int | float]]
    boundary_states: list[tuple[int, int, int]]
    reachable_states: list[tuple[int, int, int]]
    reachable_tail_states: list[tuple[int, int]]
    statistics: dict[str, int | float]

    @property
    def violations(self) -> list[dict[str, int | float]]:
        """Backward-compatible name for the theorem-checking violations."""

        return self.core_violations

    @property
    def deferred_boundary_layer_violations(
        self,
    ) -> list[dict[str, int | float]]:
        """Backward-compatible name for the former delta_max-1 category."""

        return self.boundary_adjacent_violations


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


def _iterate_coupled_operator(
    operator: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    value_shape: tuple[int, ...],
    tail_shape: tuple[int, ...],
    config: SolverConfig,
) -> BellmanResult:
    """Iterate a discounted Bellman operator jointly over table and tail values."""

    values = np.zeros(value_shape, dtype=float)
    tail_values = np.zeros(tail_shape, dtype=float)
    residual = np.inf
    for iteration in range(1, config.max_vi_iterations + 1):
        updated, updated_tail = operator(values, tail_values)
        step_residual = max(
            float(np.max(np.abs(updated - values))),
            float(np.max(np.abs(updated_tail - tail_values))),
        )
        values = updated
        tail_values = updated_tail
        if step_residual <= config.vi_tol:
            checked, checked_tail = operator(values, tail_values)
            residual = max(
                float(np.max(np.abs(checked - values))),
                float(np.max(np.abs(checked_tail - tail_values))),
            )
            if residual <= config.vi_tol:
                return BellmanResult(
                    values, iteration, residual, True, tail_values
                )
    checked, checked_tail = operator(values, tail_values)
    residual = max(
        float(np.max(np.abs(checked - values))),
        float(np.max(np.abs(checked_tail - tail_values))),
    )
    return BellmanResult(
        values,
        config.max_vi_iterations,
        residual,
        False,
        tail_values,
    )


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
    delta_train = config.delta_train
    tx = _validate_tx_policy(pi_tx, n_states, delta_train)
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_train)
    if mu0 is not None:
        validate_initial_distribution(mu0, n_states)
    expected_rewards = mdp.expected_rewards

    if config.boundary_model == "legacy_overflow":
        overflow = config.overflow_value

        def legacy_operator(values: np.ndarray) -> np.ndarray:
            updated = np.empty_like(values)
            for state in range(n_states):
                for age in range(delta_train + 1):
                    for last_received in range(n_states):
                        communication = int(tx[state, age, last_received])
                        if age < delta_train:
                            no_reception_action = int(
                                rx[age + 1, last_received]
                            )
                            no_reception = _branch_value(
                                mdp,
                                expected_rewards,
                                config.gamma,
                                no_reception_action,
                                state,
                                values[:, age + 1, last_received],
                            )
                        else:
                            no_reception_action = int(
                                rx[delta_train, last_received]
                            )
                            no_reception = float(
                                expected_rewards[no_reception_action, state]
                                + config.gamma * overflow
                            )
                        if communication == 0:
                            updated[state, age, last_received] = no_reception
                            continue
                        success_action = int(rx[0, state])
                        if (
                            mode == "lower_bound"
                            and success_action == no_reception_action
                        ):
                            updated[state, age, last_received] = (
                                -config.beta + no_reception
                            )
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
            return updated

        result = _iterate_operator(
            legacy_operator,
            (n_states, delta_train + 1, n_states),
            config,
        )
    else:

        def tail_operator(
            values: np.ndarray, tail_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            updated = np.empty_like(values)
            updated_tail = np.empty_like(tail_values)

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
                for last_received in range(n_states):
                    boundary_action = int(rx[delta_train, last_received])
                    tail_failure = _branch_value(
                        mdp,
                        expected_rewards,
                        config.gamma,
                        boundary_action,
                        state,
                        tail_values[:, last_received],
                    )
                    if (
                        mode == "lower_bound"
                        and success_action == boundary_action
                    ):
                        updated_tail[state, last_received] = (
                            -config.beta + tail_failure
                        )
                    else:
                        updated_tail[state, last_received] = (
                            -config.beta
                            + (1.0 - config.epsilon) * success
                            + config.epsilon * tail_failure
                        )

                for age in range(delta_train + 1):
                    for last_received in range(n_states):
                        communication = int(tx[state, age, last_received])
                        if age < delta_train:
                            no_reception_action = int(
                                rx[age + 1, last_received]
                            )
                            no_reception = _branch_value(
                                mdp,
                                expected_rewards,
                                config.gamma,
                                no_reception_action,
                                state,
                                values[:, age + 1, last_received],
                            )
                        else:
                            no_reception_action = int(
                                rx[delta_train, last_received]
                            )
                            no_reception = _branch_value(
                                mdp,
                                expected_rewards,
                                config.gamma,
                                no_reception_action,
                                state,
                                tail_values[:, last_received],
                            )
                        if communication == 0:
                            updated[state, age, last_received] = no_reception
                            continue
                        success_action = int(rx[0, state])
                        if (
                            mode == "lower_bound"
                            and success_action == no_reception_action
                        ):
                            updated[state, age, last_received] = (
                                -config.beta + no_reception
                            )
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
            return updated, updated_tail

        result = _iterate_coupled_operator(
            tail_operator,
            (n_states, delta_train + 1, n_states),
            (n_states, n_states),
            config,
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
    delta_train = config.delta_train
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_train)
    if previous_pi_tx is None:
        previous = np.zeros(
            (n_states, delta_train + 1, n_states), dtype=np.int64
        )
    else:
        previous = _validate_tx_policy(previous_pi_tx, n_states, delta_train)
    expected_rewards = mdp.expected_rewards

    if config.boundary_model == "legacy_overflow":
        overflow = config.overflow_value

        def legacy_q_values(values: np.ndarray) -> np.ndarray:
            q = np.empty(
                (n_states, delta_train + 1, n_states, 2), dtype=float
            )
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
                for age in range(delta_train + 1):
                    for last_received in range(n_states):
                        if age < delta_train:
                            no_reception_action = int(
                                rx[age + 1, last_received]
                            )
                            no_reception = _branch_value(
                                mdp,
                                expected_rewards,
                                config.gamma,
                                no_reception_action,
                                state,
                                values[:, age + 1, last_received],
                            )
                        else:
                            boundary_action = int(
                                rx[delta_train, last_received]
                            )
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
            lambda values: np.max(legacy_q_values(values), axis=3),
            (n_states, delta_train + 1, n_states),
            config,
        )
        final_q = legacy_q_values(result.values)
    else:

        def tail_q_values(
            values: np.ndarray, tail_values: np.ndarray
        ) -> np.ndarray:
            q = np.empty(
                (n_states, delta_train + 1, n_states, 2), dtype=float
            )
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
                for age in range(delta_train + 1):
                    for last_received in range(n_states):
                        if age < delta_train:
                            no_reception_action = int(
                                rx[age + 1, last_received]
                            )
                            no_reception = _branch_value(
                                mdp,
                                expected_rewards,
                                config.gamma,
                                no_reception_action,
                                state,
                                values[:, age + 1, last_received],
                            )
                        else:
                            boundary_action = int(
                                rx[delta_train, last_received]
                            )
                            no_reception = _branch_value(
                                mdp,
                                expected_rewards,
                                config.gamma,
                                boundary_action,
                                state,
                                tail_values[:, last_received],
                            )
                        q[state, age, last_received, 0] = no_reception
                        q[state, age, last_received, 1] = (
                            -config.beta
                            + (1.0 - config.epsilon) * success
                            + config.epsilon * no_reception
                        )
            return q

        def tail_operator(
            values: np.ndarray, tail_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            q = tail_q_values(values, tail_values)
            updated = np.max(q, axis=3)
            if config.effective_boundary_tx_mode == "force_transmit":
                updated[:, delta_train, :] = q[:, delta_train, :, 1]

            updated_tail = np.empty_like(tail_values)
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
                for last_received in range(n_states):
                    boundary_action = int(rx[delta_train, last_received])
                    failure = _branch_value(
                        mdp,
                        expected_rewards,
                        config.gamma,
                        boundary_action,
                        state,
                        tail_values[:, last_received],
                    )
                    updated_tail[state, last_received] = (
                        -config.beta
                        + (1.0 - config.epsilon) * success
                        + config.epsilon * failure
                    )
            return updated, updated_tail

        result = _iterate_coupled_operator(
            tail_operator,
            (n_states, delta_train + 1, n_states),
            (n_states, n_states),
            config,
        )
        assert result.tail_values is not None
        final_q = tail_q_values(result.values, result.tail_values)

    _require_converged(result, "Tx value iteration")
    policy = np.empty_like(previous)
    for state in range(n_states):
        for age in range(delta_train + 1):
            for last_received in range(n_states):
                if (
                    age == delta_train
                    and config.effective_boundary_tx_mode == "force_transmit"
                ):
                    policy[state, age, last_received] = 1
                else:
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
    delta_train = config.delta_train
    tx = _validate_tx_policy(pi_tx, n_states, delta_train)
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_train)
    beliefs = np.zeros((delta_train + 1, n_states, n_states), dtype=float)
    valid = np.zeros((delta_train + 1, n_states), dtype=bool)
    for last_received in range(n_states):
        beliefs[0, last_received, last_received] = 1.0
        valid[0, last_received] = True
        for age in range(delta_train):
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
    delta_train = config.delta_train
    tx = _validate_tx_policy(pi_tx, n_states, delta_train)
    current = _validate_rx_policy(
        current_pi_rx, n_states, mdp.n_actions, delta_train
    )
    if belief_result.beliefs.shape != (delta_train + 1, n_states, n_states):
        raise ValueError("belief array has the wrong shape")
    if belief_result.valid.shape != (delta_train + 1, n_states):
        raise ValueError("belief validity mask has the wrong shape")
    expected_rewards = mdp.expected_rewards

    def base_q_values(
        values: np.ndarray,
        boundary_continuation: Callable[[int], np.ndarray],
    ) -> np.ndarray:
        q = np.full(
            (delta_train + 1, n_states, mdp.n_actions),
            -np.inf,
            dtype=float,
        )
        for age in range(delta_train + 1):
            for last_received in range(n_states):
                if not belief_result.valid[age, last_received]:
                    old_action = int(current[age, last_received])
                    q[age, last_received, old_action] = values[
                        age, last_received
                    ]
                    continue
                belief = belief_result.beliefs[age, last_received]
                communication = tx[:, age, last_received]
                if age < delta_train:
                    no_reception = np.full(
                        n_states,
                        float(values[age + 1, last_received]),
                    )
                else:
                    no_reception = boundary_continuation(last_received)
                continuation = np.where(
                    communication == 1,
                    -config.beta
                    + (1.0 - config.epsilon) * values[0, :]
                    + config.epsilon * no_reception,
                    no_reception,
                )
                for action in range(mdp.n_actions):
                    state_values = expected_rewards[action] + config.gamma * (
                        mdp.P[action] @ continuation
                    )
                    q[age, last_received, action] = float(
                        np.dot(belief, state_values)
                    )
        return q

    if config.boundary_model == "legacy_overflow":
        overflow = config.overflow_value

        def legacy_q_values(values: np.ndarray) -> np.ndarray:
            return base_q_values(
                values,
                lambda _last_received: np.full(n_states, overflow),
            )

        result = _iterate_operator(
            lambda values: np.max(legacy_q_values(values), axis=2),
            (delta_train + 1, n_states),
            config,
        )
        final_q = legacy_q_values(result.values)
    else:

        def tail_q_values(
            values: np.ndarray, tail_values: np.ndarray
        ) -> np.ndarray:
            return base_q_values(
                values,
                lambda last_received: tail_values[:, last_received],
            )

        def tail_operator(
            values: np.ndarray, tail_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            q = tail_q_values(values, tail_values)
            updated = np.max(q, axis=2)
            updated_tail = np.empty_like(tail_values)
            for state in range(n_states):
                success_action = int(current[0, state])
                # Aggregated Rx indexing: successful transmission of current
                # physical state `state` resets memory to (0, state).  The
                # continuation is V_rx[0, state], never V_rx[0, next_state].
                success = float(
                    expected_rewards[success_action, state]
                    + config.gamma * values[0, state]
                )
                for last_received in range(n_states):
                    boundary_action = int(
                        current[delta_train, last_received]
                    )
                    failure = _branch_value(
                        mdp,
                        expected_rewards,
                        config.gamma,
                        boundary_action,
                        state,
                        tail_values[:, last_received],
                    )
                    updated_tail[state, last_received] = (
                        -config.beta
                        + (1.0 - config.epsilon) * success
                        + config.epsilon * failure
                    )
            return updated, updated_tail

        result = _iterate_coupled_operator(
            tail_operator,
            (delta_train + 1, n_states),
            (n_states, n_states),
            config,
        )
        assert result.tail_values is not None
        final_q = tail_q_values(result.values, result.tail_values)

    _require_converged(result, "fixed-belief Rx value iteration")
    candidate = np.array(current, copy=True)
    for age in range(delta_train + 1):
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
    tx = _validate_tx_policy(pi_tx, mdp.n_states, config.delta_train)
    current = np.array(
        _validate_rx_policy(
            initial_pi_rx,
            mdp.n_states,
            mdp.n_actions,
            config.delta_train,
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
    tx_shape = (mdp.n_states, config.delta_train + 1, mdp.n_states)
    rx_shape = (config.delta_train + 1, mdp.n_states)
    if tx_mode == "never":
        pi_tx = np.zeros(tx_shape, dtype=np.int64)
    elif tx_mode == "random":
        pi_tx = rng.integers(0, 2, size=tx_shape, dtype=np.int64)
    else:
        raise ValueError("tx_mode must be 'never' or 'random'")
    if config.effective_boundary_tx_mode == "force_transmit":
        pi_tx[:, config.delta_train, :] = 1

    if rx_mode == "fully_observed":
        state_policy, _ = fully_observed_mdp_policy(mdp, config)
        pi_rx = np.broadcast_to(state_policy, rx_shape).copy()
    elif rx_mode == "random":
        pi_rx = rng.integers(0, mdp.n_actions, size=rx_shape, dtype=np.int64)
    else:
        raise ValueError("rx_mode must be 'fully_observed' or 'random'")
    return pi_tx, pi_rx


def _propagate_policy_mass(
    mdp: FiniteMDP,
    config: SolverConfig,
    tx: np.ndarray,
    rx: np.ndarray,
    table_mass: np.ndarray,
    tail_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the transpose of the synchronized transition kernel to mass."""

    n_states = mdp.n_states
    delta_train = config.delta_train
    next_table = np.zeros_like(table_mass)
    next_tail = np.zeros_like(tail_mass)
    table_exit_mass = 0.0

    for state in range(n_states):
        for age in range(delta_train + 1):
            for last_received in range(n_states):
                mass = float(table_mass[state, age, last_received])
                if mass == 0.0:
                    continue
                communication = int(tx[state, age, last_received])
                if age < delta_train:
                    if communication == 0:
                        action = int(rx[age + 1, last_received])
                        next_table[:, age + 1, last_received] += (
                            mass * mdp.P[action, state]
                        )
                    else:
                        success_action = int(rx[0, state])
                        next_table[:, 0, state] += (
                            mass
                            * (1.0 - config.epsilon)
                            * mdp.P[success_action, state]
                        )
                        failure_action = int(rx[age + 1, last_received])
                        next_table[:, age + 1, last_received] += (
                            mass
                            * config.epsilon
                            * mdp.P[failure_action, state]
                        )
                    continue

                boundary_action = int(rx[delta_train, last_received])
                if communication == 1:
                    success_action = int(rx[0, state])
                    next_table[:, 0, state] += (
                        mass
                        * (1.0 - config.epsilon)
                        * mdp.P[success_action, state]
                    )
                exit_probability = (
                    1.0 if communication == 0 else config.epsilon
                )
                table_exit_mass += mass * exit_probability
                if config.boundary_model == "tail":
                    next_tail[:, last_received] += (
                        mass
                        * exit_probability
                        * mdp.P[boundary_action, state]
                    )

    if config.boundary_model == "tail":
        for state in range(n_states):
            success_action = int(rx[0, state])
            for last_received in range(n_states):
                mass = float(tail_mass[state, last_received])
                if mass == 0.0:
                    continue
                next_table[:, 0, state] += (
                    mass
                    * (1.0 - config.epsilon)
                    * mdp.P[success_action, state]
                )
                boundary_action = int(rx[delta_train, last_received])
                next_tail[:, last_received] += (
                    mass
                    * config.epsilon
                    * mdp.P[boundary_action, state]
                )
    return next_table, next_tail, table_exit_mass


def _discounted_policy_occupancy(
    mdp: FiniteMDP,
    config: SolverConfig,
    tx: np.ndarray,
    rx: np.ndarray,
    distribution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Compute discounted table and tail occupancy without a dense matrix."""

    n_states = mdp.n_states
    initial_table = np.zeros(
        (n_states, config.delta_train + 1, n_states), dtype=float
    )
    states = np.arange(n_states)
    initial_table[states, 0, states] = distribution
    initial_tail = np.zeros((n_states, n_states), dtype=float)
    table = np.zeros_like(initial_table)
    tail = np.zeros_like(initial_tail)
    residual = np.inf

    for iteration in range(1, config.max_vi_iterations + 1):
        propagated_table, propagated_tail, _ = _propagate_policy_mass(
            mdp, config, tx, rx, table, tail
        )
        updated_table = initial_table + config.gamma * propagated_table
        updated_tail = initial_tail + config.gamma * propagated_tail
        residual = max(
            float(np.max(np.abs(updated_table - table))),
            float(np.max(np.abs(updated_tail - tail))),
        )
        table = updated_table
        tail = updated_tail
        if residual <= config.vi_tol:
            break
    else:
        raise ConvergenceError(
            "discounted occupancy did not converge after "
            f"{config.max_vi_iterations} iterations; residual={residual:.3e}"
        )

    _, _, table_exit_mass = _propagate_policy_mass(
        mdp, config, tx, rx, table, tail
    )
    discounted_entry_flow = config.gamma * table_exit_mass
    return table, tail, discounted_entry_flow, iteration, residual


def check_revealing(
    mdp: FiniteMDP,
    config: SolverConfig,
    pi_tx: np.ndarray,
    pi_rx: np.ndarray,
    mu0: np.ndarray | None = None,
    all_states: bool = False,
    compute_occupancy: bool = True,
) -> RevealingResult:
    """Classify revealing violations on reachable finite-table states."""

    n_states = mdp.n_states
    delta_train = config.delta_train
    tx = _validate_tx_policy(pi_tx, n_states, delta_train)
    rx = _validate_rx_policy(pi_rx, n_states, mdp.n_actions, delta_train)
    distribution = (
        initial_distribution(n_states)
        if mu0 is None
        else validate_initial_distribution(mu0, n_states)
    )

    if all_states:
        reachable = {
            (state, age, last_received)
            for state in range(n_states)
            for age in range(delta_train + 1)
            for last_received in range(n_states)
        }
        reachable_tail = (
            {
                (state, last_received)
                for state in range(n_states)
                for last_received in range(n_states)
            }
            if config.boundary_model == "tail"
            else set()
        )
    else:
        table_queue = deque(
            (state, 0, state)
            for state in range(n_states)
            if distribution[state] > 0.0
        )
        tail_queue: deque[tuple[int, int]] = deque()
        reachable: set[tuple[int, int, int]] = set(table_queue)
        reachable_tail: set[tuple[int, int]] = set()

        def enqueue_table_successors(
            state: int, action: int, next_age: int, next_last_received: int
        ) -> None:
            for next_state, probability in enumerate(mdp.P[action, state]):
                if probability > 0.0:
                    target = (next_state, next_age, next_last_received)
                    if target not in reachable:
                        reachable.add(target)
                        table_queue.append(target)

        def enqueue_tail_successors(
            state: int, action: int, next_last_received: int
        ) -> None:
            for next_state, probability in enumerate(mdp.P[action, state]):
                if probability > 0.0:
                    target = (next_state, next_last_received)
                    if target not in reachable_tail:
                        reachable_tail.add(target)
                        tail_queue.append(target)

        while table_queue or tail_queue:
            while table_queue:
                state, age, last_received = table_queue.popleft()
                communication = int(tx[state, age, last_received])
                if age < delta_train:
                    if communication == 0:
                        action = int(rx[age + 1, last_received])
                        enqueue_table_successors(
                            state, action, age + 1, last_received
                        )
                    else:
                        if (1.0 - config.epsilon) > 0.0:
                            success_action = int(rx[0, state])
                            enqueue_table_successors(
                                state, success_action, 0, state
                            )
                        if config.epsilon > 0.0:
                            failure_action = int(rx[age + 1, last_received])
                            enqueue_table_successors(
                                state,
                                failure_action,
                                age + 1,
                                last_received,
                            )
                    continue

                if communication == 1 and (1.0 - config.epsilon) > 0.0:
                    success_action = int(rx[0, state])
                    enqueue_table_successors(state, success_action, 0, state)
                if config.boundary_model == "tail" and (
                    communication == 0 or config.epsilon > 0.0
                ):
                    boundary_action = int(rx[delta_train, last_received])
                    enqueue_tail_successors(
                        state, boundary_action, last_received
                    )

            while tail_queue:
                state, last_received = tail_queue.popleft()
                if (1.0 - config.epsilon) > 0.0:
                    success_action = int(rx[0, state])
                    enqueue_table_successors(state, success_action, 0, state)
                if config.epsilon > 0.0:
                    boundary_action = int(rx[delta_train, last_received])
                    enqueue_tail_successors(
                        state, boundary_action, last_received
                    )

    if compute_occupancy:
        (
            table_occupancy,
            tail_occupancy,
            discounted_entry_flow,
            occupancy_iterations,
            occupancy_residual,
        ) = _discounted_policy_occupancy(
            mdp, config, tx, rx, distribution
        )
    else:
        table_occupancy = np.zeros(
            (n_states, delta_train + 1, n_states), dtype=float
        )
        tail_occupancy = np.zeros((n_states, n_states), dtype=float)
        discounted_entry_flow = 0.0
        occupancy_iterations = 0
        occupancy_residual = 0.0

    core_violations: list[dict[str, int | float]] = []
    buffer_violations: list[dict[str, int | float]] = []
    boundary_adjacent_violations: list[dict[str, int | float]] = []
    boundary_transmissions: list[dict[str, int | float]] = []
    boundary_states: list[tuple[int, int, int]] = []
    for state, age, last_received in sorted(reachable):
        communication = int(tx[state, age, last_received])
        occupancy = float(table_occupancy[state, age, last_received])
        if age == delta_train:
            boundary_states.append((state, age, last_received))
            if communication == 1:
                boundary_transmissions.append(
                    {
                        "state": state,
                        "age": age,
                        "last_received": last_received,
                        "tx_action": communication,
                        "success_action": int(rx[0, state]),
                        "boundary_action": int(
                            rx[delta_train, last_received]
                        ),
                        "discounted_occupancy": occupancy,
                    }
                )
            continue
        if communication != 1:
            continue
        success_action = int(rx[0, state])
        no_reception_action = int(rx[age + 1, last_received])
        if success_action != no_reception_action:
            continue
        record: dict[str, int | float] = {
            "state": state,
            "age": age,
            "last_received": last_received,
            "tx_action": communication,
            "success_action": success_action,
            "no_reception_action": no_reception_action,
            "distance_to_boundary": delta_train - age,
            "discounted_occupancy": occupancy,
        }
        if age <= config.delta_check:
            core_violations.append(record)
        elif age == delta_train - 1:
            boundary_adjacent_violations.append(record)
        else:
            buffer_violations.append(record)

    total_states = n_states * n_states * (delta_train + 1)
    statistics: dict[str, int | float] = {
        "reachable_count": len(reachable),
        "reachable_interior_count": len(reachable) - len(boundary_states),
        "reachable_boundary_count": len(boundary_states),
        "reachable_tail_count": len(reachable_tail),
        "total_tabular_states": total_states,
        "total_tail_states": n_states * n_states,
        "reachable_fraction": len(reachable) / total_states,
        "initial_support_size": int(np.count_nonzero(distribution > 0.0)),
        "core_violation_count": len(core_violations),
        "buffer_violation_count": len(buffer_violations),
        "boundary_adjacent_violation_count": len(
            boundary_adjacent_violations
        ),
        "discounted_table_occupancy": float(np.sum(table_occupancy)),
        "discounted_tail_occupancy": float(np.sum(tail_occupancy)),
        "discounted_tail_entry_flow": (
            discounted_entry_flow
            if config.boundary_model == "tail"
            else 0.0
        ),
        "discounted_legacy_overflow_entry_flow": (
            discounted_entry_flow
            if config.boundary_model == "legacy_overflow"
            else 0.0
        ),
        "occupancy_iterations": occupancy_iterations,
        "occupancy_residual": occupancy_residual,
    }
    return RevealingResult(
        is_revealing=not core_violations,
        core_violations=core_violations,
        buffer_violations=buffer_violations,
        boundary_adjacent_violations=boundary_adjacent_violations,
        boundary_transmissions=boundary_transmissions,
        boundary_states=boundary_states,
        reachable_states=sorted(reachable),
        reachable_tail_states=sorted(reachable_tail),
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
        _validate_tx_policy(initial_pi_tx, mdp.n_states, config.delta_train),
        copy=True,
    )
    rx = np.array(
        _validate_rx_policy(
            initial_pi_rx,
            mdp.n_states,
            mdp.n_actions,
            config.delta_train,
        ),
        copy=True,
    )
    if (
        config.effective_boundary_tx_mode == "force_transmit"
        and np.any(tx[:, config.delta_train, :] != 1)
    ):
        raise ValueError(
            "initial_pi_tx must transmit at delta_train when "
            "boundary_tx_mode='force_transmit'"
        )
    logged_initial_tx = np.array(tx, copy=True)
    logged_initial_rx = np.array(rx, copy=True)
    current_evaluation = evaluate_policy(mdp, config, tx, rx, distribution)
    current_objective = objective_from_values(current_evaluation.values, distribution)
    initial_objective = current_objective
    history: list[dict[str, object]] = []
    initial_revealing = check_revealing(
        mdp, config, tx, rx, distribution, compute_occupancy=False
    )
    violation_history: list[dict[str, object]] = [
        {
            "api_iteration": 0,
            "revealing_violation_count": len(initial_revealing.violations),
            "core_violation_count": len(initial_revealing.core_violations),
            "buffer_violation_count": len(initial_revealing.buffer_violations),
            "boundary_adjacent_violation_count": len(
                initial_revealing.boundary_adjacent_violations
            ),
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
            mdp,
            config,
            tx,
            rx,
            distribution,
            compute_occupancy=False,
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
                "core_violation_count": len(
                    iteration_revealing.core_violations
                ),
                "buffer_violation_count": len(
                    iteration_revealing.buffer_violations
                ),
                "boundary_adjacent_violation_count": len(
                    iteration_revealing.boundary_adjacent_violations
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
                "core_violation_count": len(
                    iteration_revealing.core_violations
                ),
                "buffer_violation_count": len(
                    iteration_revealing.buffer_violations
                ),
                "boundary_adjacent_violation_count": len(
                    iteration_revealing.boundary_adjacent_violations
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
        "delta_train": config.delta_train,
        "delta_check": config.delta_check,
        "delta_max": config.delta_train,
        "boundary_model": config.boundary_model,
        "boundary_tx_mode": config.boundary_tx_mode,
        "effective_boundary_tx_mode": config.effective_boundary_tx_mode,
        "v_overflow": (
            config.overflow_value
            if config.boundary_model == "legacy_overflow"
            else None
        ),
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
        "core_violation_count": len(revealing.core_violations),
        "core_violations": revealing.core_violations,
        "buffer_violation_count": len(revealing.buffer_violations),
        "buffer_violations": revealing.buffer_violations,
        "boundary_adjacent_violation_count": len(
            revealing.boundary_adjacent_violations
        ),
        "boundary_adjacent_violations": (
            revealing.boundary_adjacent_violations
        ),
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
