"""JPO policy analysis, restriction, and deterministic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from jpo_model import JPOModel


@dataclass(frozen=True)
class PolicyAnalysisConfig:
    discounted_tail_tolerance: float = 1e-8
    max_depth: int = 10_000
    max_belief_nodes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.discounted_tail_tolerance <= 0.0 or not np.isfinite(
            self.discounted_tail_tolerance
        ):
            raise ValueError("discounted_tail_tolerance must be finite and positive")
        for name in ("max_depth", "max_belief_nodes"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class PolicyAnalysisResult:
    is_revealing: bool
    violations: list[dict[str, Any]]
    belief_records: list[dict[str, Any]]
    horizon: int
    discounted_tail_bound: float
    total_belief_nodes: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "is_revealing": self.is_revealing,
            "violation_count": len(self.violations),
            "violations": self.violations,
            "belief_record_count": len(self.belief_records),
            "belief_records": self.belief_records,
            "horizon": self.horizon,
            "discounted_tail_bound": self.discounted_tail_bound,
            "total_belief_nodes": self.total_belief_nodes,
        }


@dataclass(frozen=True)
class PolicyEvaluationConfig:
    tail_interval_tolerance: float = 1e-8
    max_horizon: int = 10_000
    max_belief_nodes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.tail_interval_tolerance <= 0.0 or not np.isfinite(
            self.tail_interval_tolerance
        ):
            raise ValueError("tail_interval_tolerance must be finite and positive")
        for name in ("max_horizon", "max_belief_nodes"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class PolicyEvaluationResult:
    lower_values: np.ndarray
    upper_values: np.ndarray
    value_estimates: np.ndarray
    lower_objective: float
    upper_objective: float
    objective_estimate: float
    horizon: int
    tail_interval_width: float
    total_belief_nodes: int
    method: str = "deterministic_joint_state_belief_propagation"

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "lower_values": self.lower_values.tolist(),
            "upper_values": self.upper_values.tolist(),
            "value_estimates": self.value_estimates.tolist(),
            "lower_objective": self.lower_objective,
            "upper_objective": self.upper_objective,
            "objective_estimate": self.objective_estimate,
            "horizon": self.horizon,
            "tail_interval_width": self.tail_interval_width,
            "total_belief_nodes": self.total_belief_nodes,
        }


@dataclass
class PolicySimulationResult:
    episode_returns: np.ndarray
    mean_return: float
    standard_error: float
    mean_transmission_attempts: float
    mean_successful_messages: float
    episodes: int
    horizon: int
    seed: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "method": "monte_carlo_validation_only",
            "mean_return": self.mean_return,
            "standard_error": self.standard_error,
            "mean_transmission_attempts": self.mean_transmission_attempts,
            "mean_successful_messages": self.mean_successful_messages,
            "episodes": self.episodes,
            "horizon": self.horizon,
            "seed": self.seed,
            "episode_returns": self.episode_returns.tolist(),
        }


class RestrictedJPOPolicy:
    """API-style lower-bound controller constructed after JPO training.

    The selected transformed action is unchanged.  After a successful message,
    if the next receiver action is identical to the no-reception receiver
    action, the controller retains the no-reception posterior instead of using
    the revealing Dirac posterior.  This restriction is post-processing only;
    it is never used during JPO training.
    """

    def __init__(self, model: JPOModel, base_policy: Any):
        self.model = model
        self.base_policy = base_policy

    def action(self, belief: np.ndarray) -> int:
        return int(self.base_policy.action(belief))

    def next_belief(
        self, belief: np.ndarray, action: int, observation: int
    ) -> np.ndarray | None:
        standard = self.model.posterior(belief, action, observation)
        if standard is None or observation == self.model.null_observation:
            return standard
        null_posterior = self.model.posterior(
            belief, action, self.model.null_observation
        )
        if null_posterior is None:
            return standard
        success_action = self.action(standard)
        no_reception_action = self.action(null_posterior)
        success_rx = self.model.decode_action(success_action).receiver_action
        no_reception_rx = self.model.decode_action(
            no_reception_action
        ).receiver_action
        if success_rx == no_reception_rx:
            return null_posterior
        return standard


def analyze_jpo_policy(
    model: JPOModel,
    policy: Any,
    config: PolicyAnalysisConfig | None = None,
) -> PolicyAnalysisResult:
    """Analyze revealing violations over the discounted reachable policy tree."""

    analysis_config = config or PolicyAnalysisConfig()
    horizon = _discounted_occupancy_horizon(
        model.config.gamma,
        analysis_config.discounted_tail_tolerance,
        analysis_config.max_depth,
    )
    roots, weights = model.initial_beliefs()
    layer: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
    for state, (belief, weight) in enumerate(zip(roots, weights)):
        state_mass = np.zeros(model.n_states)
        state_mass[state] = weight
        layer[_belief_key(belief)] = (belief, state_mass)

    records: dict[tuple[float, ...], dict[str, Any]] = {}
    violations: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}
    total_nodes = 0

    for depth in range(horizon):
        discount = model.config.gamma**depth
        next_layer: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
        for key, (belief, state_mass) in layer.items():
            total_nodes += 1
            if total_nodes > analysis_config.max_belief_nodes:
                raise RuntimeError("policy analysis exceeded max_belief_nodes")
            probability = float(state_mass.sum())
            action = int(policy.action(belief))
            decoded = model.decode_action(action)
            record = records.get(key)
            if record is None:
                record = {
                    "belief": belief.tolist(),
                    "action": action,
                    "receiver_action": decoded.receiver_action,
                    "prescription_index": decoded.prescription_index,
                    "prescription": decoded.prescription.tolist(),
                    "first_depth": depth,
                    "expected_visits_through_horizon": 0.0,
                    "discounted_occupancy": 0.0,
                }
                records[key] = record
            elif record["action"] != action:
                raise RuntimeError("deterministic policy changed action at one belief")
            record["expected_visits_through_horizon"] += probability
            record["discounted_occupancy"] += discount * probability

            observation_probabilities = model.observation_probabilities(
                belief, action
            )
            null_probability = float(
                observation_probabilities[model.null_observation]
            )
            null_posterior = model.posterior(
                belief, action, model.null_observation
            )
            if null_posterior is not None:
                no_reception_action = int(policy.action(null_posterior))
                no_reception_rx = model.decode_action(
                    no_reception_action
                ).receiver_action
                prediction_mass = state_mass @ model.transition_matrix(action)
                for state in range(model.n_states):
                    success_probability = float(observation_probabilities[state])
                    if success_probability <= model.config.probability_tolerance:
                        continue
                    dirac = np.zeros(model.n_states)
                    dirac[state] = 1.0
                    success_action = int(policy.action(dirac))
                    success_rx = model.decode_action(
                        success_action
                    ).receiver_action
                    if success_rx != no_reception_rx:
                        continue
                    violation_key = (key, state)
                    actual_success_flow = float(
                        prediction_mass[state]
                        * model.observation_kernel(action)[state, state]
                    )
                    violation = violations.get(violation_key)
                    if violation is None:
                        violation = {
                            "belief": belief.tolist(),
                            "action": action,
                            "receiver_action": decoded.receiver_action,
                            "prescription_index": decoded.prescription_index,
                            "prescription": decoded.prescription.tolist(),
                            "reached_state": state,
                            "success_probability_given_belief": success_probability,
                            "null_probability_given_belief": null_probability,
                            "success_next_action": success_action,
                            "no_reception_next_action": no_reception_action,
                            "common_next_receiver_action": success_rx,
                            "first_depth": depth,
                            "discounted_event_occupancy": 0.0,
                        }
                        violations[violation_key] = violation
                    violation["discounted_event_occupancy"] += (
                        model.config.gamma * discount * actual_success_flow
                    )

            prediction_mass = state_mass @ model.transition_matrix(action)
            kernel = model.observation_kernel(action)
            for observation in range(model.n_observations):
                branch_state_mass = prediction_mass * kernel[observation]
                if branch_state_mass.sum() <= model.config.probability_tolerance:
                    continue
                posterior = model.posterior(belief, action, observation)
                if posterior is None:
                    raise RuntimeError(
                        "a physically reachable observation has no JPO posterior"
                    )
                _merge_node(next_layer, posterior, branch_state_mass)
        layer = next_layer

    tail_bound = model.config.gamma**horizon / (1.0 - model.config.gamma)
    violation_list = sorted(
        violations.values(),
        key=lambda item: (
            item["first_depth"],
            item["belief"],
            item["reached_state"],
        ),
    )
    belief_records = sorted(
        records.values(), key=lambda item: (item["first_depth"], item["belief"])
    )
    return PolicyAnalysisResult(
        is_revealing=not violation_list,
        violations=violation_list,
        belief_records=belief_records,
        horizon=horizon,
        discounted_tail_bound=tail_bound,
        total_belief_nodes=total_nodes,
    )


def evaluate_jpo_policy(
    model: JPOModel,
    policy: Any,
    config: PolicyEvaluationConfig | None = None,
) -> PolicyEvaluationResult:
    """Evaluate a policy without Monte Carlo using joint state-belief mass.

    Tracking the physical-state mass separately from the controller's internal
    belief is essential for the restricted lower-bound policy, whose retained
    belief intentionally differs from the posterior after some successes.
    """

    evaluation_config = config or PolicyEvaluationConfig()
    stage_lower = float(np.min(model.mdp.R) - model.config.gamma * model.config.beta)
    stage_upper = float(np.max(model.mdp.R))
    # Reserve half the requested width for the certified contribution of
    # individually negligible branches discarded during propagation.
    horizon = _tail_interval_horizon(
        model.config.gamma,
        stage_upper - stage_lower,
        0.5 * evaluation_config.tail_interval_tolerance,
        evaluation_config.max_horizon,
    )
    roots, weights = model.initial_beliefs()
    truncated_lower_values = np.empty(model.n_states)
    truncated_upper_values = np.empty(model.n_states)
    total_nodes = 0

    for initial_state, root in enumerate(roots):
        state_mass = np.zeros(model.n_states)
        state_mass[initial_state] = 1.0
        layer: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {
            _belief_key(root): (root, state_mass)
        }
        truncated_lower = 0.0
        truncated_upper = 0.0

        for depth in range(horizon):
            next_layer: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
            stage_value = 0.0
            retained_probability = sum(
                float(node_state_mass.sum())
                for _, node_state_mass in layer.values()
            )
            if retained_probability < -1e-12 or retained_probability > 1.0 + 1e-9:
                raise FloatingPointError(
                    "joint state-belief propagation produced invalid probability mass"
                )
            missing_probability = max(0.0, 1.0 - retained_probability)
            for belief, node_state_mass in layer.values():
                total_nodes += 1
                if total_nodes > evaluation_config.max_belief_nodes:
                    raise RuntimeError("policy evaluation exceeded max_belief_nodes")
                action = int(policy.action(belief))
                stage_value += float(
                    np.dot(node_state_mass, model.reward_vector(action))
                )
                prediction_mass = node_state_mass @ model.transition_matrix(action)
                kernel = model.observation_kernel(action)
                for observation in range(model.n_observations):
                    branch_state_mass = prediction_mass * kernel[observation]
                    if branch_state_mass.sum() <= model.config.probability_tolerance:
                        continue
                    if hasattr(policy, "next_belief"):
                        next_belief = policy.next_belief(
                            belief, action, observation
                        )
                    else:
                        next_belief = model.posterior(
                            belief, action, observation
                        )
                    if next_belief is None:
                        raise RuntimeError(
                            "the evaluated controller has no update for a "
                            "physically reachable observation"
                        )
                    _merge_node(next_layer, next_belief, branch_state_mass)
            next_probability = sum(
                float(node_state_mass.sum())
                for _, node_state_mass in next_layer.values()
            )
            if next_probability > retained_probability + 1e-9:
                raise FloatingPointError(
                    "joint state-belief propagation created probability mass"
                )
            discount = model.config.gamma**depth
            truncated_lower += discount * (
                stage_value + missing_probability * stage_lower
            )
            truncated_upper += discount * (
                stage_value + missing_probability * stage_upper
            )
            layer = next_layer
        truncated_lower_values[initial_state] = truncated_lower
        truncated_upper_values[initial_state] = truncated_upper

    tail_scale = model.config.gamma**horizon / (1.0 - model.config.gamma)
    lower_values = truncated_lower_values + tail_scale * stage_lower
    upper_values = truncated_upper_values + tail_scale * stage_upper
    estimates = 0.5 * (lower_values + upper_values)
    lower_objective = float(np.dot(weights, lower_values))
    upper_objective = float(np.dot(weights, upper_values))
    estimate = float(np.dot(weights, estimates))
    return PolicyEvaluationResult(
        lower_values=lower_values,
        upper_values=upper_values,
        value_estimates=estimates,
        lower_objective=lower_objective,
        upper_objective=upper_objective,
        objective_estimate=estimate,
        horizon=horizon,
        tail_interval_width=upper_objective - lower_objective,
        total_belief_nodes=total_nodes,
    )


def simulate_jpo_policy(
    model: JPOModel,
    policy: Any,
    episodes: int,
    horizon: int,
    seed: int = 1234,
) -> PolicySimulationResult:
    """Run optional Monte Carlo validation using the required JPO timing."""

    for name, value in (("episodes", episodes), ("horizon", horizon)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rng = np.random.default_rng(seed)
    returns = np.zeros(episodes)
    attempts = np.zeros(episodes)
    successes = np.zeros(episodes)

    for episode in range(episodes):
        state = int(rng.integers(model.n_states))
        belief = np.zeros(model.n_states)
        belief[state] = 1.0
        discounted_return = 0.0

        for depth in range(horizon):
            action = int(policy.action(belief))
            decoded = model.decode_action(action)
            next_state = int(
                rng.choice(
                    model.n_states,
                    p=model.mdp.P[decoded.receiver_action, state],
                )
            )
            transmission = int(decoded.prescription[next_state])
            attempts[episode] += transmission
            if transmission and rng.random() >= model.config.epsilon:
                observation = next_state
                successes[episode] += 1.0
            else:
                observation = model.null_observation
            reward = (
                model.mdp.R[decoded.receiver_action, state, next_state]
                - model.config.gamma * model.config.beta * transmission
            )
            discounted_return += model.config.gamma**depth * reward
            if hasattr(policy, "next_belief"):
                next_belief = policy.next_belief(belief, action, observation)
            else:
                next_belief = model.posterior(belief, action, observation)
            if next_belief is None:
                raise RuntimeError(
                    "simulation reached an observation without a controller update"
                )
            belief = next_belief
            state = next_state
        returns[episode] = discounted_return

    standard_error = (
        float(np.std(returns, ddof=1) / np.sqrt(episodes))
        if episodes > 1
        else 0.0
    )
    return PolicySimulationResult(
        episode_returns=returns,
        mean_return=float(np.mean(returns)),
        standard_error=standard_error,
        mean_transmission_attempts=float(np.mean(attempts)),
        mean_successful_messages=float(np.mean(successes)),
        episodes=episodes,
        horizon=horizon,
        seed=seed,
    )


def _merge_node(
    layer: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]],
    belief: np.ndarray,
    state_mass: np.ndarray,
) -> None:
    key = _belief_key(belief)
    existing = layer.get(key)
    if existing is None:
        layer[key] = (np.array(belief, copy=True), np.array(state_mass, copy=True))
    else:
        existing[1][:] += state_mass


def _belief_key(belief: np.ndarray) -> tuple[float, ...]:
    # No quantization: merging nearby beliefs could cross an alpha-vector
    # policy boundary and would invalidate the deterministic value interval.
    return tuple(float(value) for value in belief)


def _discounted_occupancy_horizon(
    gamma: float, tolerance: float, maximum: int
) -> int:
    required = max(
        1,
        int(math.ceil(math.log(tolerance * (1.0 - gamma)) / math.log(gamma))),
    )
    if required > maximum:
        raise RuntimeError(
            "max_depth is too small to certify the requested analysis tail "
            f"tolerance; need at least {required}, got {maximum}"
        )
    return required


def _tail_interval_horizon(
    gamma: float, stage_width: float, tolerance: float, maximum: int
) -> int:
    if stage_width == 0.0:
        return 1
    target = tolerance * (1.0 - gamma) / stage_width
    required = max(1, int(math.ceil(math.log(target) / math.log(gamma))))
    if required > maximum:
        raise RuntimeError(
            "max_horizon is too small to certify the requested policy-value "
            f"interval; need at least {required}, got {maximum}"
        )
    return required
