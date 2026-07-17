"""Tests for policy analysis and deterministic JPO evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from jpo_model import JPOConfig, JPOModel
from jpo_policy import (
    PolicyAnalysisConfig,
    PolicyEvaluationConfig,
    RestrictedJPOPolicy,
    analyze_jpo_policy,
    evaluate_jpo_policy,
    simulate_jpo_policy,
)
from mdp import FiniteMDP


class ConstantPolicy:
    def __init__(self, action: int):
        self._action = action

    def action(self, _belief: np.ndarray) -> int:
        return self._action


def test_deterministic_evaluation_contains_known_infinite_horizon_value() -> None:
    reward = 0.5
    gamma = 0.8
    mdp = FiniteMDP(P=np.ones((1, 1, 1)), R=np.full((1, 1, 1), reward))
    model = JPOModel(
        mdp,
        JPOConfig(gamma=gamma, beta=0.3, epsilon=0.2),
    )
    never_transmit = model.encode_action(0, np.array([0]))

    result = evaluate_jpo_policy(
        model,
        ConstantPolicy(never_transmit),
        PolicyEvaluationConfig(tail_interval_tolerance=1e-10),
    )
    exact = reward / (1.0 - gamma)

    assert result.lower_values[0] <= exact <= result.upper_values[0]
    assert result.lower_objective <= exact <= result.upper_objective
    assert result.tail_interval_width <= 1e-10
    assert "Monte" not in result.method


def informative_model() -> JPOModel:
    transition = np.array([[[0.5, 0.5], [0.5, 0.5]]])
    reward = np.zeros_like(transition)
    return JPOModel(
        FiniteMDP(P=transition, R=reward),
        JPOConfig(gamma=0.5, beta=0.1, epsilon=0.2),
    )


def test_restricted_policy_keeps_null_posterior_only_after_training() -> None:
    model = informative_model()
    action = model.encode_action(0, np.array([1, 0]))
    base = ConstantPolicy(action)
    restricted = RestrictedJPOPolicy(model, base)
    belief = np.array([0.5, 0.5])
    null_posterior = model.posterior(
        belief, action, model.null_observation
    )

    assert null_posterior is not None
    assert restricted.action(belief) == base.action(belief)
    assert restricted.next_belief(belief, action, 0) == pytest.approx(
        null_posterior
    )
    assert restricted.next_belief(
        belief, action, model.null_observation
    ) == pytest.approx(null_posterior)


def test_analysis_finds_nonrevealing_transmissions_and_saves_policy_points() -> None:
    model = informative_model()
    action = model.encode_action(0, np.array([1, 0]))
    analysis = analyze_jpo_policy(
        model,
        ConstantPolicy(action),
        PolicyAnalysisConfig(
            discounted_tail_tolerance=1e-5,
            max_depth=50,
        ),
    )

    assert not analysis.is_revealing
    assert analysis.violations
    assert any(item["reached_state"] == 0 for item in analysis.violations)
    assert analysis.belief_records
    for record in analysis.belief_records:
        assert "belief" in record
        assert "action" in record
        assert "prescription" in record
        assert "discounted_occupancy" in record
    assert analysis.discounted_tail_bound <= 1e-5


def test_restricted_policy_is_evaluated_on_joint_state_and_internal_belief() -> None:
    model = informative_model()
    action = model.encode_action(0, np.array([1, 0]))
    restricted = RestrictedJPOPolicy(model, ConstantPolicy(action))

    result = evaluate_jpo_policy(
        model,
        restricted,
        PolicyEvaluationConfig(tail_interval_tolerance=1e-7),
    )

    assert np.all(np.isfinite(result.lower_values))
    assert np.all(result.lower_values <= result.upper_values)
    assert result.lower_objective <= result.upper_objective
    assert result.tail_interval_width <= 1e-7


def test_deterministic_evaluation_bounds_discarded_tiny_branch_mass() -> None:
    transition = np.array([[[0.999, 0.001], [0.999, 0.001]]])
    reward = np.array([[[0.0, 1.0], [0.0, 1.0]]])
    model = JPOModel(
        FiniteMDP(P=transition, R=reward),
        JPOConfig(
            gamma=0.5,
            beta=0.0,
            epsilon=0.0,
            probability_tolerance=0.01,
        ),
    )
    transmit_only_state_one = model.encode_action(0, np.array([0, 1]))
    result = evaluate_jpo_policy(
        model,
        ConstantPolicy(transmit_only_state_one),
        PolicyEvaluationConfig(tail_interval_tolerance=0.1),
    )
    exact = 0.001 / (1.0 - model.config.gamma)

    assert result.lower_objective <= exact <= result.upper_objective
    assert result.tail_interval_width <= 0.1


def test_simulation_is_optional_reproducible_validation() -> None:
    reward = 0.5
    gamma = 0.8
    model = JPOModel(
        FiniteMDP(P=np.ones((1, 1, 1)), R=np.full((1, 1, 1), reward)),
        JPOConfig(gamma=gamma, beta=0.2, epsilon=0.1),
    )
    policy = ConstantPolicy(model.encode_action(0, np.array([0])))

    first = simulate_jpo_policy(model, policy, episodes=3, horizon=4, seed=7)
    second = simulate_jpo_policy(model, policy, episodes=3, horizon=4, seed=7)
    expected = sum(gamma**depth * reward for depth in range(4))

    assert first.episode_returns == pytest.approx([expected] * 3)
    assert first.episode_returns == pytest.approx(second.episode_returns)
    assert first.mean_transmission_attempts == 0.0
