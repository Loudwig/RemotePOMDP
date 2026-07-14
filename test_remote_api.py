from pathlib import Path

import numpy as np
import pytest

from mdp import (
    FiniteMDP,
    create_effcom_control_family,
    initial_distribution,
    select_density,
)
from remote_api import (
    SolverConfig,
    check_revealing,
    compute_rx_beliefs,
    evaluate_policy,
    initialize_policies,
    objective_from_values,
    run_api,
    rx_greedy_candidate,
    rx_restricted_best_response,
    tx_best_response,
)


def test_effcom_family_has_expected_densities_and_normalized_rewards() -> None:
    family = create_effcom_control_family(10, 2, reward_decay=1.0, seed=1234)
    assert [mdp.density for mdp in family] == pytest.approx(
        [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    for support_size, mdp in zip((1, 3, 5, 7, 9), family):
        assert np.all(np.count_nonzero(mdp.P, axis=2) == support_size)
        assert np.min(mdp.R) >= 0.0
        assert np.max(mdp.R) <= 1.0
        assert np.max(mdp.R) == pytest.approx(1.0)


def test_zero_erasure_is_rejected_and_documented() -> None:
    with pytest.raises(NotImplementedError, match="TODO.md"):
        SolverConfig(gamma=0.9, beta=0.1, epsilon=0.0)
    todo = Path(__file__).with_name("TODO.md").read_text()
    assert "epsilon == 0" in todo


def test_legacy_boundary_uses_overflow_without_clipping() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.5]]]))
    config = SolverConfig(
        gamma=0.5,
        beta=0.2,
        epsilon=0.3,
        delta_train=1,
        delta_check=0,
        boundary_model="legacy_overflow",
        vi_tol=1e-12,
    )
    pi_tx = np.zeros((1, 2, 1), dtype=np.int64)
    pi_rx = np.zeros((2, 1), dtype=np.int64)
    result = evaluate_policy(mdp, config, pi_tx, pi_rx)
    expected = 0.5 + config.gamma * config.overflow_value
    assert result.values[0, 1, 0] == pytest.approx(expected, abs=1e-12)
    assert result.tail_values is None


def test_tail_value_is_solved_jointly_with_full_table_value() -> None:
    reward = 0.5
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[reward]]]))
    config = SolverConfig(
        gamma=0.5,
        beta=0.2,
        epsilon=0.3,
        delta_train=1,
        delta_check=0,
        boundary_tx_mode="free",
        vi_tol=1e-12,
    )
    pi_tx = np.zeros((1, 2, 1), dtype=np.int64)
    pi_rx = np.zeros((2, 1), dtype=np.int64)
    result = evaluate_policy(mdp, config, pi_tx, pi_rx)
    assert result.tail_values is not None

    denominator = (
        1.0
        - config.gamma * config.epsilon
        - config.gamma**3 * (1.0 - config.epsilon)
    )
    expected_tail = (
        -config.beta
        + reward
        + config.gamma
        * (1.0 - config.epsilon)
        * (reward + config.gamma * reward)
    ) / denominator
    assert result.tail_values[0, 0] == pytest.approx(
        expected_tail, abs=2e-11
    )
    assert result.values[0, 1, 0] == pytest.approx(
        reward + config.gamma * expected_tail, abs=2e-11
    )


def test_full_tail_success_continuation_is_v_next_state_zero_current_state() -> None:
    transition = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    rewards = np.array(
        [
            [[0.0, 0.2], [0.7, 0.0]],
            [[0.4, 0.0], [0.0, 1.0]],
        ]
    )
    mdp = FiniteMDP(P=transition, R=rewards)
    config = SolverConfig(
        gamma=0.6,
        beta=0.15,
        epsilon=0.25,
        delta_train=1,
        delta_check=0,
        boundary_tx_mode="free",
        vi_tol=1e-12,
    )
    pi_tx = np.zeros((2, 2, 2), dtype=np.int64)
    pi_tx[1, 0, 1] = 1
    pi_rx = np.array([[0, 1], [1, 0]], dtype=np.int64)
    result = evaluate_policy(mdp, config, pi_tx, pi_rx)
    assert result.tail_values is not None

    state = 0
    last_received = 0
    success_action = int(pi_rx[0, state])
    boundary_action = int(pi_rx[config.delta_train, last_received])
    correct_success = mdp.expected_rewards[success_action, state] + config.gamma * (
        mdp.P[success_action, state] @ result.values[:, 0, state]
    )
    failure = mdp.expected_rewards[boundary_action, state] + config.gamma * (
        mdp.P[boundary_action, state] @ result.tail_values[:, last_received]
    )
    expected = (
        -config.beta
        + (1.0 - config.epsilon) * correct_success
        + config.epsilon * failure
    )
    assert result.tail_values[state, last_received] == pytest.approx(
        expected, abs=2e-11
    )
    wrong_success = mdp.expected_rewards[success_action, state] + config.gamma * (
        mdp.P[success_action, state] @ result.values[:, 0, :].diagonal()
    )
    assert abs(correct_success - wrong_success) > 1e-4


def test_rx_tail_success_uses_aggregated_memory_value_v_rx_zero_s() -> None:
    transition = np.array([[[0.0, 1.0], [0.0, 1.0]]])
    rewards = np.array([[[0.0, 0.0], [0.0, 1.0]]])
    mdp = FiniteMDP(P=transition, R=rewards)
    config = SolverConfig(
        gamma=0.55,
        beta=0.1,
        epsilon=0.3,
        delta_train=1,
        delta_check=0,
        vi_tol=1e-12,
    )
    pi_tx = np.ones((2, 2, 2), dtype=np.int64)
    pi_rx = np.zeros((2, 2), dtype=np.int64)
    beliefs = compute_rx_beliefs(mdp, config, pi_tx, pi_rx)
    candidate = rx_greedy_candidate(mdp, config, pi_tx, pi_rx, beliefs)
    values = candidate.bellman.values
    tail = candidate.bellman.tail_values
    assert tail is not None
    assert abs(values[0, 0] - values[0, 1]) > 1e-4

    state = 0
    last_received = 0
    action = 0
    success = mdp.expected_rewards[action, state] + config.gamma * values[0, state]
    failure = mdp.expected_rewards[action, state] + config.gamma * (
        mdp.P[action, state] @ tail[:, last_received]
    )
    expected = (
        -config.beta
        + (1.0 - config.epsilon) * success
        + config.epsilon * failure
    )
    assert tail[state, last_received] == pytest.approx(expected, abs=2e-11)

    next_state = 1
    wrong_success = (
        mdp.expected_rewards[action, state]
        + config.gamma * values[0, next_state]
    )
    assert abs(success - wrong_success) > 1e-4


def test_boundary_tx_mode_forces_or_frees_boundary_action() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.0]]]))
    common = dict(
        gamma=0.6,
        beta=1.0,
        epsilon=0.3,
        delta_train=1,
        delta_check=0,
        vi_tol=1e-12,
    )
    pi_rx = np.zeros((2, 1), dtype=np.int64)
    previous = np.zeros((1, 2, 1), dtype=np.int64)
    forced = tx_best_response(
        mdp,
        SolverConfig(**common, boundary_tx_mode="force_transmit"),
        pi_rx,
        previous,
    )
    free = tx_best_response(
        mdp,
        SolverConfig(**common, boundary_tx_mode="free"),
        pi_rx,
        previous,
    )
    assert forced.policy[0, 1, 0] == 1
    assert free.policy[0, 1, 0] == 0


def test_success_memory_stores_transmitted_state_not_next_state() -> None:
    transition = np.array([[[0.0, 1.0], [0.0, 1.0]]])
    mdp = FiniteMDP(P=transition, R=np.zeros_like(transition))
    config = SolverConfig(
        gamma=0.5,
        beta=0.1,
        epsilon=0.5,
        delta_train=1,
        delta_check=0,
        boundary_model="legacy_overflow",
    )
    pi_tx = np.zeros((2, 2, 2), dtype=np.int64)
    pi_tx[0, 0, 0] = 1
    pi_rx = np.zeros((2, 2), dtype=np.int64)
    result = check_revealing(
        mdp,
        config,
        pi_tx,
        pi_rx,
        mu0=initial_distribution(2, initial_state=0),
    )
    assert (1, 0, 0) in result.reachable_states
    assert (1, 0, 1) not in result.reachable_states


def test_belief_recursion_uses_erasure_likelihood() -> None:
    transition = np.array([[[0.5, 0.5], [0.25, 0.75]]])
    mdp = FiniteMDP(P=transition, R=np.zeros_like(transition))
    config = SolverConfig(
        gamma=0.5, beta=0.1, epsilon=0.2, delta_train=1, delta_check=0
    )
    pi_tx = np.zeros((2, 2, 2), dtype=np.int64)
    pi_tx[1, 0, 0] = 1
    pi_rx = np.zeros((2, 2), dtype=np.int64)
    beliefs = compute_rx_beliefs(mdp, config, pi_tx, pi_rx)
    assert beliefs.valid[1, 0]
    assert beliefs.beliefs[1, 0] == pytest.approx([5.0 / 6.0, 1.0 / 6.0])


def test_tx_stable_tie_keeps_previous_action() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.0]]]))
    config = SolverConfig(
        gamma=0.5,
        beta=0.0,
        epsilon=0.5,
        delta_train=1,
        delta_check=0,
        vi_tol=1e-12,
    )
    pi_rx = np.zeros((2, 1), dtype=np.int64)
    previous = np.ones((1, 2, 1), dtype=np.int64)
    result = tx_best_response(mdp, config, pi_rx, previous_pi_tx=previous)
    assert result.policy[0, 0, 0] == 1


def test_revealing_uses_only_reachable_interior_states_and_separates_boundary() -> None:
    transition = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    mdp = FiniteMDP(P=transition, R=np.zeros_like(transition))
    config = SolverConfig(
        gamma=0.5, beta=0.1, epsilon=0.5, delta_train=3, delta_check=1
    )
    pi_tx = np.zeros((2, 4, 2), dtype=np.int64)
    pi_tx[0, 0, 0] = 1  # Reachable interior violation with one Rx action.
    pi_tx[0, 3, 0] = 1  # Reachable boundary transmission.
    pi_tx[1, 0, 1] = 1  # Unreachable and therefore ignored.
    pi_rx = np.zeros((4, 2), dtype=np.int64)
    result = check_revealing(
        mdp,
        config,
        pi_tx,
        pi_rx,
        mu0=initial_distribution(2, initial_state=0),
    )
    assert not result.is_revealing
    assert len(result.violations) == 1
    assert result.violations[0]["state"] == 0
    assert len(result.boundary_transmissions) == 1
    assert all(item["state"] != 1 for item in result.violations)


def test_revealing_categories_and_tail_occupancy_are_separate() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.0]]]))
    config = SolverConfig(
        gamma=0.6,
        beta=0.1,
        epsilon=0.5,
        delta_train=4,
        delta_check=1,
        vi_tol=1e-12,
    )
    pi_tx = np.ones((1, 5, 1), dtype=np.int64)
    pi_rx = np.zeros((5, 1), dtype=np.int64)
    result = check_revealing(mdp, config, pi_tx, pi_rx)

    assert [item["age"] for item in result.core_violations] == [0, 1]
    assert [item["age"] for item in result.buffer_violations] == [2]
    assert [
        item["age"] for item in result.boundary_adjacent_violations
    ] == [3]
    assert [item["age"] for item in result.boundary_transmissions] == [4]
    assert not result.is_revealing
    assert all(
        item["discounted_occupancy"] > 0.0
        for item in (
            result.core_violations
            + result.buffer_violations
            + result.boundary_adjacent_violations
        )
    )
    assert result.statistics["reachable_tail_count"] == 1
    assert result.statistics["discounted_tail_occupancy"] > 0.0
    assert result.statistics["discounted_tail_entry_flow"] > 0.0
    total_occupancy = (
        result.statistics["discounted_table_occupancy"]
        + result.statistics["discounted_tail_occupancy"]
    )
    assert total_occupancy == pytest.approx(
        1.0 / (1.0 - config.gamma), abs=1e-10
    )


def test_boundary_adjacent_violations_are_not_core() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.0]]]))
    config = SolverConfig(
        gamma=0.5, beta=0.1, epsilon=0.5, delta_train=2, delta_check=0
    )
    pi_tx = np.zeros((1, 3, 1), dtype=np.int64)
    pi_tx[0, 1, 0] = 1
    pi_rx = np.zeros((3, 1), dtype=np.int64)
    result = check_revealing(mdp, config, pi_tx, pi_rx)
    assert result.is_revealing
    assert result.violations == []
    assert len(result.boundary_adjacent_violations) == 1
    assert result.boundary_adjacent_violations[0]["age"] == 1


def test_lower_bound_ignores_nonrevealing_success() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[1.0]]]))
    config = SolverConfig(
        gamma=0.7,
        beta=0.1,
        epsilon=0.2,
        delta_train=1,
        delta_check=0,
        boundary_tx_mode="free",
        vi_tol=1e-12,
    )
    pi_tx = np.zeros((1, 2, 1), dtype=np.int64)
    pi_rx = np.zeros((2, 1), dtype=np.int64)
    two_way = evaluate_policy(mdp, config, pi_tx, pi_rx, mode="two_way")
    lower_bound = evaluate_policy(mdp, config, pi_tx, pi_rx, mode="lower_bound")
    assert lower_bound.values[0, 0, 0] < two_way.values[0, 0, 0]


def test_restricted_rx_never_decreases_true_objective() -> None:
    mdp = select_density(create_effcom_control_family(4, 2, seed=8), 0.25)
    config = SolverConfig(
        gamma=0.6,
        beta=0.1,
        epsilon=0.2,
        delta_train=3,
        delta_check=1,
        vi_tol=1e-10,
        max_rx_iterations=10,
    )
    pi_tx, pi_rx = initialize_policies(mdp, config, seed=8)
    pi_tx = tx_best_response(mdp, config, pi_rx, pi_tx).policy
    mu0 = initial_distribution(mdp.n_states)
    before = evaluate_policy(mdp, config, pi_tx, pi_rx, mu0)
    before_objective = objective_from_values(before.values, mu0)
    improved = rx_restricted_best_response(mdp, config, pi_tx, pi_rx, mu0)
    assert improved.objective >= before_objective - 1e-12
    accepted_objectives = [
        step["candidate_objective"]
        for step in improved.history
        if step.get("accepted")
    ]
    assert accepted_objectives == sorted(accepted_objectives)


@pytest.mark.parametrize("density", [0.1, 0.9])
def test_api_smoke_on_simple_and_dense_effcom_mdps(density: float) -> None:
    mdp = select_density(create_effcom_control_family(10, 2, seed=12), density)
    config = SolverConfig(
        gamma=0.5,
        beta=0.15,
        epsilon=0.2,
        delta_train=20,
        delta_check=10,
        vi_tol=1e-8,
        rx_accept_tol=1e-7,
        api_tol=1e-7,
        ne_tol=1e-6,
        max_vi_iterations=10_000,
        max_rx_iterations=10,
        max_api_iterations=5,
    )
    result = run_api(mdp, config, seed=12)
    assert np.isfinite(result.objective)
    assert result.two_way.residual <= config.vi_tol
    assert result.pi_tx.shape == (10, 21, 10)
    assert result.pi_rx.shape == (21, 10)
    assert result.diagnostics["density"] == pytest.approx(density)
    assert result.objective >= result.diagnostics["initial_objective"] - 1e-7
    assert result.violation_history[0]["api_iteration"] == 0
    assert len(result.violation_history) == result.api_iterations + 1
