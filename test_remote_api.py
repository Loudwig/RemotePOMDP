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


def test_boundary_uses_overflow_without_clipping() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.5]]]))
    config = SolverConfig(
        gamma=0.5, beta=0.2, epsilon=0.3, delta_max=0, vi_tol=1e-12
    )
    pi_tx = np.zeros((1, 1, 1), dtype=np.int64)
    pi_rx = np.zeros((1, 1), dtype=np.int64)
    result = evaluate_policy(mdp, config, pi_tx, pi_rx)
    expected = 0.5 + config.gamma * config.overflow_value
    assert result.values[0, 0, 0] == pytest.approx(expected, abs=1e-12)


def test_success_memory_stores_transmitted_state_not_next_state() -> None:
    transition = np.array([[[0.0, 1.0], [0.0, 1.0]]])
    mdp = FiniteMDP(P=transition, R=np.zeros_like(transition))
    config = SolverConfig(gamma=0.5, beta=0.1, epsilon=0.5, delta_max=1)
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
    config = SolverConfig(gamma=0.5, beta=0.1, epsilon=0.2, delta_max=1)
    pi_tx = np.zeros((2, 2, 2), dtype=np.int64)
    pi_tx[1, 0, 0] = 1
    pi_rx = np.zeros((2, 2), dtype=np.int64)
    beliefs = compute_rx_beliefs(mdp, config, pi_tx, pi_rx)
    assert beliefs.valid[1, 0]
    assert beliefs.beliefs[1, 0] == pytest.approx([5.0 / 6.0, 1.0 / 6.0])


def test_tx_stable_tie_keeps_previous_action() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.0]]]))
    config = SolverConfig(
        gamma=0.5, beta=0.0, epsilon=0.5, delta_max=0, vi_tol=1e-12
    )
    pi_rx = np.zeros((1, 1), dtype=np.int64)
    previous = np.ones((1, 1, 1), dtype=np.int64)
    result = tx_best_response(mdp, config, pi_rx, previous_pi_tx=previous)
    assert result.policy[0, 0, 0] == 1


def test_revealing_uses_only_reachable_interior_states_and_separates_boundary() -> None:
    transition = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    mdp = FiniteMDP(P=transition, R=np.zeros_like(transition))
    config = SolverConfig(gamma=0.5, beta=0.1, epsilon=0.5, delta_max=2)
    pi_tx = np.zeros((2, 3, 2), dtype=np.int64)
    pi_tx[0, 0, 0] = 1  # Reachable interior violation with one Rx action.
    pi_tx[0, 2, 0] = 1  # Reachable boundary transmission.
    pi_tx[1, 0, 1] = 1  # Unreachable and therefore ignored.
    pi_rx = np.zeros((3, 2), dtype=np.int64)
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


def test_delta_max_minus_one_violations_are_deferred() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[0.0]]]))
    config = SolverConfig(gamma=0.5, beta=0.1, epsilon=0.5, delta_max=1)
    pi_tx = np.ones((1, 2, 1), dtype=np.int64)
    pi_rx = np.zeros((2, 1), dtype=np.int64)
    result = check_revealing(mdp, config, pi_tx, pi_rx)
    assert result.is_revealing
    assert result.violations == []
    assert len(result.deferred_boundary_layer_violations) == 1
    assert result.deferred_boundary_layer_violations[0]["age"] == 0


def test_lower_bound_ignores_nonrevealing_success() -> None:
    mdp = FiniteMDP(P=np.array([[[1.0]]]), R=np.array([[[1.0]]]))
    config = SolverConfig(
        gamma=0.7, beta=0.1, epsilon=0.2, delta_max=1, vi_tol=1e-12
    )
    pi_tx = np.ones((1, 2, 1), dtype=np.int64)
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
        delta_max=3,
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
        delta_max=20,
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
