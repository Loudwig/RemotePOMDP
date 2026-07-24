"""Tests for the action-independent CMAB estimation-MDP family."""

from __future__ import annotations

import numpy as np
import pytest

from jpo_model import JPOConfig, JPOModel
from mdp import create_cmab_estimation_family, select_density
from run_jpo import build_parser, build_physical_mdp


def test_cmab_family_has_shared_dynamics_and_estimation_rewards() -> None:
    family = create_cmab_estimation_family(n_states=6, seed=1111)

    assert [mdp.density for mdp in family] == pytest.approx([1 / 6, 0.5, 5 / 6])
    expected_rewards = np.eye(6)
    deterministic_centers = np.argmax(family[0].P[0], axis=1)

    for support_size, mdp in zip((1, 3, 5), family):
        assert mdp.P.shape == (6, 6, 6)
        assert mdp.R.shape == (6, 6, 6)
        assert mdp.n_actions == mdp.n_states == 6
        assert mdp.optimal_state is None
        assert np.allclose(mdp.P, mdp.P[0][None, :, :])
        assert np.all(np.count_nonzero(mdp.P, axis=2) == support_size)
        assert np.allclose(mdp.P.sum(axis=2), 1.0)
        assert np.array_equal(np.argmax(mdp.P[0], axis=1), deterministic_centers)
        assert np.array_equal(mdp.expected_rewards, expected_rewards)

        for action in range(mdp.n_actions):
            for state in range(mdp.n_states):
                assert np.all(mdp.R[action, state] == float(action == state))


def test_cmab_family_is_reproducible_and_jpo_compatible() -> None:
    first = create_cmab_estimation_family(n_states=4, seed=8)
    second = create_cmab_estimation_family(n_states=4, seed=8)
    for first_mdp, second_mdp in zip(first, second):
        assert np.array_equal(first_mdp.P, second_mdp.P)
        assert np.array_equal(first_mdp.R, second_mdp.R)

    mdp = select_density(first, 0.25)
    model = JPOModel(mdp, JPOConfig(gamma=0.9, beta=0.1, epsilon=0.1))
    prescription = np.zeros(mdp.n_states, dtype=np.int64)
    first_action = model.encode_action(0, prescription)
    for receiver_action in range(1, mdp.n_actions):
        action = model.encode_action(receiver_action, prescription)
        assert np.array_equal(
            model.transition_matrix(action),
            model.transition_matrix(first_action),
        )


def test_cmab_density_half_uses_effcom_circular_weights() -> None:
    mdp = select_density(
        create_cmab_estimation_family(n_states=6, seed=1111),
        0.5,
    )

    for state, row in enumerate(mdp.P[0]):
        center = int(np.argmax(row))
        expected = np.zeros(mdp.n_states)
        expected[(center - 1) % mdp.n_states] = 5 / 28
        expected[center] = 9 / 14
        expected[(center + 1) % mdp.n_states] = 5 / 28
        assert row == pytest.approx(expected), f"unexpected row for state {state}"


def test_cmab_family_requires_a_positive_even_state_count() -> None:
    for n_states in (0, 1, 3):
        with pytest.raises(ValueError, match="positive even"):
            create_cmab_estimation_family(n_states=n_states)


def test_run_jpo_cli_can_select_cmab_family() -> None:
    args = build_parser().parse_args(
        [
            "--mdp-type",
            "cmab-estimation",
            "--n-states",
            "6",
            "--density",
            "0.5",
            "--mdp-seed",
            "1111",
        ]
    )
    mdp = build_physical_mdp(args)

    assert mdp.n_states == mdp.n_actions == 6
    assert mdp.density == pytest.approx(0.5)
    assert np.allclose(mdp.P, mdp.P[0][None, :, :])
