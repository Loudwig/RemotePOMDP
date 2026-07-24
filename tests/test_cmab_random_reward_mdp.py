"""Tests for the CMAB family with random state-action rewards."""

from __future__ import annotations

import numpy as np
import pytest

from mdp import (
    create_cmab_estimation_family,
    create_cmab_random_reward_family,
)
from run_jpo import build_parser, build_physical_mdp


def test_random_reward_cmab_keeps_estimation_family_transitions() -> None:
    estimation_family = create_cmab_estimation_family(n_states=6, seed=1111)
    random_reward_family = create_cmab_random_reward_family(
        n_states=6,
        n_actions=3,
        seed=1111,
    )

    assert [mdp.density for mdp in random_reward_family] == pytest.approx(
        [1 / 6, 0.5, 5 / 6]
    )
    for estimation_mdp, random_reward_mdp in zip(
        estimation_family,
        random_reward_family,
    ):
        assert random_reward_mdp.P.shape == (3, 6, 6)
        assert np.array_equal(
            random_reward_mdp.P,
            np.broadcast_to(estimation_mdp.P[:1], (3, 6, 6)),
        )


def test_random_reward_cmab_rewards_depend_only_on_state_and_action() -> None:
    family = create_cmab_random_reward_family(
        n_states=6,
        n_actions=3,
        seed=1111,
    )
    state_action_rewards = family[0].R[:, :, 0]

    assert state_action_rewards.shape == (3, 6)
    assert np.all((0.0 <= state_action_rewards) & (state_action_rewards <= 1.0))
    assert np.unique(state_action_rewards).size > 1

    for mdp in family:
        assert np.array_equal(
            mdp.R,
            np.broadcast_to(state_action_rewards[:, :, None], mdp.R.shape),
        )
        assert np.allclose(
            mdp.expected_rewards,
            state_action_rewards,
            atol=1e-15,
            rtol=1e-15,
        )


def test_random_reward_cmab_is_reproducible() -> None:
    first = create_cmab_random_reward_family(n_states=4, n_actions=3, seed=8)
    second = create_cmab_random_reward_family(n_states=4, n_actions=3, seed=8)

    for first_mdp, second_mdp in zip(first, second):
        assert np.array_equal(first_mdp.P, second_mdp.P)
        assert np.array_equal(first_mdp.R, second_mdp.R)


def test_random_reward_cmab_requires_a_positive_action_count() -> None:
    for n_actions in (0, -1):
        with pytest.raises(ValueError, match="n_actions must be positive"):
            create_cmab_random_reward_family(n_states=4, n_actions=n_actions)


def test_run_jpo_cli_can_select_random_reward_cmab_family() -> None:
    args = build_parser().parse_args(
        [
            "--mdp-type",
            "cmab-random-reward",
            "--n-states",
            "6",
            "--n-actions",
            "3",
            "--density",
            "0.5",
            "--mdp-seed",
            "1111",
        ]
    )
    mdp = build_physical_mdp(args)

    assert mdp.n_states == 6
    assert mdp.n_actions == 3
    assert mdp.density == pytest.approx(0.5)
    assert np.allclose(mdp.P, mdp.P[0][None, :, :])
    assert np.array_equal(
        mdp.R,
        np.broadcast_to(mdp.R[:, :, :1], mdp.R.shape),
    )
