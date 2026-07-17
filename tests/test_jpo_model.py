"""Focused tests for the transformed JPO POMDP model."""

from __future__ import annotations

import numpy as np
import pytest

from jpo_model import JPOConfig, JPOModel
from mdp import FiniteMDP


def two_state_mdp() -> FiniteMDP:
    transitions = np.array(
        [
            [[0.7, 0.3], [0.2, 0.8]],
            [[0.4, 0.6], [0.9, 0.1]],
        ]
    )
    rewards = np.array(
        [
            [[0.1, 0.8], [0.4, 0.2]],
            [[0.3, 0.7], [0.9, 0.0]],
        ]
    )
    return FiniteMDP(P=transitions, R=rewards)


def test_effcom_action_ordering_round_trips() -> None:
    mdp = FiniteMDP(
        P=np.broadcast_to(np.eye(3), (2, 3, 3)),
        R=np.zeros((2, 3, 3)),
    )
    model = JPOModel(mdp, JPOConfig(gamma=0.9, beta=0.1, epsilon=0.2))

    # 101 is prescription index five.  EffCom makes the prescription the
    # major action index and the receiver action the minor index.
    action = model.encode_action(1, np.array([1, 0, 1]))
    assert action == 5 * mdp.n_actions + 1
    decoded = model.decode_action(action)
    assert decoded.receiver_action == 1
    assert decoded.prescription_index == 5
    assert decoded.prescription.tolist() == [1, 0, 1]
    assert model.encode_action(1, 5) == action


def test_observation_probabilities_normalize_for_every_action() -> None:
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=0.25),
    )
    belief = np.array([0.35, 0.65])

    for action in range(model.n_actions):
        probabilities = model.observation_probabilities(belief, action)
        assert np.all(probabilities >= 0.0)
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-13)
        for observation, probability in enumerate(probabilities):
            posterior = model.posterior(belief, action, observation)
            if probability <= model.config.probability_tolerance:
                assert posterior is None
            else:
                assert posterior is not None
                assert np.all(np.isfinite(posterior))
                assert np.all(posterior >= -1e-14)
                assert posterior.sum() == pytest.approx(1.0, abs=1e-13)


def test_successful_message_reveals_reached_state() -> None:
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=0.25),
    )
    belief = np.array([0.4, 0.6])
    action = model.encode_action(0, np.array([1, 1]))

    assert model.posterior(belief, action, 0) == pytest.approx([1.0, 0.0])
    assert model.posterior(belief, action, 1) == pytest.approx([0.0, 1.0])


def test_no_transmission_anywhere_gives_predictive_null_posterior() -> None:
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=0.25),
    )
    belief = np.array([0.4, 0.6])
    action = model.encode_action(1, np.array([0, 0]))
    prediction = belief @ model.mdp.P[1]

    probabilities = model.observation_probabilities(belief, action)
    assert probabilities == pytest.approx([0.0, 0.0, 1.0])
    assert model.posterior(belief, action, model.null_observation) == pytest.approx(
        prediction
    )


def test_transmission_everywhere_has_state_independent_failure() -> None:
    epsilon = 0.25
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=epsilon),
    )
    belief = np.array([0.4, 0.6])
    action = model.encode_action(1, np.array([1, 1]))
    prediction = belief @ model.mdp.P[1]

    probabilities = model.observation_probabilities(belief, action)
    assert probabilities[model.null_observation] == pytest.approx(epsilon)
    assert model.posterior(belief, action, model.null_observation) == pytest.approx(
        prediction
    )


def test_perfect_channel_null_eliminates_transmitting_states() -> None:
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=0.0),
    )
    belief = np.array([0.4, 0.6])
    action = model.encode_action(0, np.array([1, 0]))

    posterior = model.posterior(belief, action, model.null_observation)
    assert posterior == pytest.approx([0.0, 1.0])


def test_state_dependent_null_observation_is_informative() -> None:
    epsilon = 0.2
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=epsilon),
    )
    belief = np.array([0.4, 0.6])
    action = model.encode_action(0, np.array([1, 0]))
    prediction = belief @ model.mdp.P[0]
    expected = prediction * np.array([epsilon, 1.0])
    expected /= expected.sum()

    posterior = model.posterior(belief, action, model.null_observation)
    assert posterior == pytest.approx(expected)
    assert not np.allclose(posterior, prediction)


def test_impossible_success_is_not_given_an_arbitrary_posterior() -> None:
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=0.2),
    )
    belief = np.array([0.4, 0.6])
    action = model.encode_action(0, np.array([0, 1]))

    assert model.posterior(belief, action, 0) is None


def test_reward_places_communication_cost_after_transition() -> None:
    mdp = two_state_mdp()
    gamma = 0.8
    beta = 0.3
    model = JPOModel(
        mdp,
        JPOConfig(gamma=gamma, beta=beta, epsilon=0.2),
    )
    prescription = np.array([1, 0])
    action = model.encode_action(1, prescription)
    expected = np.sum(
        mdp.P[1] * (mdp.R[1] - gamma * beta * prescription[None, :]),
        axis=1,
    )

    assert model.reward_vector(action) == pytest.approx(expected)


def test_solver_root_represents_average_of_dirac_values_not_uniform_belief() -> None:
    gamma = 0.8
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=gamma, beta=0.3, epsilon=0.2),
    )
    arrays = model.build_solver_arrays()
    continuation_values = np.array([2.0, 5.0, 123.0])
    root_continuation = arrays.transitions[
        arrays.initialization_action, arrays.dummy_state
    ] @ continuation_values

    assert arrays.initial_belief == pytest.approx([0.0, 0.0, 1.0])
    assert np.allclose(
        arrays.physical_initial_beliefs,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert arrays.physical_initial_weights == pytest.approx([0.5, 0.5])
    assert gamma * root_continuation == pytest.approx(
        gamma * np.mean(continuation_values[:2])
    )


def test_explicit_solver_arrays_are_normalized() -> None:
    model = JPOModel(
        two_state_mdp(),
        JPOConfig(gamma=0.8, beta=0.3, epsilon=0.2),
    )
    arrays = model.build_solver_arrays()

    assert np.allclose(arrays.transitions.sum(axis=2), 1.0)
    assert np.allclose(arrays.observations.sum(axis=1), 1.0)
    assert arrays.transitions.shape == (model.n_actions + 1, 3, 3)
    assert arrays.observations.shape == (
        model.n_actions + 1,
        model.n_observations,
        3,
    )
