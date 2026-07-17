"""JPO POMDP model built on the repository's existing :class:`FiniteMDP`.

The physical state transition and reward arrays are never redefined here:
``FiniteMDP`` remains their single source of truth.  This module adds only the
transformed JPO actions, channel observations, belief update, and the compact
array export required by an external POMDP solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mdp import FiniteMDP


@dataclass(frozen=True)
class JPOConfig:
    gamma: float
    beta: float
    epsilon: float
    probability_tolerance: float = 1e-14

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if self.beta < 0.0 or not np.isfinite(self.beta):
            raise ValueError("beta must be finite and nonnegative")
        if not 0.0 <= self.epsilon <= 1.0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must lie in [0, 1]")
        if self.probability_tolerance < 0.0 or not np.isfinite(
            self.probability_tolerance
        ):
            raise ValueError(
                "probability_tolerance must be finite and nonnegative"
            )


@dataclass(frozen=True)
class DecodedJPOAction:
    receiver_action: int
    prescription_index: int
    prescription: np.ndarray


@dataclass(frozen=True)
class JPOSolverArrays:
    """Explicit arrays sent to NativeSARSOP, including one synthetic root.

    Solver action zero is the forced initialization action.  Solver action
    ``1 + a`` corresponds to JPO action ``a``.  The last state is a synthetic
    root which samples a physical state and reveals it through the forced
    initialization action.  Consequently, its value is ``gamma * J``.
    """

    transitions: np.ndarray
    observations: np.ndarray
    rewards: np.ndarray
    initial_belief: np.ndarray
    physical_initial_beliefs: np.ndarray
    physical_initial_weights: np.ndarray
    dummy_state: int
    initialization_action: int
    jpo_action_offset: int
    forbidden_action_reward: float


class JPOModel:
    """Transformed JPO POMDP whose physical dynamics come from ``FiniteMDP``."""

    def __init__(self, mdp: FiniteMDP, config: JPOConfig):
        self.mdp = mdp
        self.config = config
        self.n_states = mdp.n_states
        self.n_receiver_actions = mdp.n_actions
        self.n_prescriptions = 1 << self.n_states
        self.n_actions = self.n_prescriptions * self.n_receiver_actions
        self.null_observation = self.n_states
        self.n_observations = self.n_states + 1

        # EffCom ordering: the prescription is the major index, receiver action
        # the minor index.  In the zero-padded binary representation, the bit
        # for physical state zero is the most significant bit.
        shifts = np.arange(self.n_states - 1, -1, -1, dtype=np.int64)
        masks = np.arange(self.n_prescriptions, dtype=np.int64)[:, None]
        prescriptions = ((masks >> shifts[None, :]) & 1).astype(np.int8)
        prescriptions.setflags(write=False)
        self.prescriptions = prescriptions

    def encode_action(
        self, receiver_action: int, prescription: int | np.ndarray
    ) -> int:
        if int(receiver_action) != receiver_action or not (
            0 <= receiver_action < self.n_receiver_actions
        ):
            raise ValueError(
                "receiver_action must be a valid FiniteMDP action index"
            )
        if np.isscalar(prescription):
            prescription_index = int(prescription)
            if not 0 <= prescription_index < self.n_prescriptions:
                raise ValueError(
                    f"prescription index must lie in [0, {self.n_prescriptions})"
                )
        else:
            vector = np.asarray(prescription)
            if vector.shape != (self.n_states,):
                raise ValueError(
                    f"prescription must have shape ({self.n_states},)"
                )
            if np.any((vector != 0) & (vector != 1)):
                raise ValueError("prescription entries must be zero or one")
            powers = 1 << np.arange(self.n_states - 1, -1, -1)
            prescription_index = int(np.dot(vector.astype(np.int64), powers))
        return prescription_index * self.n_receiver_actions + int(receiver_action)

    def decode_action(self, action: int) -> DecodedJPOAction:
        self._validate_action(action)
        receiver_action = int(action % self.n_receiver_actions)
        prescription_index = int(action // self.n_receiver_actions)
        return DecodedJPOAction(
            receiver_action=receiver_action,
            prescription_index=prescription_index,
            prescription=self.prescriptions[prescription_index],
        )

    def transition_matrix(self, action: int) -> np.ndarray:
        """Return the existing physical matrix ``mdp.P[a_rx]`` directly."""

        decoded = self.decode_action(action)
        return self.mdp.P[decoded.receiver_action]

    def reward_vector(self, action: int) -> np.ndarray:
        """Expected stage reward for every current physical state.

        With the required control-transition-communication timing, the reward
        for ``z -> s`` is ``R[a_rx,z,s] - gamma*beta*g(s)``.  The communication
        cost is discounted because the prescription is applied after the
        physical transition.
        """

        decoded = self.decode_action(action)
        transition = self.mdp.P[decoded.receiver_action]
        transition_reward = self.mdp.R[decoded.receiver_action] - (
            self.config.gamma
            * self.config.beta
            * decoded.prescription[None, :]
        )
        return np.sum(transition * transition_reward, axis=1)

    def observation_kernel(self, action: int) -> np.ndarray:
        """Return ``O[o, s_next]`` for one transformed action."""

        decoded = self.decode_action(action)
        kernel = np.zeros((self.n_observations, self.n_states), dtype=float)
        success = (1.0 - self.config.epsilon) * decoded.prescription
        states = np.arange(self.n_states)
        kernel[states, states] = success
        kernel[self.null_observation] = 1.0 - success
        return kernel

    def predictive_belief(self, belief: np.ndarray, action: int) -> np.ndarray:
        belief = self.validate_belief(belief)
        return belief @ self.transition_matrix(action)

    def observation_probabilities(
        self, belief: np.ndarray, action: int
    ) -> np.ndarray:
        prediction = self.predictive_belief(belief, action)
        decoded = self.decode_action(action)
        probabilities = np.zeros(self.n_observations, dtype=float)
        probabilities[: self.n_states] = (
            (1.0 - self.config.epsilon)
            * decoded.prescription
            * prediction
        )
        probabilities[self.null_observation] = float(
            np.dot(
                prediction,
                1.0
                - decoded.prescription
                + self.config.epsilon * decoded.prescription,
            )
        )
        probabilities[np.abs(probabilities) < 1e-15] = 0.0
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=1e-12):
            raise FloatingPointError("JPO observation probabilities do not sum to one")
        return probabilities

    def posterior(
        self, belief: np.ndarray, action: int, observation: int
    ) -> np.ndarray | None:
        if int(observation) != observation or not (
            0 <= observation < self.n_observations
        ):
            raise ValueError(
                f"observation must lie in [0, {self.n_observations})"
            )
        probabilities = self.observation_probabilities(belief, action)
        probability = float(probabilities[observation])
        if probability <= self.config.probability_tolerance:
            return None
        if observation < self.n_states:
            posterior = np.zeros(self.n_states, dtype=float)
            posterior[observation] = 1.0
            return posterior

        prediction = self.predictive_belief(belief, action)
        prescription = self.decode_action(action).prescription
        likelihood = 1.0 - prescription + self.config.epsilon * prescription
        unnormalized = prediction * likelihood
        denominator = float(unnormalized.sum())
        if denominator <= self.config.probability_tolerance:
            return None
        posterior = unnormalized / denominator
        self._check_posterior(posterior)
        return posterior

    def expected_reward(self, belief: np.ndarray, action: int) -> float:
        belief = self.validate_belief(belief)
        return float(np.dot(belief, self.reward_vector(action)))

    def initial_beliefs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all physical Dirac beliefs and their uniform weights."""

        beliefs = np.eye(self.n_states, dtype=float)
        weights = np.full(self.n_states, 1.0 / self.n_states)
        return beliefs, weights

    def build_solver_arrays(self) -> JPOSolverArrays:
        """Build the explicit single-root problem consumed by NativeSARSOP."""

        physical_states = self.n_states
        solver_states = physical_states + 1
        dummy_state = physical_states
        solver_actions = self.n_actions + 1
        initialization_action = 0
        jpo_action_offset = 1

        transitions = np.zeros(
            (solver_actions, solver_states, solver_states), dtype=float
        )
        observations = np.zeros(
            (solver_actions, self.n_observations, solver_states), dtype=float
        )
        rewards = np.empty((solver_actions, solver_states), dtype=float)

        # A penalty which dominates any possible gain from using the synthetic
        # action outside the root, or delaying initialization at the root.
        penalty_magnitude = 1.0 + 1.0 / (1.0 - self.config.gamma)
        forbidden_reward = -penalty_magnitude

        # Forced initialization: sample S0 uniformly and reveal it.  If this
        # action were selected later it would leave the physical state fixed;
        # the dominating penalty makes that branch strictly suboptimal.
        transitions[initialization_action, dummy_state, :physical_states] = (
            1.0 / physical_states
        )
        states = np.arange(physical_states)
        transitions[initialization_action, states, states] = 1.0
        observations[initialization_action, states, states] = 1.0
        observations[
            initialization_action, self.null_observation, dummy_state
        ] = 1.0
        rewards[initialization_action, :physical_states] = forbidden_reward
        rewards[initialization_action, dummy_state] = 0.0

        for action in range(self.n_actions):
            solver_action = action + jpo_action_offset
            transitions[
                solver_action, :physical_states, :physical_states
            ] = self.transition_matrix(action)
            transitions[solver_action, dummy_state, dummy_state] = 1.0
            observations[
                solver_action, :, :physical_states
            ] = self.observation_kernel(action)
            observations[
                solver_action, self.null_observation, dummy_state
            ] = 1.0
            rewards[solver_action, :physical_states] = self.reward_vector(action)
            rewards[solver_action, dummy_state] = forbidden_reward

        initial_belief = np.zeros(solver_states, dtype=float)
        initial_belief[dummy_state] = 1.0
        physical_beliefs, weights = self.initial_beliefs()
        augmented_beliefs = np.pad(physical_beliefs, ((0, 0), (0, 1)))

        if not np.allclose(transitions.sum(axis=2), 1.0, atol=1e-12, rtol=1e-12):
            raise FloatingPointError("solver transition rows do not sum to one")
        if not np.allclose(observations.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
            raise FloatingPointError(
                "solver observation probabilities do not sum to one"
            )

        return JPOSolverArrays(
            transitions=transitions,
            observations=observations,
            rewards=rewards,
            initial_belief=initial_belief,
            physical_initial_beliefs=augmented_beliefs,
            physical_initial_weights=weights,
            dummy_state=dummy_state,
            initialization_action=initialization_action,
            jpo_action_offset=jpo_action_offset,
            forbidden_action_reward=forbidden_reward,
        )

    def validate_belief(self, belief: np.ndarray) -> np.ndarray:
        out = np.asarray(belief, dtype=float)
        if out.shape != (self.n_states,):
            raise ValueError(f"belief must have shape ({self.n_states},)")
        if not np.all(np.isfinite(out)) or np.any(out < -1e-14):
            raise ValueError("belief must be finite and nonnegative")
        if not np.isclose(out.sum(), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("belief must sum to one")
        out = np.maximum(out, 0.0)
        return out / out.sum()

    def _validate_action(self, action: int) -> None:
        if int(action) != action or not 0 <= action < self.n_actions:
            raise ValueError(f"action must lie in [0, {self.n_actions})")

    @staticmethod
    def _check_posterior(posterior: np.ndarray) -> None:
        if not np.all(np.isfinite(posterior)):
            raise FloatingPointError("posterior contains non-finite entries")
        if np.any(posterior < -1e-14):
            raise FloatingPointError("posterior contains a negative entry")
        if not np.isclose(posterior.sum(), 1.0, atol=1e-12, rtol=1e-12):
            raise FloatingPointError("posterior does not sum to one")
