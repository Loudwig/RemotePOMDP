"""Finite control MDPs used by the remote-control simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteMDP:
    """A finite MDP with transition-dependent rewards.

    Arrays use the EffCom convention ``P[a, s, s_next]`` and
    ``R[a, s, s_next]``.
    """

    P: np.ndarray
    R: np.ndarray
    density: float | None = None
    seed: int | None = None
    optimal_state: int | None = None

    def __post_init__(self) -> None:
        p = np.asarray(self.P, dtype=float)
        r = np.asarray(self.R, dtype=float)
        if p.ndim != 3:
            raise ValueError(f"P must have shape (A, S, S), got {p.shape}")
        if p.shape[1] != p.shape[2]:
            raise ValueError(f"P must have square state axes, got {p.shape}")
        if r.shape != p.shape:
            raise ValueError(f"R must have shape {p.shape}, got {r.shape}")
        if p.shape[0] < 1 or p.shape[1] < 1:
            raise ValueError("The MDP must have at least one state and one action")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise ValueError("P and R must contain only finite values")
        if np.any(p < 0.0):
            raise ValueError("Transition probabilities must be nonnegative")
        if not np.allclose(p.sum(axis=2), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("Every P[a, s, :] row must sum to one")
        if np.any(r < 0.0) or np.any(r > 1.0):
            raise ValueError("Rewards must lie in [0, 1]")

        # Own the arrays so family members cannot accidentally share mutations.
        p = np.array(p, dtype=float, copy=True, order="C")
        r = np.array(r, dtype=float, copy=True, order="C")
        p.setflags(write=False)
        r.setflags(write=False)
        object.__setattr__(self, "P", p)
        object.__setattr__(self, "R", r)

    @property
    def n_actions(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.P.shape[1])

    @property
    def expected_rewards(self) -> np.ndarray:
        """Return E[r(s, a, s_next)] with shape (A, S)."""

        return np.sum(self.P * self.R, axis=2)


def initial_distribution(n_states: int, initial_state: int | None = None) -> np.ndarray:
    """Return uniform ``mu0`` or a point mass for debugging."""

    if n_states < 1:
        raise ValueError("n_states must be positive")
    if initial_state is None:
        return np.full(n_states, 1.0 / n_states)
    if not 0 <= initial_state < n_states:
        raise ValueError(f"initial_state must be in [0, {n_states})")
    mu0 = np.zeros(n_states)
    mu0[initial_state] = 1.0
    return mu0


def validate_initial_distribution(mu0: np.ndarray, n_states: int) -> np.ndarray:
    """Validate and return a private floating-point copy of ``mu0``."""

    out = np.asarray(mu0, dtype=float)
    if out.shape != (n_states,):
        raise ValueError(f"mu0 must have shape ({n_states},), got {out.shape}")
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError("mu0 must be finite and nonnegative")
    if not np.isclose(out.sum(), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("mu0 must sum to one")
    return np.array(out, dtype=float, copy=True)


def create_effcom_control_family(
    n_states: int = 10,
    n_actions: int = 2,
    reward_decay: float = 10.0,
    seed: int = 1234,
) -> list[FiniteMDP]:
    """Create the randomized control-MDP density family used by EffCom.

    The transition construction follows EffCom's control generator. Rewards
    are divided by its original scale of ten, yielding values in ``[0, 1]``.
    For ``n_states=10`` the returned densities are 0.1, 0.3, ..., 0.9.
    """

    if n_states < 2 or n_states % 2 != 0:
        raise ValueError("The EffCom family requires a positive even number of states")
    if n_actions < 1:
        raise ValueError("n_actions must be positive")
    if reward_decay < 0.0 or not np.isfinite(reward_decay):
        raise ValueError("reward_decay must be finite and nonnegative")

    # RandomState deliberately reproduces np.random.seed/randint from EffCom.
    rng = np.random.RandomState(seed)
    optimal_state = int(rng.randint(0, n_states))
    number_of_mdps = n_states // 2
    transitions = [np.zeros((n_actions, n_states, n_states)) for _ in range(number_of_mdps)]

    for action in range(n_actions):
        for state in range(n_states):
            center = int(rng.randint(0, n_states))
            transitions[0][action, state, center] = 1.0
            for family_index in range(1, number_of_mdps):
                offsets = range(-family_index, family_index + 1)
                support_size = 2 * family_index + 1
                for offset in offsets:
                    next_state = (center + offset) % n_states
                    transitions[family_index][action, state, next_state] = (
                        0.5 - abs(offset) / support_size
                    )
                transitions[family_index][action, state, center] += 0.1
                transitions[family_index][action, state] /= transitions[family_index][
                    action, state
                ].sum()

    next_states = np.arange(n_states)
    reward_by_next_state = np.exp(-np.abs(next_states - optimal_state) * reward_decay)
    rewards = np.broadcast_to(
        reward_by_next_state, (n_actions, n_states, n_states)
    ).copy()

    return [
        FiniteMDP(
            P=transition,
            R=rewards,
            density=(2 * family_index + 1) / n_states,
            seed=seed,
            optimal_state=optimal_state,
        )
        for family_index, transition in enumerate(transitions)
    ]


def select_density(family: list[FiniteMDP], density: float) -> FiniteMDP:
    """Select one member of an EffCom family by density."""

    for mdp in family:
        if mdp.density is not None and np.isclose(mdp.density, density, atol=1e-12, rtol=0.0):
            return mdp
    available = [mdp.density for mdp in family]
    raise ValueError(f"density {density} is unavailable; choose one of {available}")
