"""Tests for the NativeSARSOP bridge that do not require Julia."""

from __future__ import annotations

import numpy as np
import pytest

from jpo_model import JPOConfig, JPOModel
from jpo_sarsop import (
    NativeSARSOPConfig,
    NativeSARSOPError,
    NativeSARSOPUnavailable,
    SARSOPPolicy,
    _read_float64,
    _write_solver_input,
    run_native_sarsop,
)
from mdp import FiniteMDP


def small_model() -> JPOModel:
    mdp = FiniteMDP(
        P=np.array([[[0.8, 0.2], [0.3, 0.7]]]),
        R=np.array([[[0.2, 0.9], [0.4, 0.1]]]),
    )
    return JPOModel(
        mdp,
        JPOConfig(gamma=0.8, beta=0.2, epsilon=0.1),
    )


def test_binary_solver_input_preserves_jpo_arrays(tmp_path) -> None:
    model = small_model()
    arrays = model.build_solver_arrays()
    config = NativeSARSOPConfig(
        max_time=1.0,
        max_steps=2,
        export_beliefs=False,
    )
    _write_solver_input(tmp_path, model, arrays, config)

    transitions = _read_float64(
        tmp_path / "transitions.bin",
        (model.n_states + 1, model.n_states + 1, model.n_actions + 1),
    )
    observations = _read_float64(
        tmp_path / "observations.bin",
        (model.n_observations, model.n_states + 1, model.n_actions + 1),
    )
    rewards = _read_float64(
        tmp_path / "rewards.bin",
        (model.n_states + 1, model.n_actions + 1),
    )

    assert transitions == pytest.approx(np.transpose(arrays.transitions, (1, 2, 0)))
    assert observations == pytest.approx(
        np.transpose(arrays.observations, (1, 2, 0))
    )
    assert rewards == pytest.approx(arrays.rewards.T)
    metadata = (tmp_path / "metadata.tsv").read_text(encoding="utf-8")
    assert "precision\t0.01" in metadata
    assert "use_binning\ttrue" in metadata
    assert "export_beliefs\tfalse" in metadata


def test_alpha_policy_uses_physical_belief_and_removes_solver_offset() -> None:
    model = small_model()
    policy = SARSOPPolicy(
        model=model,
        alpha_vectors=np.array(
            [
                [2.0, 0.0],
                [0.0, 3.0],
                [-100.0, -100.0],
            ]
        ),
        solver_action_map=np.array([1, 2]),
    )

    assert policy.action(np.array([1.0, 0.0])) == 0
    assert policy.action(np.array([0.0, 1.0])) == 1
    assert policy.lower_value(np.array([0.25, 0.75])) == pytest.approx(2.25)


def test_policy_rejects_synthetic_action_at_physical_belief() -> None:
    model = small_model()
    policy = SARSOPPolicy(
        model=model,
        alpha_vectors=np.array([[1.0], [1.0], [0.0]]),
        solver_action_map=np.array([0]),
    )

    with pytest.raises(NativeSARSOPError, match="synthetic"):
        policy.action(np.array([1.0, 0.0]))


def test_missing_julia_has_explicit_error(tmp_path) -> None:
    with pytest.raises(NativeSARSOPUnavailable, match="Julia executable"):
        run_native_sarsop(
            small_model(),
            NativeSARSOPConfig(max_time=1.0),
            tmp_path,
            julia_executable="definitely-not-a-julia-executable",
        )
