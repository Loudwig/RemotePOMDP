"""Small end-to-end NativeSARSOP regression test (skipped without Julia)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from jpo_model import JPOConfig, JPOModel
from jpo_policy import PolicyEvaluationConfig, evaluate_jpo_policy
from jpo_sarsop import NativeSARSOPConfig, run_native_sarsop
from mdp import FiniteMDP


@pytest.mark.skipif(shutil.which("julia") is None, reason="Julia is not installed")
def test_native_sarsop_end_to_end_bounds_and_policy_value(tmp_path) -> None:
    mdp = FiniteMDP(
        P=np.array([[[0.8, 0.2], [0.3, 0.7]]]),
        R=np.array([[[0.2, 0.9], [0.4, 0.1]]]),
    )
    model = JPOModel(
        mdp,
        JPOConfig(gamma=0.5, beta=0.2, epsilon=0.1),
    )
    training = run_native_sarsop(
        model,
        NativeSARSOPConfig(
            search_epsilon=0.1,
            precision=0.05,
            max_time=5.0,
            max_steps=1_000,
            initial_bound_residual=1e-8,
            initial_bound_max_time=5.0,
            export_beliefs=False,
        ),
        tmp_path,
    )
    evaluation = evaluate_jpo_policy(
        model,
        training.policy,
        PolicyEvaluationConfig(tail_interval_tolerance=1e-8),
    )

    assert training.lower_objective == pytest.approx(
        np.mean(training.lower_values)
    )
    assert training.upper_objective == pytest.approx(
        np.mean(training.upper_values)
    )
    assert all(
        point["root_lower"] <= point["root_upper"] + 1e-8
        for point in training.history
    )
    assert training.lower_objective <= evaluation.upper_objective + 1e-8
    assert evaluation.lower_objective <= training.upper_objective + 1e-8
    assert (tmp_path / "policy.npz").is_file()
    assert (tmp_path / "training.json").is_file()
    assert training.belief_points.shape == (model.n_states + 1, 0)
    assert not training.belief_metadata
    assert not (tmp_path / "native_output" / "belief_points.bin").exists()
    assert not (tmp_path / "native_output" / "belief_metadata.tsv").exists()
