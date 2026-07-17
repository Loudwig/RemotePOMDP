"""Tests for the separation of JPO workflow stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import jpo
from jpo_model import JPOConfig
from jpo_policy import PolicyAnalysisConfig, PolicyEvaluationConfig
from jpo_sarsop import SARSOPTrainingResult
from mdp import FiniteMDP


class ConstantPolicy:
    def __init__(self, action: int):
        self._action = action

    def action(self, _belief: np.ndarray) -> int:
        return self._action


def test_training_analysis_and_restriction_are_separate_stages(
    tmp_path, monkeypatch
) -> None:
    mdp = FiniteMDP(
        P=np.array([[[0.5, 0.5], [0.5, 0.5]]]),
        R=np.zeros((1, 2, 2)),
    )

    def fake_training(model, _config, output_directory, julia_executable):
        del julia_executable
        policy = ConstantPolicy(model.encode_action(0, np.array([1, 0])))
        return SARSOPTrainingResult(
            policy=policy,
            lower_values=np.array([-10.0, -10.0]),
            upper_values=np.array([10.0, 10.0]),
            lower_objective=-10.0,
            upper_objective=10.0,
            gap=20.0,
            root_lower_objective=-10.0,
            root_upper_objective=10.0,
            root_gap=20.0,
            iterations=1,
            elapsed_seconds=0.0,
            stop_reason="test",
            history=[],
            belief_points=np.empty((3, 0)),
            belief_metadata=[],
            corner_upper=np.zeros(3),
            output_directory=Path(output_directory),
            diagnostics={"unrestricted": True},
        )

    monkeypatch.setattr(jpo, "run_native_sarsop", fake_training)
    result = jpo.run_jpo(
        mdp,
        JPOConfig(gamma=0.5, beta=0.1, epsilon=0.2),
        analysis_config=PolicyAnalysisConfig(
            discounted_tail_tolerance=1e-4, max_depth=30
        ),
        evaluation_config=PolicyEvaluationConfig(
            tail_interval_tolerance=1e-5
        ),
        output_directory=tmp_path,
    )

    assert result.analysis.violations
    assert result.restricted_policy is not None
    assert result.restricted_evaluation is not None
    assert result.diagnostics["stages"]["1_unrestricted_training"] == {
        "unrestricted": True
    }
    assert (tmp_path / "result.json").exists()
