"""Python bridge from :mod:`jpo_model` to Julia NativeSARSOP.

EffCom delegates POMDP optimization to NativeSARSOP; this module follows the
same architecture while retaining all experiment inputs and solver outputs in
the local run directory.  Julia is invoked as a subprocess so the Python API
does not depend on a fragile process-global PyJulia configuration.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from jpo_model import JPOModel, JPOSolverArrays


class NativeSARSOPUnavailable(RuntimeError):
    """Raised when the configured Julia executable cannot be found."""


class NativeSARSOPError(RuntimeError):
    """Raised when the Julia solver fails or returns inconsistent bounds."""


@dataclass(frozen=True)
class NativeSARSOPConfig:
    """NativeSARSOP settings, using the values chosen by EffCom by default."""

    search_epsilon: float = 0.01
    precision: float = 0.01
    kappa: float = 0.5
    delta: float = 0.0001
    max_time: float = 300.0
    max_steps: int = 1_000_000
    prune_threshold: float = 0.10
    use_binning: bool = True
    initial_bound_residual: float = 1e-8
    initial_bound_max_time: float = 30.0
    initial_upper_bound: str = "fully_observable"
    export_beliefs: bool = True

    def __post_init__(self) -> None:
        for name in (
            "search_epsilon",
            "precision",
            "kappa",
            "delta",
            "max_time",
            "prune_threshold",
            "initial_bound_residual",
            "initial_bound_max_time",
        ):
            value = getattr(self, name)
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.precision == 0.0:
            raise ValueError("precision must be positive")
        if self.max_time == 0.0:
            raise ValueError("max_time must be positive")
        if self.initial_bound_residual == 0.0:
            raise ValueError("initial_bound_residual must be positive")
        if self.initial_bound_max_time == 0.0:
            raise ValueError("initial_bound_max_time must be positive")
        if self.initial_bound_max_time < 1e-3:
            raise ValueError("initial_bound_max_time must be at least 1e-3 seconds")
        if self.initial_upper_bound not in ("fully_observable", "fib"):
            raise ValueError(
                "initial_upper_bound must be 'fully_observable' or 'fib'"
            )
        if int(self.max_steps) != self.max_steps or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")


@dataclass
class SARSOPPolicy:
    """Alpha-vector policy returned by NativeSARSOP."""

    model: JPOModel
    alpha_vectors: np.ndarray  # augmented_state, alpha
    solver_action_map: np.ndarray  # zero-based, includes synthetic action zero
    jpo_action_offset: int = 1

    def __post_init__(self) -> None:
        alphas = np.asarray(self.alpha_vectors, dtype=float)
        actions = np.asarray(self.solver_action_map)
        expected_states = self.model.n_states + 1
        if alphas.ndim != 2 or alphas.shape[0] != expected_states:
            raise ValueError(
                f"alpha_vectors must have shape ({expected_states}, K)"
            )
        if actions.shape != (alphas.shape[1],):
            raise ValueError("solver_action_map must contain one action per alpha")
        if not np.all(np.isfinite(alphas)):
            raise ValueError("alpha_vectors must be finite")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("solver_action_map must be integral")
        self.alpha_vectors = alphas
        self.solver_action_map = actions.astype(np.int64, copy=False)

    def alpha_index(self, belief: np.ndarray) -> int:
        physical = self.model.validate_belief(belief)
        augmented = np.append(physical, 0.0)
        return int(np.argmax(augmented @ self.alpha_vectors))

    def lower_value(self, belief: np.ndarray) -> float:
        physical = self.model.validate_belief(belief)
        augmented = np.append(physical, 0.0)
        return float(np.max(augmented @ self.alpha_vectors))

    def action(self, belief: np.ndarray) -> int:
        alpha_index = self.alpha_index(belief)
        solver_action = int(self.solver_action_map[alpha_index])
        if solver_action < self.jpo_action_offset:
            raise NativeSARSOPError(
                "NativeSARSOP selected the synthetic initialization action at "
                "a physical belief"
            )
        action = solver_action - self.jpo_action_offset
        self.model.decode_action(action)
        return action


@dataclass
class SARSOPTrainingResult:
    policy: SARSOPPolicy
    lower_values: np.ndarray
    upper_values: np.ndarray
    lower_objective: float
    upper_objective: float
    gap: float
    root_lower_objective: float
    root_upper_objective: float
    root_gap: float
    iterations: int
    elapsed_seconds: float
    stop_reason: str
    history: list[dict[str, float]]
    belief_points: np.ndarray
    belief_metadata: list[dict[str, Any]]
    corner_upper: np.ndarray
    output_directory: Path
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_native_sarsop(
    model: JPOModel,
    config: NativeSARSOPConfig | None = None,
    output_directory: str | Path = "jpo_run",
    julia_executable: str = "julia",
) -> SARSOPTrainingResult:
    """Solve one JPO POMDP and retain a reproducible training artifact."""

    solver_config = config or NativeSARSOPConfig()
    julia_path = shutil.which(julia_executable)
    if julia_path is None:
        raise NativeSARSOPUnavailable(
            f"Julia executable {julia_executable!r} was not found; install Julia "
            "and instantiate the pinned julia/Project.toml environment"
        )

    root = Path(output_directory).expanduser().resolve()
    input_directory = root / "native_input"
    native_output = root / "native_output"
    input_directory.mkdir(parents=True, exist_ok=True)
    native_output.mkdir(parents=True, exist_ok=True)

    arrays = model.build_solver_arrays()
    _write_solver_input(input_directory, model, arrays, solver_config)
    np.savez_compressed(
        root / "model.npz",
        P=model.mdp.P,
        R=model.mdp.R,
        gamma=model.config.gamma,
        beta=model.config.beta,
        epsilon=model.config.epsilon,
        initial_beliefs=arrays.physical_initial_beliefs[:, : model.n_states],
        initial_weights=arrays.physical_initial_weights,
    )

    julia_directory = Path(__file__).resolve().parent / "julia"
    script = julia_directory / "solve_jpo.jl"
    command = [
        julia_path,
        "--startup-file=no",
        f"--project={julia_directory}",
        str(script),
        str(input_directory),
        str(native_output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    (root / "solver_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (root / "solver_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise NativeSARSOPError(
            "NativeSARSOP failed with exit code "
            f"{completed.returncode}; see {root / 'solver_stderr.txt'}"
        )

    result = _load_solver_result(model, arrays, solver_config, root, native_output)
    np.savez_compressed(
        root / "policy.npz",
        alpha_vectors=result.policy.alpha_vectors,
        solver_action_map=result.policy.solver_action_map,
        jpo_action_offset=result.policy.jpo_action_offset,
    )
    _write_json(root / "training.json", result.diagnostics)
    return result


def load_sarsop_policy(model: JPOModel, path: str | Path) -> SARSOPPolicy:
    """Load an exported alpha-vector policy without starting Julia."""

    with np.load(Path(path), allow_pickle=False) as payload:
        offset = int(np.asarray(payload["jpo_action_offset"]).item())
        return SARSOPPolicy(
            model=model,
            alpha_vectors=payload["alpha_vectors"],
            solver_action_map=payload["solver_action_map"],
            jpo_action_offset=offset,
        )


def _write_solver_input(
    directory: Path,
    model: JPOModel,
    arrays: JPOSolverArrays,
    config: NativeSARSOPConfig,
) -> None:
    transitions = np.transpose(arrays.transitions, (1, 2, 0))
    observations = np.transpose(arrays.observations, (1, 2, 0))
    rewards = arrays.rewards.T
    _write_float64(directory / "transitions.bin", transitions)
    _write_float64(directory / "observations.bin", observations)
    _write_float64(directory / "rewards.bin", rewards)
    _write_float64(directory / "initial_belief.bin", arrays.initial_belief)

    metadata = {
        "n_states": transitions.shape[0],
        "n_physical_states": model.n_states,
        "n_actions": transitions.shape[2],
        "n_observations": observations.shape[0],
        "gamma": model.config.gamma,
        **asdict(config),
    }
    lines = [f"{key}\t{_metadata_value(value)}\n" for key, value in metadata.items()]
    (directory / "metadata.tsv").write_text("".join(lines), encoding="utf-8")


def _load_solver_result(
    model: JPOModel,
    arrays: JPOSolverArrays,
    config: NativeSARSOPConfig,
    root: Path,
    native_output: Path,
) -> SARSOPTrainingResult:
    summary = _read_key_value_file(native_output / "result.tsv")
    alpha_count = int(summary["alpha_count"])
    belief_count = int(summary["belief_count"])
    solver_states = model.n_states + 1

    alpha_vectors = _read_float64(
        native_output / "alpha_vectors.bin", (solver_states, alpha_count)
    )
    action_map = _read_int64(native_output / "action_map.bin", (alpha_count,))
    policy = SARSOPPolicy(
        model=model,
        alpha_vectors=alpha_vectors,
        solver_action_map=action_map,
        jpo_action_offset=arrays.jpo_action_offset,
    )
    lower_values = _read_float64(
        native_output / "dirac_lower.bin", (model.n_states,)
    )
    upper_values = _read_float64(
        native_output / "dirac_upper.bin", (model.n_states,)
    )
    if not np.all(np.isfinite(lower_values)) or not np.all(np.isfinite(upper_values)):
        raise NativeSARSOPError("NativeSARSOP returned non-finite initial bounds")
    if np.any(lower_values > upper_values + 1e-8):
        raise NativeSARSOPError("NativeSARSOP returned L(delta_s) > U(delta_s)")
    weights = arrays.physical_initial_weights
    lower_objective = float(np.dot(weights, lower_values))
    upper_objective = float(np.dot(weights, upper_values))
    gap = upper_objective - lower_objective

    root_lower = float(summary["root_lower"]) / model.config.gamma
    root_upper = float(summary["root_upper"]) / model.config.gamma
    root_gap = float(summary["root_gap"]) / model.config.gamma
    if root_lower > root_upper + 1e-8 or gap < -1e-8:
        raise NativeSARSOPError("NativeSARSOP returned inconsistent weighted bounds")

    history = _read_history(native_output / "history.tsv", model.config.gamma)
    if any(
        point["root_lower"] > point["root_upper"] + 1e-8
        for point in history
    ):
        raise NativeSARSOPError(
            "NativeSARSOP returned an inconsistent bound in its training history"
        )
    if config.export_beliefs:
        belief_points = _read_float64(
            native_output / "belief_points.bin", (solver_states, belief_count)
        )
        belief_metadata = _read_belief_metadata(
            native_output / "belief_metadata.tsv"
        )
    else:
        belief_points = np.empty((solver_states, 0), dtype=float)
        belief_metadata = []
    corner_upper = _read_float64(
        native_output / "corner_upper.bin", (solver_states,)
    )

    diagnostics: dict[str, Any] = {
        "model": {
            "n_states": model.n_states,
            "n_receiver_actions": model.n_receiver_actions,
            "n_prescriptions": model.n_prescriptions,
            "n_jpo_actions": model.n_actions,
            "gamma": model.config.gamma,
            "beta": model.config.beta,
            "epsilon": model.config.epsilon,
            "mdp_density": model.mdp.density,
            "mdp_seed": model.mdp.seed,
            "mdp_sha256": _mdp_digest(model),
        },
        "action_encoding": {
            "formula": "action = prescription_index * n_receiver_actions + receiver_action",
            "state_zero_bit": "most_significant",
            "solver_jpo_action_offset": arrays.jpo_action_offset,
        },
        "initialization": {
            "kind": "single synthetic root revealing a uniformly sampled physical state",
            "initial_beliefs": np.eye(model.n_states).tolist(),
            "initial_weights": weights.tolist(),
            "root_value_scale": f"V_root / gamma, gamma={model.config.gamma}",
        },
        "solver": asdict(config),
        "runtime": {
            "julia": summary["julia_version"],
            "native_sarsop": summary["native_sarsop_version"],
            "pomdps": summary["pomdps_version"],
            "pomdp_tools": summary["pomdp_tools_version"],
        },
        "bounds": {
            "lower_values": lower_values.tolist(),
            "upper_values": upper_values.tolist(),
            "lower_objective": lower_objective,
            "upper_objective": upper_objective,
            "gap": max(0.0, gap),
            "root_lower_objective": root_lower,
            "root_upper_objective": root_upper,
            "root_gap": max(0.0, root_gap),
        },
        "training": {
            "iterations": int(summary["iterations"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
            "initialization_seconds": float(summary["initialization_seconds"]),
            "initial_lower_residual": float(summary["initial_lower_residual"]),
            "initial_upper_residual": float(summary["initial_upper_residual"]),
            "initial_upper_iterations": int(summary["initial_upper_iterations"]),
            "initial_upper_bound": summary["initial_upper_bound"],
            "initial_lower_subsolution_shift": float(
                summary["initial_lower_subsolution_shift"]
            ),
            "stop_reason": summary["stop_reason"],
            "alpha_count": alpha_count,
            "belief_count": belief_count,
            "history": history,
        },
    }
    return SARSOPTrainingResult(
        policy=policy,
        lower_values=lower_values,
        upper_values=upper_values,
        lower_objective=lower_objective,
        upper_objective=upper_objective,
        gap=max(0.0, gap),
        root_lower_objective=root_lower,
        root_upper_objective=root_upper,
        root_gap=max(0.0, root_gap),
        iterations=int(summary["iterations"]),
        elapsed_seconds=float(summary["elapsed_seconds"]),
        stop_reason=summary["stop_reason"],
        history=history,
        belief_points=belief_points,
        belief_metadata=belief_metadata,
        corner_upper=corner_upper,
        output_directory=root,
        diagnostics=diagnostics,
    )


def _write_float64(path: Path, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64, order="F")
    array.ravel(order="F").tofile(path)


def _read_float64(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float64)
    if values.size != int(np.prod(shape)):
        raise NativeSARSOPError(
            f"{path} contains {values.size} values, expected {int(np.prod(shape))}"
        )
    return values.reshape(shape, order="F")


def _read_int64(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.int64)
    if values.size != int(np.prod(shape)):
        raise NativeSARSOPError(
            f"{path} contains {values.size} values, expected {int(np.prod(shape))}"
        )
    return values.reshape(shape, order="F")


def _metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _read_key_value_file(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        output[key] = value
    return output


def _read_history(path: Path, gamma: float) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            root_lower = float(row["root_lower"])
            root_upper = float(row["root_upper"])
            history.append(
                {
                    "iteration": int(float(row["iteration"])),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "root_lower": root_lower,
                    "root_upper": root_upper,
                    "root_gap": float(row["root_gap"]),
                    "initial_lower_objective": root_lower / gamma,
                    "initial_upper_objective": root_upper / gamma,
                    "initial_gap": float(row["root_gap"]) / gamma,
                }
            )
    return history


def _read_belief_metadata(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            output.append(
                {
                    "index": int(row["index"]),
                    "lower": float(row["lower"]),
                    "upper": float(row["upper"]),
                    "pruned": row["pruned"] == "true",
                    "real": row["real"] == "true",
                    "terminal": row["terminal"] == "true",
                }
            )
    return output


def _mdp_digest(model: JPOModel) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(model.mdp.P).tobytes())
    digest.update(np.ascontiguousarray(model.mdp.R).tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
