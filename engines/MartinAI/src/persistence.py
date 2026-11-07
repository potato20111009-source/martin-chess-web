"""Atomic model, optimizer, replay buffer, and training-state persistence."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer

from .encoding import ACTION_SIZE, BASE_BOARD_PLANES, BOARD_PLANES
from .network import ChessNetwork
from .training import TrainingExample

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINTS_DIR / "latest_model.pt"
STATE_PATH = CHECKPOINTS_DIR / "training_state.json"
ARCHIVED_MODELS_DIR = PROJECT_ROOT / "archived_models"


def load_training_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {
            "iteration": 0,
            "loss_history": [],
        }
    with path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    required = ("iteration", "loss_history")
    if any(key not in state for key in required):
        raise ValueError(f"Training state file is missing required keys: {path}")
    return state


def load_checkpoint(
    model: ChessNetwork,
    optimizer: Optimizer,
    device: torch.device,
    path: Path = MODEL_PATH,
    metadata: dict[str, Any] | None = None,
) -> list[TrainingExample]:
    if not path.exists():
        return []
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid model checkpoint: {path}")
    if metadata is not None:
        metadata.update(
            saved_iteration=int(checkpoint.get("iteration", 0)),
            has_iteration="iteration" in checkpoint,
        )
    saved_model_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    first_conv_key = "backbone.0.weight"
    migrated_history_planes = False
    if (
        first_conv_key in saved_model_state
        and first_conv_key in model_state
        and saved_model_state[first_conv_key].shape[1] == BASE_BOARD_PLANES
        and model_state[first_conv_key].shape[1] == BOARD_PLANES
    ):
        expanded_weight = torch.zeros_like(model_state[first_conv_key])
        expanded_weight[:, :BASE_BOARD_PLANES] = saved_model_state[first_conv_key]
        saved_model_state = dict(saved_model_state)
        saved_model_state[first_conv_key] = expanded_weight
        migrated_history_planes = True
        print(
            "Migrated legacy 18-plane checkpoint to history-aware inputs; "
            "new history channels start with zero weights."
        )
    model.load_state_dict(saved_model_state)
    if "optimizer_state_dict" in checkpoint and not migrated_history_planes:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif migrated_history_planes:
        print("Optimizer state reset because the model input layer changed.")
    examples: list[TrainingExample] = []
    for replay_entry in checkpoint.get("replay_buffer", []):
        if len(replay_entry) == 3:
            board, policy, value = replay_entry
            example_iteration = 0
        elif len(replay_entry) == 4:
            board, policy, value, example_iteration = replay_entry
        else:
            raise ValueError("Checkpoint contains a replay entry with an invalid format.")
        board_array = np.asarray(board, dtype=np.float32)
        if isinstance(policy, (tuple, list)) and len(policy) == 2:
            action_indices = np.asarray(policy[0], dtype=np.int32)
            probabilities = np.asarray(policy[1], dtype=np.float32)
        else:
            dense_policy = np.asarray(policy, dtype=np.float32)
            if dense_policy.shape != (ACTION_SIZE,):
                raise ValueError("Checkpoint contains a replay entry with an invalid policy.")
            action_indices = np.flatnonzero(dense_policy).astype(np.int32)
            probabilities = dense_policy[action_indices]
        if board_array.shape == (BASE_BOARD_PLANES, 8, 8):
            board_array = np.concatenate(
                (
                    board_array,
                    np.zeros(
                        (BOARD_PLANES - BASE_BOARD_PLANES, 8, 8),
                        dtype=np.float32,
                    ),
                ),
                axis=0,
            )
        if (
            board_array.shape != (BOARD_PLANES, 8, 8)
            or action_indices.ndim != 1
            or probabilities.shape != action_indices.shape
            or len(action_indices) == 0
            or np.any(action_indices < 0)
            or np.any(action_indices >= ACTION_SIZE)
            or np.any(probabilities < 0)
            or not np.isfinite(probabilities).all()
        ):
            raise ValueError("Checkpoint contains a replay entry with an invalid shape.")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-4):
            raise ValueError("Checkpoint contains a replay policy that is not normalized.")
        examples.append(
            TrainingExample(
                board_array,
                (action_indices, probabilities),
                float(value),
                int(example_iteration),
            )
        )
    return examples


def load_model_weights(
    model: ChessNetwork,
    device: torch.device,
    path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Load model weights and return checkpoint metadata without restoring training state."""
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid model checkpoint: {path}")

    saved_model_state = checkpoint["model_state_dict"]
    model_state = model.state_dict()
    first_conv_key = "backbone.0.weight"
    if (
        first_conv_key in saved_model_state
        and first_conv_key in model_state
        and saved_model_state[first_conv_key].shape[1] == BASE_BOARD_PLANES
        and model_state[first_conv_key].shape[1] == BOARD_PLANES
    ):
        expanded_weight = torch.zeros_like(model_state[first_conv_key])
        expanded_weight[:, :BASE_BOARD_PLANES] = saved_model_state[first_conv_key]
        saved_model_state = dict(saved_model_state)
        saved_model_state[first_conv_key] = expanded_weight

    model.load_state_dict(saved_model_state)
    model.to(device)
    model.eval()
    return {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }


def load_archived_model(
    path: Path,
    device: torch.device,
) -> tuple[ChessNetwork, dict[str, Any]]:
    """Load an archived network using its saved width and residual-block count."""
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid model checkpoint: {path}")

    saved_model_state = checkpoint["model_state_dict"]
    first_conv_key = "backbone.0.weight"
    if first_conv_key not in saved_model_state:
        raise ValueError(f"Model checkpoint is missing {first_conv_key}: {path}")
    channels = int(saved_model_state[first_conv_key].shape[0])
    block_indices = {
        int(match.group(1))
        for key in saved_model_state
        if (match := re.match(r"backbone\.(\d+)\.layers\.0\.weight$", key))
    }
    model = ChessNetwork(channels=channels, residual_blocks=len(block_indices))
    model_state = model.state_dict()
    if (
        saved_model_state[first_conv_key].shape[1] == BASE_BOARD_PLANES
        and model_state[first_conv_key].shape[1] == BOARD_PLANES
    ):
        expanded_weight = torch.zeros_like(model_state[first_conv_key])
        expanded_weight[:, :BASE_BOARD_PLANES] = saved_model_state[first_conv_key]
        saved_model_state = dict(saved_model_state)
        saved_model_state[first_conv_key] = expanded_weight
    model.load_state_dict(saved_model_state)
    model.to(device)
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    return model, metadata


def clear_checkpoint_replay_buffer(path: Path = MODEL_PATH) -> int:
    """Remove replay examples from a checkpoint without changing model or optimizer."""
    if not path.exists():
        return 0
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid model checkpoint; cannot clear replay buffer: {path}")
    removed = len(checkpoint.get("replay_buffer", []))
    checkpoint["replay_buffer"] = []
    _atomic_write(path, lambda file: torch.save(checkpoint, file))
    return removed


def _atomic_write(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = temporary.name
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def save_checkpoint(
    model: ChessNetwork,
    optimizer: Optimizer,
    replay_buffer: list[TrainingExample],
    path: Path = MODEL_PATH,
    iteration: int = 0,
) -> None:
    if iteration < 0:
        raise ValueError("Checkpoint iteration must be non-negative.")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "replay_buffer": [
            (item.board, item.policy, item.value, item.iteration)
            for item in replay_buffer
        ],
    }
    _atomic_write(path, lambda file: torch.save(payload, file))


def save_model_archive(
    model: ChessNetwork,
    iteration: int,
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save compact model weights for historical evaluation or rollback."""
    if iteration < 0:
        raise ValueError("Archive iteration must be non-negative.")
    archive_path = ARCHIVED_MODELS_DIR / (
        filename or f"model_iter_{iteration:03d}.pt"
    )
    payload: dict[str, Any] = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "iteration": iteration,
    }
    if metadata:
        payload["metadata"] = metadata
    _atomic_write(archive_path, lambda file: torch.save(payload, file))
    return archive_path


def update_model_archive_metadata(
    path: Path, metadata_updates: dict[str, Any]
) -> None:
    """Atomically update metadata on an archived model without touching its weights."""
    if not path.is_file():
        raise FileNotFoundError(f"Archived model not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid archived model checkpoint: {path}")
    metadata = checkpoint.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid model archive metadata: {path}")
    metadata.update(metadata_updates)
    _atomic_write(path, lambda file: torch.save(checkpoint, file))


def archive_legacy_model_files(
    project_root: Path = PROJECT_ROOT,
    archive_directory: Path = ARCHIVED_MODELS_DIR,
) -> list[Path]:
    """Move legacy iteration checkpoints from the project root into the archive."""
    archive_directory.mkdir(parents=True, exist_ok=True)
    archived: list[Path] = []
    legacy_pattern = re.compile(
        r"^(?:best_model|model|checkpoint)_iter_.*\.pt$", re.IGNORECASE
    )
    for source in sorted(project_root.iterdir()):
        if not source.is_file() or not legacy_pattern.match(source.name):
            continue
        destination = archive_directory / source.name
        suffix = 1
        while destination.exists():
            destination = archive_directory / f"{source.stem}_legacy_{suffix}{source.suffix}"
            suffix += 1
        shutil.move(str(source), str(destination))
        archived.append(destination)
    return archived


def save_training_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            json.dump(state, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
