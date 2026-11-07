"""Load supervised examples from PGN files or pre-encoded NPZ files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

import chess
import chess.pgn
import numpy as np
from tqdm.auto import tqdm

from .encoding import ACTION_SIZE, BASE_BOARD_PLANES, BOARD_PLANES, encode_board, move_to_action

SupervisedExample: TypeAlias = tuple[np.ndarray, int, float]


def find_pgn_files(path: str | Path) -> list[Path]:
    """Return one PGN or all PGNs in a directory in deterministic filename order."""
    dataset_path = Path(path)
    if dataset_path.is_dir():
        pgn_files = sorted(
            (file for file in dataset_path.iterdir() if file.is_file() and file.suffix.lower() == ".pgn"),
            key=lambda file: file.name.casefold(),
        )
        if not pgn_files:
            raise ValueError(f"No .pgn files found in dataset directory: {dataset_path}")
        return pgn_files
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist or is not a file: {dataset_path}")
    if dataset_path.suffix.lower() != ".pgn":
        raise ValueError(f"Expected a .pgn file or directory, got: {dataset_path}")
    return [dataset_path]


def iter_pgn_dataset(
    path: str | Path, max_positions: int | None = None
) -> Iterator[SupervisedExample]:
    """Stream labeled positions from a PGN file or sorted directory of PGN files.

    Positions are buffered only for the current game because its final result is
    needed to create value targets. They are yielded and released before reading
    the next game. If supplied, max_positions limits positions across the entire
    file sequence.
    """
    if max_positions is not None and max_positions < 1:
        raise ValueError("max_positions must be a positive integer.")

    pgn_files = find_pgn_files(path)
    file_count = len(pgn_files)
    positions_yielded = 0
    for file_index, pgn_path in enumerate(pgn_files, start=1):
        tqdm.write(f"Processing {pgn_path.name} ({file_index}/{file_count})...")
        with pgn_path.open("r", encoding="utf-8", errors="replace") as pgn_file:
            game_number = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                game_number += 1
                if game.errors:
                    raise ValueError(
                        f"Invalid PGN game {game_number} in {pgn_path}: {game.errors[0]}"
                    )
                board = game.board()
                positions: list[tuple[np.ndarray, int, bool]] = []
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        raise ValueError(
                            f"Illegal move in PGN game {game_number} in {pgn_path}: {move}"
                        )
                    positions.append((encode_board(board), move_to_action(move), board.turn))
                    board.push(move)

                outcome = game.headers.get("Result", "*")
                winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(outcome)
                if outcome not in ("1-0", "0-1", "1/2-1/2", "*"):
                    raise ValueError(
                        f"Unsupported PGN result {outcome!r} in game {game_number} "
                        f"of {pgn_path}"
                    )
                for encoded, action, side_to_move in positions:
                    value = (
                        0.0
                        if winner is None
                        else (1.0 if winner == side_to_move else -1.0)
                    )
                    yield encoded, action, value
                    positions_yielded += 1
                    if max_positions is not None and positions_yielded >= max_positions:
                        return


def load_pgn_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize PGN positions for callers that explicitly need array data."""
    boards: list[np.ndarray] = []
    actions: list[int] = []
    values: list[float] = []
    for board, action, value in iter_pgn_dataset(path):
        boards.append(board)
        actions.append(action)
        values.append(value)

    if not boards:
        raise ValueError(f"No training positions found in PGN dataset: {path}")
    return (
        np.stack(boards),
        np.asarray(actions, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
    )


def load_npz_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load NPZ arrays with legacy 18-plane or history-aware 90-plane boards."""
    with np.load(path, allow_pickle=False) as dataset:
        required = {"boards", "policies", "values"}
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {', '.join(sorted(missing))}")
        boards = np.asarray(dataset["boards"], dtype=np.float32)
        policies = np.asarray(dataset["policies"])
        values = np.asarray(dataset["values"], dtype=np.float32)

    if boards.ndim != 4 or boards.shape[1:] not in (
        (BASE_BOARD_PLANES, 8, 8),
        (BOARD_PLANES, 8, 8),
    ):
        raise ValueError(
            f"NPZ boards must have shape (N,{BASE_BOARD_PLANES},8,8) "
            f"or (N,{BOARD_PLANES},8,8), got {boards.shape}"
        )
    if policies.ndim not in (1, 2):
        raise ValueError(f"NPZ policies must be one- or two-dimensional, got {policies.shape}")
    if values.shape != (len(boards),) or policies.shape[0] != len(boards):
        raise ValueError("NPZ boards, policies, and values must have matching sample counts.")
    if len(boards) == 0:
        raise ValueError("NPZ dataset must contain at least one sample.")
    if not np.isfinite(boards).all():
        raise ValueError("NPZ boards must contain only finite values.")
    if not np.isfinite(values).all() or np.any(values < -1) or np.any(values > 1):
        raise ValueError("NPZ values must be finite and in [-1, 1].")
    if boards.shape[1] == BASE_BOARD_PLANES:
        history_planes = np.zeros(
            (len(boards), BOARD_PLANES - BASE_BOARD_PLANES, 8, 8),
            dtype=np.float32,
        )
        boards = np.concatenate((boards, history_planes), axis=1)
    if policies.ndim == 1:
        if not np.issubdtype(policies.dtype, np.integer):
            if not np.isfinite(policies).all() or not np.equal(policies, np.floor(policies)).all():
                raise ValueError("NPZ action indices must be integers.")
        action_indices = policies.astype(np.int64)
        if np.any(action_indices < 0) or np.any(action_indices >= ACTION_SIZE):
            raise ValueError("NPZ action indices must be in [0, ACTION_SIZE).")
        return boards, action_indices, values
    if policies.ndim == 2 and policies.shape[1] == ACTION_SIZE:
        if np.any(policies < 0) or not np.all(np.isfinite(policies)):
            raise ValueError("Dense NPZ policies must be finite and non-negative.")
        row_sums = policies.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Every dense NPZ policy must have positive total probability.")
        return boards, policies.astype(np.float32) / row_sums[:, None], values
    raise ValueError(
        f"NPZ policies must have shape (N,) or (N,{ACTION_SIZE}), got {policies.shape}"
    )


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = Path(path).suffix.lower()
    if suffix == ".pgn":
        return load_pgn_dataset(path)
    if suffix == ".npz":
        return load_npz_dataset(path)
    raise ValueError(f"Unsupported dataset format {suffix!r}; use .pgn or .npz.")
