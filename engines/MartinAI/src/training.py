"""Supervised and self-play optimization routines."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import random
import sys
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from queue import Empty
import traceback

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data import SupervisedExample, iter_pgn_dataset
from .encoding import ACTION_SIZE, BOARD_PLANES, encode_board, move_to_action
from .mcts import (
    MCTS,
    RESIGN_CONSECUTIVE_EVALUATIONS,
    RESIGN_VALUE_THRESHOLD,
    VALUE_DISCOUNT,
    _RustMCTS,
)
from .model_export import export_model_to_onnx
from .network import ChessNetwork

VALUE_LOSS_WEIGHT = 2.0
RECENT_REPLAY_ITERATIONS = 5
RECENT_REPLAY_SAMPLE_FRACTION = 0.75
MAX_INFERENCE_POSITIONS_PER_WORKER = 16


def set_optimizer_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    """Set the learning rate on every optimizer parameter group."""
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be a finite positive number.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def reset_learning_rate_after_promotion(
    optimizer: Optimizer, promoted: bool, initial_learning_rate: float
) -> bool:
    """Restore the configured starting rate after loading a promoted checkpoint."""
    if not promoted:
        return False
    set_optimizer_learning_rate(optimizer, initial_learning_rate)
    return True


def _export_selfplay_model(model: ChessNetwork) -> Path:
    output_path = Path(__file__).resolve().parents[1] / "exports" / "model.onnx"
    return export_model_to_onnx(model, output_path)


def effective_selfplay_batch_size(requested_batch_size: int, workers: int) -> int:
    """Cap inference batching to the number of roots active across CPU workers."""
    if requested_batch_size < 1 or workers < 1:
        raise ValueError("Batch size and worker count must be positive.")
    return min(
        requested_batch_size,
        workers * MAX_INFERENCE_POSITIONS_PER_WORKER,
    )


@dataclass
class TrainingExample:
    board: np.ndarray
    policy: tuple[np.ndarray, np.ndarray]
    value: float
    iteration: int = 0


def _policy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        return F.cross_entropy(logits, targets.long())
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _sparse_policy_loss(
    logits: torch.Tensor, action_targets: torch.Tensor, probability_targets: torch.Tensor
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    selected_log_probabilities = log_probabilities.gather(1, action_targets)
    return -(selected_log_probabilities * probability_targets).sum(dim=1).mean()


def _validate_loss_weights(policy_weight: float, value_weight: float) -> None:
    if (
        not np.isfinite(policy_weight)
        or not np.isfinite(value_weight)
        or policy_weight < 0
        or value_weight < 0
        or policy_weight + value_weight == 0
    ):
        raise ValueError("Loss weights must be finite, non-negative, and not both zero.")


def self_play_temperature(
    move_number: int,
    switch_move: int = 20,
    greedy_after_move: int = 60,
    early_temperature: float = 1.0,
    late_temperature: float = 0.1,
) -> float:
    """Explore early, then play deterministically after 30 full moves."""
    if move_number < 1 or switch_move < 1:
        raise ValueError("Move numbers must be positive.")
    if greedy_after_move < switch_move:
        raise ValueError("greedy_after_move must be at least switch_move.")
    if early_temperature < 0 or late_temperature < 0:
        raise ValueError("Self-play temperatures must be non-negative.")
    if move_number > greedy_after_move:
        return 0.0
    return early_temperature if move_number <= switch_move else late_temperature


def update_resignation_streak(
    current_streak: int, value: float | None
) -> int:
    """Track consecutive low value estimates for the side to move."""
    if current_streak < 0:
        raise ValueError("Resignation streak cannot be negative.")
    if value is not None and value < RESIGN_VALUE_THRESHOLD:
        return current_streak + 1
    return 0


def should_resign(streak: int) -> bool:
    return streak >= RESIGN_CONSECUTIVE_EVALUATIONS


def _self_play_examples(
    positions: list[tuple[np.ndarray, np.ndarray, bool]],
    winner: chess.Color | None,
    terminal_move_pending: bool = True,
    value_discount: float = VALUE_DISCOUNT,
) -> list[TrainingExample]:
    last_index = len(positions) - 1
    terminal_distance = int(terminal_move_pending)
    return [
        TrainingExample(
            board=features,
            policy=(
                np.flatnonzero(policy).astype(np.int32),
                policy[policy > 0],
            ),
            value=(
                0.0
                if winner is None
                else (1.0 if winner == side else -1.0)
                * value_discount ** (last_index - index + terminal_distance)
            ),
        )
        for index, (features, policy, side) in enumerate(positions)
    ]


def sample_replay_examples(
    replay_buffer: list[TrainingExample],
    sample_size: int,
    current_iteration: int,
) -> list[TrainingExample]:
    """Prioritize recent experience while retaining a random sample of older data."""
    if sample_size < 1 or sample_size > len(replay_buffer):
        raise ValueError("sample_size must be between 1 and the replay buffer size.")
    if current_iteration < 0:
        raise ValueError("current_iteration must be non-negative.")

    recent_cutoff = max(0, current_iteration - RECENT_REPLAY_ITERATIONS + 1)
    recent = [
        example for example in replay_buffer if example.iteration >= recent_cutoff
    ]
    older = [example for example in replay_buffer if example.iteration < recent_cutoff]
    recent_count = min(
        len(recent),
        max(1, int(np.ceil(sample_size * RECENT_REPLAY_SAMPLE_FRACTION))),
    )
    older_count = min(len(older), sample_size - recent_count)
    recent_count += sample_size - recent_count - older_count

    samples = random.sample(recent, recent_count)
    if older_count:
        samples.extend(random.sample(older, older_count))
    random.shuffle(samples)
    return samples


def apply_temperature(policy: np.ndarray, temperature: float) -> np.ndarray:
    """Convert visit probabilities to a normalized action-sampling distribution."""
    if temperature < 0:
        raise ValueError("Temperature must be non-negative.")
    if policy.ndim != 1 or not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError("Policy must be a finite, non-negative one-dimensional array.")
    total = float(policy.sum())
    if total <= 0:
        raise ValueError("Policy must have positive total probability.")
    normalized = policy / total
    if temperature == 0:
        result = np.zeros_like(normalized)
        result[int(np.argmax(normalized))] = 1.0
        return result

    # Scaling before exponentiation keeps low-temperature policies numerically stable.
    scaled = normalized / float(normalized.max())
    adjusted = np.power(scaled, 1.0 / temperature)
    return adjusted / adjusted.sum()


def train_supervised(
    model: ChessNetwork,
    optimizer: Optimizer,
    boards: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 256,
    loss_components: dict[str, float] | None = None,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
) -> float:
    _validate_loss_weights(policy_loss_weight, value_loss_weight)
    policy_tensor = torch.from_numpy(np.asarray(policies))
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(boards, dtype=np.float32)),
        policy_tensor,
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
    )
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    model.train()
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    for _ in range(epochs):
        for features, targets, value_targets in loader:
            features = features.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            value_targets = value_targets.to(device, non_blocking=pin_memory)
            logits, predicted_values = model(features)
            policy_loss = _policy_loss(logits, targets)
            value_loss = F.mse_loss(predicted_values, value_targets)
            loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
    if not losses:
        raise ValueError("Supervised dataset is empty.")
    if loss_components is not None:
        loss_components.update(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
        )
    return float(np.mean(losses))


def train_supervised_stream(
    model: ChessNetwork,
    optimizer: Optimizer,
    dataset_path: str,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 256,
    max_positions: int | None = None,
    loss_components: dict[str, float] | None = None,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
) -> float:
    """Train from PGN positions in bounded batches instead of materializing a dataset."""
    _validate_loss_weights(policy_loss_weight, value_loss_weight)
    if max_positions is not None and max_positions < 1:
        raise ValueError("--max-positions must be a positive integer.")

    model.train()
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    def optimize_batch(batch: list[SupervisedExample]) -> None:
        pin_memory = device.type == "cuda"
        boards = torch.from_numpy(np.stack([item[0] for item in batch]))
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
        values = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        if pin_memory:
            boards = boards.pin_memory()
            actions = actions.pin_memory()
            values = values.pin_memory()
        boards = boards.to(device, non_blocking=pin_memory)
        actions = actions.to(device, non_blocking=pin_memory)
        values = values.to(device, non_blocking=pin_memory)
        logits, predicted_values = model(boards)
        policy_loss = F.cross_entropy(logits, actions)
        value_loss = F.mse_loss(predicted_values, values)
        loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

    for _ in range(epochs):
        batch: list[SupervisedExample] = []
        position_count = 0
        examples = iter_pgn_dataset(dataset_path, max_positions=max_positions)
        for example in tqdm(
            examples,
            desc="Martin PGN pre-training",
            unit="position",
            file=sys.stdout,
            leave=False,
        ):
            batch.append(example)
            position_count += 1
            if len(batch) >= batch_size:
                optimize_batch(batch)
                batch.clear()
        if batch:
            optimize_batch(batch)
        if position_count == 0:
            raise ValueError(f"No training positions found in PGN dataset: {dataset_path}")

    if loss_components is not None:
        loss_components.update(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
        )
    return float(np.mean(losses))


def play_self_play_game(
    model: ChessNetwork,
    device: torch.device,
    simulations: int,
    max_moves: int,
    progress_callback: Callable[[int], None] | None = None,
    tactical_checks: bool = False,
    value_discount: float = VALUE_DISCOUNT,
    temperature_switch_ply: int = 20,
    greedy_after_ply: int = 60,
    early_temperature: float = 1.0,
    late_temperature: float = 0.1,
) -> list[TrainingExample]:
    board = chess.Board()
    positions: list[tuple[np.ndarray, np.ndarray, bool]] = []
    search = MCTS(
        model,
        device,
        simulations=simulations,
        dirichlet_alpha=0.3,
        noise_fraction=0.3,
        tactical_checks=tactical_checks,
        value_discount=value_discount,
    )
    resignation_streaks = {chess.WHITE: 0, chess.BLACK: 0}
    resignation_winner: chess.Color | None = None
    while not board.is_game_over(claim_draw=True) and board.ply() < max_moves:
        _, policy = search.search(board, add_root_noise=True)
        if not np.any(policy):
            break
        legal_policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move in board.legal_moves:
            action = move_to_action(move)
            legal_policy[action] = policy[action]
        total = float(legal_policy.sum())
        if total <= 0:
            legal_actions = [move_to_action(move) for move in board.legal_moves]
            legal_policy[legal_actions] = 1.0 / len(legal_actions)
        else:
            legal_policy /= total

        positions.append((encode_board(board), legal_policy, board.turn))
        side_to_move = board.turn
        resignation_streaks[side_to_move] = update_resignation_streak(
            resignation_streaks[side_to_move], search.last_root_value
        )
        if should_resign(resignation_streaks[side_to_move]):
            resignation_winner = not side_to_move
            break
        sampling_policy = apply_temperature(
            legal_policy,
            self_play_temperature(
                board.ply() + 1,
                temperature_switch_ply,
                greedy_after_ply,
                early_temperature,
                late_temperature,
            ),
        )
        action = int(np.random.choice(ACTION_SIZE, p=sampling_policy))
        from .encoding import action_to_move

        move = action_to_move(action)
        if move not in board.legal_moves:
            raise RuntimeError(f"MCTS selected illegal move {move} in {board.fen()}")
        board.push(move)
        if progress_callback is not None:
            progress_callback(board.fullmove_number)

    if resignation_winner is not None:
        return _self_play_examples(
            positions,
            resignation_winner,
            terminal_move_pending=False,
            value_discount=value_discount,
        )
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return []
    return _self_play_examples(
        positions, outcome.winner, value_discount=value_discount
    )


def play_self_play_games_batched(
    model: ChessNetwork | None,
    device: torch.device,
    games: int,
    simulations: int,
    max_moves: int,
    inference_batch_size: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
    game_completed_callback: Callable[[], None] | None = None,
    inference_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    game_index_offset: int = 0,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_fraction: float = 0.3,
    value_discount: float = VALUE_DISCOUNT,
    temperature_switch_ply: int = 20,
    greedy_after_ply: int = 60,
    early_temperature: float = 1.0,
    late_temperature: float = 0.1,
    tactical_checks: bool = False,
    profile_timings: dict[str, float] | None = None,
) -> list[TrainingExample]:
    """Advance independent self-play games together and batch their MCTS inference."""
    if games < 1 or inference_batch_size < 1:
        raise ValueError("Game count and inference batch size must be positive.")
    if simulations < 1 or max_moves < 1:
        raise ValueError("Simulation count and move limit must be positive.")

    search = MCTS(
        model,
        device,
        simulations=simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        noise_fraction=noise_fraction,
        inference_fn=inference_fn,
        tactical_checks=tactical_checks,
        value_discount=value_discount,
    )
    boards: dict[int, chess.Board] = {}
    positions: dict[int, list[tuple[np.ndarray, np.ndarray, bool]]] = {}
    resignation_streaks: dict[int, dict[chess.Color, int]] = {}
    next_game = 0
    search_step = 0
    generated: list[TrainingExample] = []
    timings = profile_timings if profile_timings is not None else {}

    def collect_examples(*args: object, **kwargs: object) -> None:
        started = time.time()
        generated.extend(_self_play_examples(*args, **kwargs))
        timings["data_collection"] = (
            timings.get("data_collection", 0.0) + time.time() - started
        )

    def start_next_game() -> int | None:
        nonlocal next_game
        if next_game >= games:
            return None
        game_index = next_game
        next_game += 1
        boards[game_index] = chess.Board()
        positions[game_index] = []
        resignation_streaks[game_index] = {chess.WHITE: 0, chess.BLACK: 0}
        return game_index

    active = [
        game_index
        for game_index in (
            start_next_game() for _ in range(min(games, inference_batch_size))
        )
        if game_index is not None
    ]
    while active:
        search_step += 1
        batch_boards = [boards[game_index] for game_index in active]
        search_started = time.time()
        batch_results = search.search_batch(batch_boards, add_root_noise=True)
        search_elapsed = time.time() - search_started
        timings["mcts_search"] = (
            timings.get("mcts_search", 0.0) + search_elapsed
        )
        if inference_fn is None and search_step <= 3:
            tqdm.write(
                f"[Self-play Timing] Step {search_step} MCTS search "
                f"({len(batch_boards)} roots, "
                f"{search.simulations} simulations/root): {search_elapsed:.3f}s"
            )
        next_active: list[int] = []
        root_values = search.last_root_values
        step_started = time.time()
        collection_before = timings.get("data_collection", 0.0)
        for batch_index, (game_index, (visits, policy)) in enumerate(
            zip(active, batch_results)
        ):
            board = boards[game_index]
            if not visits or not np.any(policy):
                outcome = board.outcome(claim_draw=True)
                if outcome is not None:
                    collect_examples(
                        positions[game_index],
                        outcome.winner,
                        value_discount=value_discount,
                    )
                if game_completed_callback is not None:
                    game_completed_callback()
                continue

            legal_policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            for move in board.legal_moves:
                action = move_to_action(move)
                legal_policy[action] = policy[action]
            total = float(legal_policy.sum())
            if total <= 0:
                legal_actions = [move_to_action(move) for move in board.legal_moves]
                legal_policy[legal_actions] = 1.0 / len(legal_actions)
            else:
                legal_policy /= total
            encode_started = time.time()
            encoded_board = encode_board(board)
            timings["feature_encoding"] = (
                timings.get("feature_encoding", 0.0) + time.time() - encode_started
            )
            collect_started = time.time()
            positions[game_index].append((encoded_board, legal_policy, board.turn))
            timings["data_collection"] = (
                timings.get("data_collection", 0.0) + time.time() - collect_started
            )
            side_to_move = board.turn
            resignation_streaks[game_index][side_to_move] = update_resignation_streak(
                resignation_streaks[game_index][side_to_move],
                root_values[batch_index],
            )
            if should_resign(resignation_streaks[game_index][side_to_move]):
                collect_examples(
                    positions[game_index],
                    not side_to_move,
                    terminal_move_pending=False,
                    value_discount=value_discount,
                )
                if game_completed_callback is not None:
                    game_completed_callback()
                continue

            sampling_policy = apply_temperature(
                legal_policy,
                self_play_temperature(
                    board.ply() + 1,
                    temperature_switch_ply,
                    greedy_after_ply,
                    early_temperature,
                    late_temperature,
                ),
            )
            from .encoding import action_to_move

            action = int(np.random.choice(ACTION_SIZE, p=sampling_policy))
            move = action_to_move(action)
            if move not in board.legal_moves:
                raise RuntimeError(f"MCTS selected illegal move {move} in {board.fen()}")
            board.push(move)
            if progress_callback is not None:
                progress_callback(game_index + game_index_offset, board.fullmove_number)

            if board.is_game_over(claim_draw=True) or board.ply() >= max_moves:
                outcome = board.outcome(claim_draw=True)
                if outcome is not None:
                    collect_examples(
                        positions[game_index],
                        outcome.winner,
                        value_discount=value_discount,
                    )
                if game_completed_callback is not None:
                    game_completed_callback()
            else:
                next_active.append(game_index)

        collection_elapsed = (
            timings.get("data_collection", 0.0) - collection_before
        )
        timings["board_and_feature"] = (
            timings.get("board_and_feature", 0.0)
            + max(
                0.0,
                time.time() - step_started - collection_elapsed,
            )
        )
        while len(next_active) < inference_batch_size and next_game < games:
            replacement = start_next_game()
            if replacement is not None:
                next_active.append(replacement)
        active = next_active
    return generated


def _isolated_rust_selfplay_worker(
    result_queue: object,
    settings: dict[str, object],
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.chdir(Path(__file__).resolve().parents[1])

    def report_progress(game_index: int, move_number: int) -> None:
        result_queue.put(("progress", game_index, move_number))

    def report_game_completed() -> None:
        result_queue.put(("game_completed",))

    try:
        profile_timings: dict[str, float] = {}
        generated = play_self_play_games_batched(
            None,
            torch.device("cpu"),
            progress_callback=report_progress,
            game_completed_callback=report_game_completed,
            profile_timings=profile_timings,
            **settings,
        )
        result_queue.put(("done", generated, profile_timings))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
        raise


def play_self_play_games_isolated(
    *,
    games: int,
    simulations: int,
    max_moves: int,
    inference_batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
    game_completed_callback: Callable[[], None] | None = None,
    profile_timings: dict[str, float] | None = None,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_fraction: float = 0.3,
    value_discount: float = VALUE_DISCOUNT,
    temperature_switch_ply: int = 20,
    greedy_after_ply: int = 60,
    early_temperature: float = 1.0,
    late_temperature: float = 0.1,
    tactical_checks: bool = False,
) -> list[TrainingExample]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    settings: dict[str, object] = {
        "games": games,
        "simulations": simulations,
        "max_moves": max_moves,
        "inference_batch_size": inference_batch_size,
        "c_puct": c_puct,
        "dirichlet_alpha": dirichlet_alpha,
        "noise_fraction": noise_fraction,
        "value_discount": value_discount,
        "temperature_switch_ply": temperature_switch_ply,
        "greedy_after_ply": greedy_after_ply,
        "early_temperature": early_temperature,
        "late_temperature": late_temperature,
        "tactical_checks": tactical_checks,
    }
    process = context.Process(
        target=_isolated_rust_selfplay_worker,
        args=(result_queue, settings),
        name="martin-isolated-rust-selfplay",
    )
    successful = False
    process.start()
    try:
        while True:
            try:
                message = result_queue.get(timeout=0.1)
            except Empty:
                if process.exitcode is not None:
                    raise RuntimeError(
                        "Isolated Rust self-play process exited without a result "
                        f"(exit code {process.exitcode})."
                    )
                continue
            kind, *payload = message
            if kind == "progress":
                if progress_callback is not None:
                    progress_callback(int(payload[0]), int(payload[1]))
            elif kind == "game_completed":
                if game_completed_callback is not None:
                    game_completed_callback()
            elif kind == "error":
                raise RuntimeError(
                    f"Isolated Rust self-play process failed:\n{payload[0]}"
                )
            elif kind == "done":
                generated, child_timings = payload
                if profile_timings is not None:
                    for name, elapsed in child_timings.items():
                        profile_timings[name] = (
                            profile_timings.get(name, 0.0) + float(elapsed)
                        )
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(
                        "Isolated Rust self-play process exited with "
                        f"code {process.exitcode}."
                    )
                successful = True
                return generated
            else:
                raise RuntimeError(
                    f"Unknown isolated self-play message: {kind!r}."
                )
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        result_queue.close()
        if successful:
            result_queue.join_thread()
        else:
            result_queue.cancel_join_thread()


def _self_play_worker(
    worker_id: int,
    game_count: int,
    game_index_offset: int,
    inference_batch_size: int,
    simulations: int,
    max_moves: int,
    request_queue: object,
    response_queue: object,
    result_queue: object,
    c_puct: float,
    dirichlet_alpha: float,
    noise_fraction: float,
    value_discount: float,
    temperature_switch_ply: int,
    greedy_after_ply: int,
    early_temperature: float,
    late_temperature: float,
    tactical_checks: bool,
) -> None:
    """Run CPU-only self-play and request all neural evaluations from the parent."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    request_index = 0

    def infer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal request_index
        request_id = request_index
        request_index += 1
        compact_features = np.ascontiguousarray(features, dtype=np.uint8).reshape(
            features.shape[0], -1
        )
        request_queue.put((worker_id, request_id, compact_features))
        response_id, probabilities, values = response_queue.get()
        if response_id != request_id:
            raise RuntimeError(
                f"Worker {worker_id} received response {response_id} "
                f"for inference request {request_id}."
            )
        return probabilities, values

    try:
        profile_timings: dict[str, float] = {}
        generated = play_self_play_games_batched(
            None,
            torch.device("cpu"),
            games=game_count,
            simulations=simulations,
            max_moves=max_moves,
            inference_batch_size=inference_batch_size,
            progress_callback=lambda game_index, move_number: result_queue.put(
                ("progress", worker_id, game_index, move_number)
            ),
            game_completed_callback=lambda: result_queue.put(
                ("game_completed", worker_id)
            ),
            inference_fn=infer,
            game_index_offset=game_index_offset,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            noise_fraction=noise_fraction,
            value_discount=value_discount,
            temperature_switch_ply=temperature_switch_ply,
            greedy_after_ply=greedy_after_ply,
            early_temperature=early_temperature,
            late_temperature=late_temperature,
            tactical_checks=tactical_checks,
            profile_timings=profile_timings,
        )
        result_queue.put(("done", worker_id, generated, profile_timings))
    except Exception:
        result_queue.put(("error", worker_id, traceback.format_exc()))
        raise


def play_self_play_games_multiprocess(
    model: ChessNetwork,
    device: torch.device,
    games: int,
    simulations: int,
    max_moves: int,
    inference_batch_size: int = 32,
    workers: int = 4,
    progress_callback: Callable[[int, int], None] | None = None,
    game_completed_callback: Callable[[], None] | None = None,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_fraction: float = 0.3,
    value_discount: float = VALUE_DISCOUNT,
    temperature_switch_ply: int = 20,
    greedy_after_ply: int = 60,
    early_temperature: float = 1.0,
    late_temperature: float = 0.1,
    tactical_checks: bool = False,
    profile_timings: dict[str, float] | None = None,
) -> list[TrainingExample]:
    """Run CPU MCTS workers with a centralized, batched model-inference server."""
    if games < 1 or inference_batch_size < 1 or workers < 1:
        raise ValueError("Game count, inference batch size, and workers must be positive.")
    if simulations < 1 or max_moves < 1:
        raise ValueError("Simulation count and move limit must be positive.")

    worker_count = min(workers, games, inference_batch_size)
    effective_batch_size = effective_selfplay_batch_size(
        inference_batch_size, worker_count
    )
    if effective_batch_size < inference_batch_size:
        tqdm.write(
            "[Martin Training] Warning: self-play inference batch "
            f"{inference_batch_size} exceeds "
            f"{MAX_INFERENCE_POSITIONS_PER_WORKER} positions per worker "
            f"({worker_count} workers); reducing effective batch size to "
            f"{effective_batch_size}."
        )
    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    response_queues = [context.Queue() for _ in range(worker_count)]
    processes: list[multiprocessing.Process] = []
    completed: set[int] = set()
    generated: list[TrainingExample] = []
    search = MCTS(model, device, simulations=1)
    model.eval()
    successful = False

    def process_messages() -> None:
        while True:
            try:
                message = result_queue.get_nowait()
            except Empty:
                return
            kind, worker_id, *payload = message
            if kind == "progress":
                if progress_callback is not None:
                    progress_callback(int(payload[0]), int(payload[1]))
            elif kind == "game_completed":
                if game_completed_callback is not None:
                    game_completed_callback()
            elif kind == "done":
                generated.extend(payload[0])
                if profile_timings is not None:
                    for name, elapsed in payload[1].items():
                        profile_timings[name] = (
                            profile_timings.get(name, 0.0) + float(elapsed)
                        )
                completed.add(worker_id)
            elif kind == "error":
                raise RuntimeError(
                    f"Self-play worker {worker_id} failed:\n{payload[0]}"
                )
            else:
                raise RuntimeError(f"Unknown self-play worker message: {kind!r}.")

    def evaluate_pending(
        pending: list[tuple[int, int, np.ndarray]],
    ) -> None:
        if not pending:
            return
        lengths = [features.shape[0] for _, _, features in pending]
        compact_batch = np.concatenate(
            [features for _, _, features in pending], axis=0
        )
        feature_batch = compact_batch.reshape(
            -1, BOARD_PLANES, 8, 8
        ).astype(np.float32, copy=False)
        probabilities, values = search._predict_features(feature_batch)
        offset = 0
        for (worker_id, request_id, _), length in zip(pending, lengths):
            end = offset + length
            response_queues[worker_id].put(
                (
                    request_id,
                    probabilities[offset:end].astype(np.float16),
                    values[offset:end],
                )
            )
            offset = end

    try:
        game_offset = 0
        per_worker_batch = max(1, effective_batch_size // worker_count)
        for worker_id in range(worker_count):
            worker_games = games // worker_count + int(worker_id < games % worker_count)
            process = context.Process(
                target=_self_play_worker,
                args=(
                    worker_id,
                    worker_games,
                    game_offset,
                    min(per_worker_batch, worker_games),
                    simulations,
                    max_moves,
                    request_queue,
                    response_queues[worker_id],
                    result_queue,
                    c_puct,
                    dirichlet_alpha,
                    noise_fraction,
                    value_discount,
                    temperature_switch_ply,
                    greedy_after_ply,
                    early_temperature,
                    late_temperature,
                    tactical_checks,
                ),
                name=f"martin-selfplay-{worker_id}",
            )
            process.start()
            processes.append(process)
            game_offset += worker_games

        while len(completed) < worker_count:
            try:
                first = request_queue.get(timeout=0.0001)
            except Empty:
                process_messages()
                for worker_id, process in enumerate(processes):
                    if process.exitcode not in (None, 0) and worker_id not in completed:
                        raise RuntimeError(
                            f"Self-play worker {worker_id} exited with "
                            f"code {process.exitcode}."
                        )
                continue

            pending = [(first[0], first[1], first[2])]
            while True:
                try:
                    request = request_queue.get_nowait()
                except Empty:
                    break
                pending.append((request[0], request[1], request[2]))
            evaluate_pending(pending)
            process_messages()

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Self-play worker {process.name} exited with "
                    f"code {process.exitcode}."
                )
        process_messages()
        successful = True
        return generated
    finally:
        if not successful:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join()
        for queue in [request_queue, result_queue, *response_queues]:
            if not successful:
                queue.cancel_join_thread()
            queue.close()
            if successful:
                queue.join_thread()


def train_self_play(
    model: ChessNetwork,
    optimizer: Optimizer,
    replay_buffer: list[TrainingExample],
    device: torch.device,
    games: int = 4,
    simulations: int = 64,
    max_moves: int = 400,
    epochs: int = 1,
    batch_size: int = 256,
    max_buffer_size: int = 25_000,
    iteration: int = 0,
    inference_batch_size: int = 32,
    workers: int = 8,
    loss_components: dict[str, float] | None = None,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    noise_fraction: float = 0.3,
    value_discount: float = VALUE_DISCOUNT,
    temperature_switch_ply: int = 20,
    greedy_after_ply: int = 60,
    early_temperature: float = 1.0,
    late_temperature: float = 0.1,
    tactical_checks: bool = False,
    profile_timings: dict[str, float] | None = None,
) -> tuple[float, list[TrainingExample]]:
    _validate_loss_weights(policy_loss_weight, value_loss_weight)
    if max_buffer_size < 1:
        raise ValueError("max_buffer_size must be a positive integer.")
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    for example in replay_buffer:
        if example.board.shape != (BOARD_PLANES, 8, 8):
            raise ValueError(
                "Replay examples must use history-aware board tensors with shape "
                f"({BOARD_PLANES}, 8, 8)."
            )
    if workers < 1:
        raise ValueError("workers must be a positive integer.")
    generated: list[TrainingExample] = []
    timings = profile_timings if profile_timings is not None else {}
    with tqdm(
        total=games,
        desc="Martin Self-play",
        unit="game",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
    ) as progress:
        callbacks = {
            "progress_callback": lambda game_index, move_number: progress.set_postfix_str(
                f"Game {game_index + 1}/{games} [Move {move_number}]",
                refresh=False,
            ),
            "game_completed_callback": lambda: progress.update(1),
        }
        if device.type == "cuda" and _RustMCTS is not None:
            onnx_path = _export_selfplay_model(model)
            tqdm.write(
                f"[MCTS] Running isolated Rust/ONNX self-play using {onnx_path}."
            )
            if workers != 1:
                tqdm.write(
                    "[Martin Training] Rust/ONNX isolation uses one batched "
                    "self-play process; --selfplay-workers is ignored."
                )
            generated = play_self_play_games_isolated(
                games=games,
                simulations=simulations,
                max_moves=max_moves,
                inference_batch_size=inference_batch_size,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                noise_fraction=noise_fraction,
                value_discount=value_discount,
                temperature_switch_ply=temperature_switch_ply,
                greedy_after_ply=greedy_after_ply,
                early_temperature=early_temperature,
                late_temperature=late_temperature,
                tactical_checks=tactical_checks,
                profile_timings=timings,
                **callbacks,
            )
        elif workers == 1:
            tqdm.write(
                f"[MCTS] Running single-process batch search on device: {device}"
            )
            generated = play_self_play_games_batched(
                model,
                device,
                games=games,
                simulations=simulations,
                max_moves=max_moves,
                inference_batch_size=inference_batch_size,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                noise_fraction=noise_fraction,
                value_discount=value_discount,
                temperature_switch_ply=temperature_switch_ply,
                greedy_after_ply=greedy_after_ply,
                early_temperature=early_temperature,
                late_temperature=late_temperature,
                tactical_checks=tactical_checks,
                **callbacks,
                profile_timings=timings,
            )
        else:
            generated = play_self_play_games_multiprocess(
                model,
                device,
                games=games,
                simulations=simulations,
                max_moves=max_moves,
                inference_batch_size=inference_batch_size,
                workers=workers,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                noise_fraction=noise_fraction,
                value_discount=value_discount,
                temperature_switch_ply=temperature_switch_ply,
                greedy_after_ply=greedy_after_ply,
                early_temperature=early_temperature,
                late_temperature=late_temperature,
                tactical_checks=tactical_checks,
                **callbacks,
                profile_timings=timings,
            )
    tqdm.write(
        "[Self-play Timing] "
        f"Rust/Python MCTS search: {timings.get('mcts_search', 0.0):.3f}s | "
        "board updates + feature encoding: "
        f"{timings.get('board_and_feature', 0.0):.3f}s "
        f"(encoding {timings.get('feature_encoding', 0.0):.3f}s) | "
        f"training data collection: {timings.get('data_collection', 0.0):.3f}s"
    )
    for example in generated:
        example.iteration = iteration
    replay_write_started = time.time()
    replay_buffer.extend(generated)
    deduplicate_replay_buffer(replay_buffer)
    if len(replay_buffer) > max_buffer_size:
        del replay_buffer[: len(replay_buffer) - max_buffer_size]
    timings["replay_buffer_write"] = time.time() - replay_write_started
    tqdm.write(
        "[Self-play Timing] Replay buffer write/dedup/trim: "
        f"{timings['replay_buffer_write']:.3f}s"
    )
    if not replay_buffer:
        raise RuntimeError("Self-play produced no training positions.")

    sample_size = min(len(replay_buffer), max(batch_size, batch_size * 32))
    samples = sample_replay_examples(replay_buffer, sample_size, iteration)
    boards = torch.from_numpy(np.stack([item.board for item in samples])).float()
    max_policy_size = max(len(item.policy[0]) for item in samples)
    action_targets = np.zeros((len(samples), max_policy_size), dtype=np.int64)
    probability_targets = np.zeros((len(samples), max_policy_size), dtype=np.float32)
    for index, item in enumerate(samples):
        actions, probabilities = item.policy
        action_targets[index, : len(actions)] = actions
        probability_targets[index, : len(probabilities)] = probabilities
    values = torch.tensor([item.value for item in samples], dtype=torch.float32)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        TensorDataset(
            boards,
            torch.from_numpy(action_targets),
            torch.from_numpy(probability_targets),
            values,
        ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )

    model.train()
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    for _ in range(epochs):
        for features, policy_targets, policy_probabilities, value_targets in loader:
            features = features.to(device, non_blocking=pin_memory)
            policy_targets = policy_targets.to(device, non_blocking=pin_memory)
            policy_probabilities = policy_probabilities.to(device, non_blocking=pin_memory)
            value_targets = value_targets.to(device, non_blocking=pin_memory)
            logits, predicted_values = model(features)
            policy_loss = _sparse_policy_loss(
                logits, policy_targets, policy_probabilities
            )
            value_loss = F.mse_loss(predicted_values, value_targets)
            loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
    if loss_components is not None:
        loss_components.update(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
        )
    return float(np.mean(losses)), generated


def deduplicate_replay_buffer(
    replay_buffer: list[TrainingExample],
) -> int:
    """Keep only the newest example for each encoded board position."""
    seen_positions: set[bytes] = set()
    unique_reversed: list[TrainingExample] = []
    for example in reversed(replay_buffer):
        key = hashlib.blake2b(example.board.tobytes(), digest_size=16).digest()
        if key not in seen_positions:
            seen_positions.add(key)
            unique_reversed.append(example)
    unique_reversed.reverse()
    removed = len(replay_buffer) - len(unique_reversed)
    replay_buffer[:] = unique_reversed
    return removed


def discard_oldest_replay_fraction(
    replay_buffer: list[TrainingExample], fraction: float = 0.5
) -> int:
    """Discard the oldest requested fraction of FIFO replay data."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Replay discard fraction must be in [0, 1].")
    removed = int(np.ceil(len(replay_buffer) * fraction))
    if removed:
        del replay_buffer[:removed]
    return removed
