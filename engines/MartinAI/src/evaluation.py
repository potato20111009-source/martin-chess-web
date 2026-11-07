"""Run and persist Candidate-versus-Incumbent Arena games."""

from __future__ import annotations

import math
import multiprocessing
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty

import chess
import chess.pgn
import torch
from tqdm.auto import tqdm

from .mcts import (
    MCTS,
    RESIGN_CONSECUTIVE_EVALUATIONS,
    RESIGN_VALUE_THRESHOLD,
    _RustMCTS,
    sample_move_from_visits,
)
from .model_export import export_model_to_onnx
from .network import ChessNetwork

MIN_CANDIDATE_PROMOTION_SCORE_RATE = 0.55


def _maximum_possible_score_rate(
    wins: int, draws: int, games_played: int, scheduled_games: int
) -> float:
    if not 0 <= games_played <= scheduled_games or wins + draws > games_played:
        raise ValueError("Arena results have inconsistent game counts.")
    remaining = scheduled_games - games_played
    points = wins + 0.5 * draws
    return (points + remaining) / scheduled_games


@dataclass
class _ArenaGame:
    board: chess.Board
    game: chess.pgn.Game
    node: chess.pgn.GameNode
    candidate_color: chess.Color
    stalled: bool = False
    resignation_streaks: dict[chess.Color, int] = field(
        default_factory=lambda: {chess.WHITE: 0, chess.BLACK: 0}
    )
    resigned_by: chess.Color | None = None


def decide_candidate_promotion(
    score_rate: float,
    delta_elo: float,
    promotion_threshold: float = 0.55,
) -> tuple[bool, float, str]:
    """Promote only when the configured score threshold and Elo gain are met."""
    if not 0.0 <= score_rate <= 1.0:
        raise ValueError("score_rate must be in [0, 1].")
    if not 0.0 <= promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be in [0, 1].")
    effective_threshold = max(
        promotion_threshold, MIN_CANDIDATE_PROMOTION_SCORE_RATE
    )
    if not math.isfinite(delta_elo):
        raise ValueError("delta_elo must be finite.")
    if score_rate < effective_threshold:
        return False, effective_threshold, "below promotion threshold"
    if delta_elo <= 0.0:
        return False, effective_threshold, "non-positive Elo delta"
    return True, effective_threshold, "score threshold and positive Elo delta met"


def evaluate_candidate_vs_incumbent(
    candidate: ChessNetwork | None,
    incumbent: ChessNetwork | None,
    device: torch.device,
    iteration: int,
    games: int = 10,
    simulations: int = 100,
    max_moves: int = 400,
    pgn_directory: str | Path = "pgn_logs",
    promotion_threshold: float = 0.55,
    early_stop: bool = False,
    min_games: int = 5,
    opening_random_plies: int = 16,
    opening_temperature: float = 0.8,
    batch_size: int = 32,
    c_puct: float = 1.5,
    value_discount: float = 0.99,
    candidate_onnx_path: str | Path | None = None,
    incumbent_onnx_path: str | Path | None = None,
) -> dict[str, float | int]:
    """Compare candidate and incumbent in alternating colors using MCTS."""
    if games < 2 or games % 2:
        raise ValueError("Arena games must be an even number of at least 2.")
    if simulations < 1 or max_moves < 1:
        raise ValueError("Arena simulations and move limit must be positive.")
    if not 0.0 <= promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be in [0, 1].")
    if min_games < 1:
        raise ValueError("min_games must be positive.")
    if opening_random_plies < 0:
        raise ValueError("opening_random_plies must be non-negative.")
    if opening_temperature < 0:
        raise ValueError("opening_temperature must be non-negative.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    destination = Path(pgn_directory)
    destination.mkdir(parents=True, exist_ok=True)
    pgn_path = destination / f"arena_iteration_{iteration}.pgn"
    if (candidate_onnx_path is None) != (incumbent_onnx_path is None):
        raise ValueError("Both candidate and incumbent ONNX paths are required.")
    if candidate is None and candidate_onnx_path is None:
        raise ValueError("A candidate model or ONNX path is required.")
    if incumbent is None and incumbent_onnx_path is None:
        raise ValueError("An incumbent model or ONNX path is required.")
    if candidate is not None:
        candidate.eval()
    if incumbent is not None:
        incumbent.eval()
    candidate_search = MCTS(
        candidate,
        device,
        simulations=simulations,
        c_puct=c_puct,
        tactical_checks=False,
        value_discount=value_discount,
        rust_model_path=candidate_onnx_path,
    )
    incumbent_search = MCTS(
        incumbent,
        device,
        simulations=simulations,
        c_puct=c_puct,
        tactical_checks=False,
        value_discount=value_discount,
        rust_model_path=incumbent_onnx_path,
    )
    wins = losses = draws = 0
    games_played = 0

    with pgn_path.open("w", encoding="utf-8") as pgn_file, tqdm(
        range(games),
        desc="Martin Arena",
        unit="game",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
    ) as progress:
        game_offset = 0
        stop_early = False
        effective_threshold = max(
            promotion_threshold, MIN_CANDIDATE_PROMOTION_SCORE_RATE
        )
        while game_offset < games and not stop_early:
            games_before_early_stop_check = (
                max(1, min_games - games_played) if early_stop else batch_size
            )
            wave_size = min(
                batch_size,
                games - game_offset,
                games_before_early_stop_check,
                4,
            )
            arena_games: list[_ArenaGame] = []
            for offset in range(wave_size):
                game_index = game_offset + offset
                candidate_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
                game = chess.pgn.Game()
                game.headers["Event"] = (
                    f"Candidate vs Incumbent - Iteration {iteration}"
                )
                game.headers["Round"] = str(game_index + 1)
                game.headers["White"] = "Candidate" if candidate_color else "Incumbent"
                game.headers["Black"] = "Incumbent" if candidate_color else "Candidate"
                game.headers["CandidateColor"] = (
                    "White" if candidate_color else "Black"
                )
                arena_games.append(
                    _ArenaGame(chess.Board(), game, game, candidate_color)
                )

            active = list(arena_games)
            while active:
                candidate_turns = [
                    arena_game
                    for arena_game in active
                    if arena_game.board.turn == arena_game.candidate_color
                    and not arena_game.board.is_game_over(claim_draw=True)
                    and arena_game.resigned_by is None
                    and arena_game.board.ply() < max_moves
                ]
                incumbent_turns = [
                    arena_game
                    for arena_game in active
                    if arena_game.board.turn != arena_game.candidate_color
                    and not arena_game.board.is_game_over(claim_draw=True)
                    and arena_game.resigned_by is None
                    and arena_game.board.ply() < max_moves
                ]
                for turns, search in (
                    (candidate_turns, candidate_search),
                    (incumbent_turns, incumbent_search),
                ):
                    if not turns:
                        continue
                    results = search.search_batch(
                        [arena_game.board for arena_game in turns]
                    )
                    root_values = search.last_root_values
                    for turn_index, (arena_game, (visits, _)) in enumerate(
                        zip(turns, results)
                    ):
                        if not visits:
                            arena_game.stalled = True
                            continue
                        board = arena_game.board
                        side_to_move = board.turn
                        value = root_values[turn_index]
                        if (
                            value is not None
                            and value < RESIGN_VALUE_THRESHOLD
                        ):
                            arena_game.resignation_streaks[side_to_move] += 1
                        else:
                            arena_game.resignation_streaks[side_to_move] = 0
                        if (
                            arena_game.resignation_streaks[side_to_move]
                            >= RESIGN_CONSECUTIVE_EVALUATIONS
                        ):
                            arena_game.resigned_by = side_to_move
                            continue
                        temperature = (
                            opening_temperature
                            if board.ply() < opening_random_plies
                            else 0.0
                        )
                        move = sample_move_from_visits(visits, temperature)
                        arena_game.node = arena_game.node.add_variation(move)
                        board.push(move)
                active = [
                    arena_game
                    for arena_game in active
                    if not arena_game.stalled
                    and arena_game.resigned_by is None
                    and not arena_game.board.is_game_over(claim_draw=True)
                    and arena_game.board.ply() < max_moves
                ]

            for arena_game in arena_games:
                games_played += 1
                board = arena_game.board
                node = arena_game.node
                candidate_color = arena_game.candidate_color
                outcome = board.outcome(claim_draw=True)
                winner = (
                    not arena_game.resigned_by
                    if arena_game.resigned_by is not None
                    else None if outcome is None else outcome.winner
                )
                if arena_game.resigned_by is not None:
                    node.comment = (
                        f"{'White' if arena_game.resigned_by else 'Black'} "
                        "resigned after "
                        f"{RESIGN_CONSECUTIVE_EVALUATIONS} consecutive losing "
                        "value evaluations."
                    )
                if winner is None:
                    result_text = "1/2-1/2"
                    draws += 1
                    if board.ply() >= max_moves and outcome is None:
                        node.comment = "Draw adjudicated at configured move limit."
                elif winner == candidate_color:
                    result_text = "1-0" if candidate_color else "0-1"
                    wins += 1
                else:
                    result_text = "0-1" if candidate_color else "1-0"
                    losses += 1

                arena_game.game.headers["Result"] = result_text
                game_text = arena_game.game.accept(
                    chess.pgn.StringExporter(
                        headers=True, variations=False, comments=True
                    )
                )
                pgn_file.write(game_text + "\n\n")
                pgn_file.flush()
                progress.update(1)
                progress.set_postfix(W=wins, L=losses, D=draws)

            game_offset += wave_size
            maximum_score_rate = _maximum_possible_score_rate(
                wins, draws, games_played, games
            )
            cannot_promote = maximum_score_rate < effective_threshold
            configured_early_stop = (
                early_stop
                and games_played >= min_games
                and maximum_score_rate < promotion_threshold
            )
            if cannot_promote or configured_early_stop:
                progress.set_description(
                    f"Martin Arena (early stop; threshold {effective_threshold:.0%})"
                )
                tqdm.write(
                    "[Martin Arena] Candidate refused: even winning every "
                    f"remaining game caps score rate at {maximum_score_rate:.1%} "
                    f"(required {effective_threshold:.0%})."
                )
                stop_early = True

    score_rate = (wins + 0.5 * draws) / games_played
    summary: dict[str, float | int] = {
        "games": games_played,
        "scheduled_games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": score_rate,
        "stopped_early": int(games_played < games),
    }
    tqdm.write(
        f"[Martin Arena] Finished: score {score_rate:.1%} "
        f"({wins}W/{draws}D/{losses}L, {games_played}/{games} games); "
        f"PGN: {pgn_path.resolve()}"
    )
    return summary


def _isolated_arena_worker(
    result_queue: object,
    candidate_onnx_path: str,
    incumbent_onnx_path: str,
    settings: dict[str, object],
) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.chdir(Path(__file__).resolve().parents[1])
    try:
        result = evaluate_candidate_vs_incumbent(
            None,
            None,
            torch.device("cpu"),
            candidate_onnx_path=candidate_onnx_path,
            incumbent_onnx_path=incumbent_onnx_path,
            **settings,
        )
        result_queue.put(("done", result))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
        raise


def evaluate_candidate_vs_incumbent_isolated(
    candidate: ChessNetwork,
    incumbent: ChessNetwork,
    device: torch.device,
    *,
    iteration: int,
    games: int = 10,
    simulations: int = 100,
    max_moves: int = 400,
    pgn_directory: str | Path = "pgn_logs",
    promotion_threshold: float = 0.55,
    early_stop: bool = False,
    min_games: int = 5,
    opening_random_plies: int = 16,
    opening_temperature: float = 0.8,
    batch_size: int = 32,
    c_puct: float = 1.5,
    value_discount: float = 0.99,
) -> dict[str, float | int]:
    if _RustMCTS is None:
        raise RuntimeError(
            "Isolated Arena requires the Rust MCTS extension and ONNX Runtime."
        )
    output_directory = Path(__file__).resolve().parents[1] / "exports"
    candidate_onnx_path = export_model_to_onnx(
        candidate, output_directory / "model_current.onnx"
    )
    incumbent_onnx_path = export_model_to_onnx(
        incumbent, output_directory / "model_anchor.onnx"
    )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    settings: dict[str, object] = {
        "iteration": iteration,
        "games": games,
        "simulations": simulations,
        "max_moves": max_moves,
        "pgn_directory": str(Path(pgn_directory).resolve()),
        "promotion_threshold": promotion_threshold,
        "early_stop": early_stop,
        "min_games": min_games,
        "opening_random_plies": opening_random_plies,
        "opening_temperature": opening_temperature,
        "batch_size": batch_size,
        "c_puct": c_puct,
        "value_discount": value_discount,
    }
    process = context.Process(
        target=_isolated_arena_worker,
        args=(
            result_queue,
            str(candidate_onnx_path),
            str(incumbent_onnx_path),
            settings,
        ),
        name="martin-isolated-arena",
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
                        "Isolated Arena process exited without a result "
                        f"(exit code {process.exitcode})."
                    )
                continue
            kind, *payload = message
            if kind == "error":
                raise RuntimeError(f"Isolated Arena process failed:\n{payload[0]}")
            if kind != "done":
                raise RuntimeError(f"Unknown isolated Arena message: {kind!r}.")
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Isolated Arena process exited with code {process.exitcode}."
                )
            successful = True
            return payload[0]
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        result_queue.close()
        if successful:
            result_queue.join_thread()
        else:
            result_queue.cancel_join_thread()
